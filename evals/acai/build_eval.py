"""Build the WildFake validation/eval subset and stage it as parquet shards.

Subset (fixed by the competition spec):
  Non-AIGC : COCO val2017      4998 images  -> label 0
  AIGC     : DALL-E Advanced   8843 images  -> label 1  (WildFake DALLE3, IsAdvanced=1)

Both sets sit in contiguous regions of their source archives, so we range-fetch
one span per archive rather than pulling the full 28GB.
"""
import argparse, csv, io, json, os, pickle, struct, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acai.ziphttp import RemoteZip, modelscope_resolver

DATASET = "hy2628982280/WildFake"
CHUNK = 8 << 20
WORKERS = 12

SUBSETS = {
    "coco_val2017": dict(
        zip_path="Images/Real/coco.zip", csv="real_coco.csv", strip="./Real/",
        keep=lambda p: "/val2017/" in p, label=0, expect=4998),
    "dalle3_advanced": dict(
        zip_path="Images/Diffusion_based/DALLE.zip", csv="dalle3.csv",
        strip="./Diffusion_based/", keep=lambda p: True, label=1, expect=8843),
}


def manifest(work, cfg):
    rows = []
    with open(os.path.join(work, cfg["csv"]), newline="") as f:
        for r in csv.DictReader(f):
            p = r["Image_path"]
            if cfg["keep"](p):
                rows.append((p, p[len(cfg["strip"]):]))
    assert len(rows) == cfg["expect"], f"{cfg['csv']}: {len(rows)} != {cfg['expect']}"
    return rows


def fetch_span(rz, start, end, dest, workers=WORKERS):
    """Parallel, resumable range download of [start, end) into dest.

    Each CDN connection is bandwidth-capped, so we keep several ranges in flight
    and record finished chunks in a sidecar file so an interrupted run resumes.
    """
    total = end - start
    nchunks = (total + CHUNK - 1) // CHUNK
    marks = dest + ".done"
    done = set(json.load(open(marks))) if os.path.exists(marks) else set()
    if not os.path.exists(dest) or os.path.getsize(dest) != total:
        with open(dest, "wb") as f:
            f.truncate(total)
        done = set()

    todo = [i for i in range(nchunks) if i not in done]
    if not todo:
        return
    t0 = time.time()
    lock = threading.Lock()
    fetched = [0]

    def work(i):
        off = i * CHUNK
        n = min(CHUNK, total - off)
        buf = rz.get(start + off, n)
        with lock:
            with open(dest, "r+b") as f:
                f.seek(off)
                f.write(buf)
            done.add(i)
            fetched[0] += n
            if len(done) % 8 == 0 or len(done) == nchunks:
                json.dump(sorted(done), open(marks, "w"))
                el = time.time() - t0
                print(f"    {len(done)*CHUNK/1e9:.2f}/{total/1e9:.2f} GB  "
                      f"{fetched[0]/1e6/max(el, 1e-9):.1f} MB/s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    json.dump(sorted(done), open(marks, "w"))


def extract(span_path, span_start, entries, outdir):
    """Pull each member out of the downloaded span."""
    os.makedirs(outdir, exist_ok=True)
    written = []
    with open(span_path, "rb") as f:
        for zname, ent in entries:
            f.seek(ent["lho"] - span_start)
            hdr = f.read(30)
            if hdr[:4] != b"PK\x03\x04":
                raise ValueError(f"bad local header for {zname}")
            nlen, elen = struct.unpack("<HH", hdr[26:30])
            f.seek(nlen + elen, os.SEEK_CUR)
            raw = f.read(ent["csize"])
            data = zlib.decompress(raw, -15) if ent["method"] == 8 else raw
            if len(data) != ent["usize"]:
                raise ValueError(f"size mismatch {zname}")
            if zlib.crc32(data) != ent["crc"]:
                raise ValueError(f"CRC mismatch {zname}")
            dst = os.path.join(outdir, zname.replace("/", "__"))
            with open(dst, "wb") as g:
                g.write(data)
            written.append((zname, dst, len(data)))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    work = args.work
    os.makedirs(work, exist_ok=True)

    index = {}
    for name, cfg in SUBSETS.items():
        print(f"[{name}] resolving archive ...", flush=True)
        rz = RemoteZip(modelscope_resolver(DATASET, cfg["zip_path"]))
        cd_pkl = os.path.join(work, f"{name}_cd.pkl")
        if os.path.exists(cd_pkl):
            cd = pickle.load(open(cd_pkl, "rb"))
        else:
            cd = {e["name"]: e for e in rz.entries()}
            pickle.dump(cd, open(cd_pkl, "wb"))

        rows = manifest(work, cfg)
        ents = [(z, cd[z]) for _, z in rows]
        start = min(e["lho"] for _, e in ents)
        end = max(e["lho"] + e["csize"] + 30 + len(z) + 4096 for z, e in ents)
        end = min(end, rz.size)
        print(f"[{name}] {len(ents)} files, span {(end-start)/1e9:.2f} GB", flush=True)

        span = os.path.join(work, f"{name}.span")
        fetch_span(rz, start, end, span)
        print(f"[{name}] extracting ...", flush=True)
        out = extract(span, start, ents, os.path.join(work, "img", name))
        print(f"[{name}] extracted {len(out)} files", flush=True)
        index[name] = [(orig, dst, sz, cfg["label"])
                       for (orig, _), (zn, dst, sz) in zip(rows, out)]
        os.remove(span)
        if os.path.exists(span + '.done'):
            os.remove(span + '.done')

    pickle.dump(index, open(os.path.join(work, "index.pkl"), "wb"))
    tot = sum(len(v) for v in index.values())
    print(f"DONE: {tot} images staged")


if __name__ == "__main__":
    main()
