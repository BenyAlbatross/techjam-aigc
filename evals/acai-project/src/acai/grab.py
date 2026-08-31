"""Range-extract a contiguous run of members from a remote WildFake zip.

Members inside these archives are laid out in directory order, so taking a
contiguous run keeps the download to a single byte range instead of thousands
of scattered requests.
"""
import argparse, io, json, os, pickle, struct, sys, threading, time, zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acai.ziphttp import RemoteZip, modelscope_resolver

DS = "hy2628982280/WildFake"
CHUNK = 8 << 20
WORKERS = 12
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def cd_cached(work, tag, zip_path):
    p = os.path.join(work, f"cd_{tag}.pkl")
    rz = RemoteZip(modelscope_resolver(DS, zip_path))
    if os.path.exists(p):
        return rz, pickle.load(open(p, "rb"))
    ents = rz.entries()
    pickle.dump(ents, open(p, "wb"))
    return rz, ents


def fetch_span(rz, start, end, dest):
    total = end - start
    n = (total + CHUNK - 1) // CHUNK
    marks = dest + ".done"
    done = set(json.load(open(marks))) if os.path.exists(marks) else set()
    if not os.path.exists(dest) or os.path.getsize(dest) != total:
        with open(dest, "wb") as f:
            f.truncate(total)
        done = set()
    todo = [i for i in range(n) if i not in done]
    if not todo:
        return
    lock, t0, got = threading.Lock(), time.time(), [0]

    def work(i):
        off = i * CHUNK
        ln = min(CHUNK, total - off)
        buf = rz.get(start + off, ln)
        with lock:
            with open(dest, "r+b") as f:
                f.seek(off); f.write(buf)
            done.add(i); got[0] += ln
            if len(done) % 8 == 0 or len(done) == n:
                json.dump(sorted(done), open(marks, "w"))
                print(f"    {len(done)*CHUNK/1e9:.2f}/{total/1e9:.2f} GB "
                      f"{got[0]/1e6/max(time.time()-t0,1e-9):.1f} MB/s", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, todo))
    json.dump(sorted(done), open(marks, "w"))


def extract(span, start, ents, outdir):
    os.makedirs(outdir, exist_ok=True)
    out = []
    with open(span, "rb") as f:
        for e in ents:
            f.seek(e["lho"] - start)
            h = f.read(30)
            if h[:4] != b"PK\x03\x04":
                continue
            nl, el = struct.unpack("<HH", h[26:30])
            f.seek(nl + el, os.SEEK_CUR)
            raw = f.read(e["csize"])
            try:
                data = zlib.decompress(raw, -15) if e["method"] == 8 else raw
            except zlib.error:
                continue
            if zlib.crc32(data) != e["crc"]:
                continue
            dst = os.path.join(outdir, e["name"].replace("/", "__"))
            with open(dst, "wb") as g:
                g.write(data)
            out.append((e["name"], dst))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--skip", type=int, default=0)
    a = ap.parse_args()

    rz, ents = cd_cached(a.work, a.tag, a.zip)
    pool = [e for e in ents
            if e["usize"] > 0 and e["name"].lower().endswith(IMG_EXT)
            and e["name"].startswith(a.prefix)]
    pool.sort(key=lambda e: e["lho"])
    print(f"[{a.tag}] {len(pool)} candidate members under {a.prefix!r}", flush=True)
    take = pool[a.skip:a.skip + a.count]
    if not take:
        raise SystemExit(f"[{a.tag}] nothing matched")
    start = min(e["lho"] for e in take)
    end = min(max(e["lho"] + e["csize"] + 30 + len(e["name"]) + 4096 for e in take), rz.size)
    print(f"[{a.tag}] taking {len(take)} members, span {(end-start)/1e9:.2f} GB", flush=True)

    span = os.path.join(a.work, f"{a.tag}.span")
    fetch_span(rz, start, end, span)
    out = extract(span, start, take, os.path.join(a.work, "img", a.tag))
    os.remove(span)
    if os.path.exists(span + ".done"):
        os.remove(span + ".done")
    pickle.dump(out, open(os.path.join(a.work, f"{a.tag}_files.pkl"), "wb"))
    print(f"[{a.tag}] extracted {len(out)} files", flush=True)


if __name__ == "__main__":
    main()
