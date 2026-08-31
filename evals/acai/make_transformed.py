"""Build a *_transformed variant of an existing config by applying robustness transforms."""
import argparse, io, os, pickle, sys
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acai import transforms as T

STORE_Q = 95  # high enough that storage adds little beyond the assigned transform


def _one(job):
    i, src, outdir, seed = job
    chain, primary, n = T.plan(i, seed)
    try:
        im = Image.open(io.BytesIO(src)) if isinstance(src, bytes) else Image.open(src)
        im = T.apply(im, chain, i, seed)
    except Exception:
        return None
    dst = os.path.join(outdir, f"{i:07d}.jpg")
    im.save(dst, "JPEG", quality=STORE_Q)
    return dst, "|".join(T.label(f, p) for f, p in chain), primary, n


def load_source(work, config):
    """Return parallel lists (image_source, label, source, orig_path, id)."""
    if config == "default":
        index = pickle.load(open(os.path.join(work, "index.pkl"), "rb"))
        rows = []
        for source, items in sorted(index.items()):
            for orig, path, _sz, lab in items:
                rows.append((path, lab, source, orig,
                             f"{source}/{os.path.basename(orig)}"))
        return rows
    r = pickle.load(open(os.path.join(work, f"{config}.pkl"), "rb"))
    return [(r["image"][i]["bytes"], r["label"][i], r["source"][i],
             r["orig_path"][i], r["id"][i]) for i in range(len(r["label"]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load_source(a.work, a.config)
    outdir = os.path.join(a.work, "tr", a.config)
    os.makedirs(outdir, exist_ok=True)
    jobs = [(i, rows[i][0], outdir, a.seed) for i in range(len(rows))]

    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_one, jobs, chunksize=16))

    recs = {"image": [], "label": [], "source": [], "orig_path": [], "id": [],
            "transform_chain": [], "primary_transform": [], "n_transforms": []}
    dropped = 0
    for (src, lab, source, orig, rid), out in zip(rows, res):
        if out is None:
            dropped += 1
            continue
        dst, chain, primary, n = out
        recs["image"].append(dst)
        recs["label"].append(lab)
        recs["source"].append(source)
        recs["orig_path"].append(orig)
        recs["id"].append(rid)
        recs["transform_chain"].append(chain)
        recs["primary_transform"].append(primary)
        recs["n_transforms"].append(n)

    out_pkl = os.path.join(a.work, f"{a.config}_transformed.pkl")
    pickle.dump(recs, open(out_pkl, "wb"))
    mb = sum(os.path.getsize(p) for p in recs["image"]) / 1e6
    print(f"{a.config}_transformed: {len(recs['image'])} images, {mb:.0f} MB"
          + (f", dropped {dropped}" if dropped else ""))


if __name__ == "__main__":
    main()
