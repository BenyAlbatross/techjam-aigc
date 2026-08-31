"""Resolution-normalized variant: both classes center-cropped square and resized alike.

Removes the size shortcut (AUC 1.000 -> ~0.58) at the cost of high-frequency detail.
"""
import argparse, io, os, pickle
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

SIZE = 200
QUALITY = 92


def render(args):
    path, = args,
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2,
                  (im.width + s) // 2, (im.height + s) // 2))
    im = im.resize((SIZE, SIZE), Image.BICUBIC)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=QUALITY)
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    index = pickle.load(open(os.path.join(a.work, "index.pkl"), "rb"))
    rows = []
    for source, items in sorted(index.items()):
        for orig, path, _sz, label in items:
            rows.append((orig, path, label, source))

    with ProcessPoolExecutor() as ex:
        blobs = list(ex.map(render, [r[1] for r in rows], chunksize=64))

    recs = {"image": [], "label": [], "source": [], "orig_path": [], "id": []}
    for (orig, _p, label, source), blob in zip(rows, blobs):
        recs["image"].append({"bytes": blob, "path": None})
        recs["label"].append(label)
        recs["source"].append(source)
        recs["orig_path"].append(orig)
        recs["id"].append(f"{source}/{os.path.basename(orig)}")
    pickle.dump(recs, open(a.out, "wb"))
    print(f"normalized {len(rows)} images -> {a.out} "
          f"({sum(len(b) for b in blobs)/1e6:.0f} MB of JPEG)")


if __name__ == "__main__":
    main()
