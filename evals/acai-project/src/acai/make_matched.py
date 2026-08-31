"""Build a resolution-matched config: every image center-cropped square and resized alike.

Native-resolution pairing is not possible here (LAION clusters at 800x800, DALL-E 3 at
1024x1024 - only 66 exact matches across 14k samples), so we instead restrict both classes
to natively-large images and put them through one identical downscale. That keeps far more
high-frequency detail than the 200x200 `normalized` config while removing size as a cue.
"""
import argparse, glob, io, json, os, pickle, random
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

W = None
SIZE = 512
QUALITY = 92


def maxdim(p):
    try:
        with Image.open(p) as im:
            return max(im.size)
    except Exception:
        return 0


def render(p):
    """Returns None for images that fail to decode.

    A few upstream files are truncated; they pass the zip CRC (so the bytes match the
    archive exactly) but still fail to decode. Drop them rather than abort the build.
    """
    try:
        im = Image.open(p).convert("RGB")
    except Exception:
        return None
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2,
                  (im.width + s) // 2, (im.height + s) // 2))
    im = im.resize((SIZE, SIZE), Image.BICUBIC)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=QUALITY)
    return b.getvalue()


def pick(work, tag, count, label, source, min_dim, seed, exclude=()):
    fs = sorted(glob.glob(os.path.join(work, "img", tag, "*")))
    fs = [f for f in fs if f not in exclude]
    with ProcessPoolExecutor() as ex:
        dims = list(ex.map(maxdim, fs, chunksize=64))
    ok = [f for f, d in zip(fs, dims) if d >= min_dim]
    random.Random(seed).shuffle(ok)
    ok = ok[:count]
    print(f"  {source:16s} {len(ok):5d} of {len(fs)} (>= {min_dim}px)", flush=True)
    return [(f, label, source) for f in ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True, help="JSON list of {tag,count,label,source,min_dim}")
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    global SIZE
    SIZE = a.size

    rows = []
    for s in json.loads(a.spec):
        rows += pick(a.work, s["tag"], s["count"], s["label"], s["source"],
                     s.get("min_dim", 0), s.get("seed", 0))

    with ProcessPoolExecutor() as ex:
        blobs = list(ex.map(render, [r[0] for r in rows], chunksize=32))

    dropped = sum(1 for b in blobs if b is None)
    if dropped:
        print(f"  dropped {dropped} undecodable image(s)")

    recs = {"image": [], "label": [], "source": [], "orig_path": [], "id": []}
    for (path, label, source), blob in zip(rows, blobs):
        if blob is None:
            continue
        name = os.path.basename(path).replace("__", "/")
        recs["image"].append({"bytes": blob, "path": None})
        recs["label"].append(label)
        recs["source"].append(source)
        recs["orig_path"].append("./" + name)
        recs["id"].append(f"{source}/{os.path.basename(name)}")
    pickle.dump(recs, open(a.out, "wb"))
    print(f"built {len(recs['image'])} images at {SIZE}px -> {a.out} "
          f"({sum(len(b) for b in blobs if b)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
