"""Pack the staged WildFake eval subset into parquet and push it to the Hub."""
import argparse, os, pickle, sys

from datasets import Dataset, DatasetDict, Features, Image, ClassLabel, Value, Split

SOURCE_LABEL = {"coco_val2017": 0, "dalle3_advanced": 1}


def build(work):
    index = pickle.load(open(os.path.join(work, "index.pkl"), "rb"))
    recs = {"image": [], "label": [], "source": [], "orig_path": [], "id": []}
    for source, rows in sorted(index.items()):
        for orig, path, _size, label in rows:
            recs["image"].append(path)
            recs["label"].append(label)
            recs["source"].append(source)
            recs["orig_path"].append(orig)
            recs["id"].append(f"{source}/{os.path.basename(orig)}")
    feats = Features({
        "image": Image(),
        "label": ClassLabel(names=["real", "fake"]),
        "source": Value("string"),
        "orig_path": Value("string"),
        "id": Value("string"),
    })
    ds = Dataset.from_dict(recs, features=feats)
    # Deterministic interleave so a truncated read still sees both classes.
    ds = ds.shuffle(seed=0)
    return DatasetDict({"validation": ds})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dd = build(a.work)
    ds = dd["validation"]
    print(ds)
    import collections
    print("label counts:", collections.Counter(ds["label"]))
    print("source counts:", collections.Counter(ds["source"]))
    if a.dry_run:
        print("dry run: not pushing")
        return
    dd.push_to_hub(a.repo, private=a.private, max_shard_size="400MB")
    print(f"pushed https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
