"""Push a prebuilt record dict (from a .pkl) to the Hub as a named config."""
import argparse, collections, pickle
from datasets import Dataset, Features, Image, ClassLabel, Value

BASE = {
    "image": Image(),
    "label": ClassLabel(names=["real", "fake"]),
    "source": Value("string"),
    "orig_path": Value("string"),
    "id": Value("string"),
}
EXTRA = {
    "transform_chain": Value("string"),
    "primary_transform": Value("string"),
    "n_transforms": Value("int32"),
}


def features_for(recs):
    f = dict(BASE)
    for k, v in EXTRA.items():
        if k in recs:
            f[k] = v
    return Features(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    recs = pickle.load(open(a.pkl, "rb"))
    ds = Dataset.from_dict(recs, features=features_for(recs)).shuffle(seed=0)
    print(ds)
    print("labels :", dict(collections.Counter(ds["label"])))
    print("sources:", dict(collections.Counter(ds["source"])))
    if "primary_transform" in ds.column_names:
        print("n_transforms:", dict(sorted(collections.Counter(ds["n_transforms"]).items())))
        print("transforms  :", len(set(ds["primary_transform"])), "distinct settings")
    if a.dry_run:
        print("dry run"); return
    ds.push_to_hub(a.repo, config_name=a.config, split=a.split,
                   private=True, max_shard_size="400MB")
    print(f"pushed config={a.config}")


if __name__ == "__main__":
    main()
