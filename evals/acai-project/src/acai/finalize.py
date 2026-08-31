"""Upload the dataset card once the data push lands, then verify the repo reads back."""
import argparse, re, sys
from huggingface_hub import HfApi, hf_hub_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--card", required=True)
    a = ap.parse_args()
    api = HfApi()

    body = open(a.card).read()
    # Keep the YAML block push_to_hub generated (it carries exact shard sizes/rows),
    # and append our prose beneath it.
    try:
        gen = open(hf_hub_download(a.repo, "README.md", repo_type="dataset")).read()
        m = re.match(r"---\n.*?\n---\n", gen, re.S)
        head = m.group(0) if m else ""
    except Exception:
        head = ""
    if head:
        our_body = re.sub(r"^---\n.*?\n---\n", "", body, count=1, flags=re.S)
        merged = head + our_body
    else:
        merged = body

    api.upload_file(path_or_fileobj=merged.encode(), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="dataset",
                    commit_message="Add dataset card")
    print("card uploaded")

    files = api.list_repo_files(a.repo, repo_type="dataset")
    print("files:", [f for f in files if f.endswith((".parquet", ".md"))][:20])

if __name__ == "__main__":
    main()
