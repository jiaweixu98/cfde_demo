#!/usr/bin/env python3
import json
import os
from collections import defaultdict


def build_with_pandas(paths, output_path):
    import pandas as pd

    authors_path = paths["authors"]
    paper_author_path = paths["paper_author"]

    # Load author universe (restrict keys to CFDE authors)
    authors_df = pd.read_csv(
        authors_path,
        compression="infer",
        dtype={"AID": str},
        usecols=["AID"],
        keep_default_na=False,
    )
    author_ids = set(aid for aid in authors_df["AID"].astype(str))

    # Initialize adjacency mapping
    collabs: dict[str, set[str]] = {aid: set() for aid in author_ids}

    chunksize = 500_000
    carryover = None  # rows for the last PMID in previous chunk
    cols = ["PMID", "AID"]

    for chunk in pd.read_csv(
        paper_author_path,
        compression="infer",
        dtype={"PMID": str, "AID": str},
        usecols=cols,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        # Filter empty
        chunk = chunk[(chunk["PMID"] != "") & (chunk["AID"] != "")]
        # Keep only authors in our universe
        chunk = chunk[chunk["AID"].isin(author_ids)]
        if chunk.empty:
            continue

        # Sort by PMID to make contiguous groups
        chunk.sort_values("PMID", kind="mergesort", inplace=True)

        # Prepend carryover if exists
        if carryover is not None:
            chunk = pd.concat([carryover, chunk], ignore_index=True)
            carryover = None

        # Identify the last PMID in this chunk to buffer its rows for next chunk
        last_pmid = chunk.iloc[-1]["PMID"]
        mask_last = chunk["PMID"] == last_pmid
        carryover = chunk[mask_last].copy()
        work = chunk[~mask_last]

        if not work.empty:
            for pmid, group in work.groupby("PMID"):
                a_list = group["AID"].astype(str).tolist()
                uniq = list(dict.fromkeys(a_list))  # preserve order, unique
                # Update collaborations for all pairs in this paper
                for i, aid in enumerate(uniq):
                    others = uniq[:i] + uniq[i + 1 :]
                    if not others:
                        continue
                    s = collabs.get(aid)
                    if s is None:
                        # Aid not in authors list; skip
                        continue
                    s.update(others)

    # Process any remaining carryover group
    if carryover is not None and not carryover.empty:
        uniq = list(dict.fromkeys(carryover["AID"].astype(str).tolist()))
        for i, aid in enumerate(uniq):
            others = uniq[:i] + uniq[i + 1 :]
            s = collabs.get(aid)
            if s is not None:
                s.update(others)

    # Convert to required JSON structure: keys->list of strings
    out = {str(aid): sorted(list(neighs), key=str) for aid, neighs in collabs.items()}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)


def build_without_pandas(paths, output_path):
    import csv
    import gzip

    authors_path = paths["authors"]
    paper_author_path = paths["paper_author"]

    # Load author ids
    author_ids = set()
    with gzip.open(authors_path, mode="rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = str(row.get("AID", "")).strip()
            if aid:
                author_ids.add(aid)

    collabs: dict[str, set[str]] = {aid: set() for aid in author_ids}

    # Fallback approach: build PMID -> list of AIDs in-memory; may use significant memory
    pmid_to_aids: dict[str, list[str]] = defaultdict(list)
    with gzip.open(paper_author_path, mode="rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = str(row.get("PMID", "")).strip()
            aid = str(row.get("AID", "")).strip()
            if not pmid or not aid:
                continue
            if aid not in author_ids:
                continue
            pmid_to_aids[pmid].append(aid)

    for pmid, aids in pmid_to_aids.items():
        uniq = list(dict.fromkeys(aids))
        for i, aid in enumerate(uniq):
            others = uniq[:i] + uniq[i + 1 :]
            if others:
                collabs[aid].update(others)

    out = {str(aid): sorted(list(neighs), key=str) for aid, neighs in collabs.items()}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)


def main():
    base_dir = "/data/jx4237data/PKG2TKG"
    paths = {
        "authors": os.path.join(base_dir, "deliverable_data/CFDE_Authors.csv.gz"),
        "paper_author": os.path.join(base_dir, "deliverable_data/CFDE_paper_author.csv.gz"),
    }
    output_path = os.path.join(base_dir, "cfde_demo/data/author_knowledge_graph_2024.json")

    try:
        import pandas  # noqa: F401

        build_with_pandas(paths, output_path)
    except Exception:
        build_without_pandas(paths, output_path)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()


