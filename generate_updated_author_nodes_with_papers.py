#!/usr/bin/env python3
import json
import os
import sys
from collections import Counter, defaultdict


def format_h_index(value) -> str:
    if value is None:
        return ""
    try:
        # Preserve decimals if present; otherwise cast to int then to str
        f = float(value)
        if f.is_integer():
            return str(int(f)) + ".0"
        return str(f)
    except Exception:
        return str(value)


def build_with_pandas(paths, output_path):
    import pandas as pd

    authors_path = paths["authors"]
    papers_path = paths["papers"]
    paper_author_path = paths["paper_author"]
    affiliations_path = paths["affiliations"]

    authors_cols = ["AID", "FullName", "BeginYear", "h_index"]
    dtypes_authors = {"AID": str, "FullName": str}
    authors_df = pd.read_csv(
        authors_path,
        compression="infer",
        dtype=dtypes_authors,
        usecols=[c for c in authors_cols if c],
        keep_default_na=True,
    )

    # Coerce numeric columns explicitly
    if "BeginYear" in authors_df.columns:
        authors_df["BeginYear"] = pd.to_numeric(authors_df["BeginYear"], errors="coerce").astype("Int64")
    if "h_index" in authors_df.columns:
        authors_df["h_index"] = pd.to_numeric(authors_df["h_index"], errors="coerce")

    papers_cols = ["PMID", "PubYear", "Title", "citation_count", "Journal_Title"]
    dtypes_papers = {"PMID": str, "Title": str, "Journal_Title": str}
    papers_df = pd.read_csv(
        papers_path,
        compression="infer",
        dtype=dtypes_papers,
        usecols=[c for c in papers_cols if c],
        keep_default_na=True,
    )
    if "PubYear" in papers_df.columns:
        papers_df["PubYear"] = pd.to_numeric(papers_df["PubYear"], errors="coerce").astype("Int64")
    if "citation_count" in papers_df.columns:
        papers_df["citation_count"] = pd.to_numeric(papers_df["citation_count"], errors="coerce").fillna(0).astype(int)
    else:
        papers_df["citation_count"] = 0

    pa_cols = ["PMID", "AID", "PubYear"]
    dtypes_pa = {"PMID": str, "AID": str}
    paper_author_df = pd.read_csv(
        paper_author_path,
        compression="infer",
        dtype=dtypes_pa,
        usecols=[c for c in pa_cols if c],
        keep_default_na=True,
    )
    if "PubYear" in paper_author_df.columns:
        paper_author_df["PubYear"] = pd.to_numeric(paper_author_df["PubYear"], errors="coerce").astype("Int64")

    # Merge to attach paper metadata
    merged = paper_author_df.merge(
        papers_df, on="PMID", how="left", suffixes=("", "_p")
    )
    # Prefer PubYear from papers table when available
    merged["PubYear_final"] = merged["PubYear_p"].fillna(merged["PubYear"])
    merged["PubYear_final"] = pd.to_numeric(merged["PubYear_final"], errors="coerce").astype("Int64")
    merged.rename(columns={"citation_count": "CitedCount", "Journal_Title": "Venue"}, inplace=True)
    merged["CitedCount"] = pd.to_numeric(merged["CitedCount"], errors="coerce").fillna(0).astype(int)

    # Build affiliations mapping: pick most frequent affiliation string per AID
    affiliations_df = pd.read_csv(
        affiliations_path,
        compression="infer",
        dtype={"AID": str, "Affiliation": str},
        usecols=["AID", "Affiliation"],
        keep_default_na=True,
    )
    # Handle NaN affiliations by replacing with empty string before value_counts
    affiliations_df["Affiliation"] = affiliations_df["Affiliation"].fillna("")
    most_common_aff = (
        affiliations_df.groupby("AID")["Affiliation"].agg(lambda s: s.value_counts().idxmax() if len(s) else "")
    )

    # Prepare author info dict
    authors_df.set_index("AID", inplace=True)

    result = {}

    def select_papers_for_group(group_df):
        # Most cited top 3
        most_cited = (
            group_df.sort_values(["CitedCount", "PubYear_final", "PMID"], ascending=[False, False, True])
            .head(3)
        )
        # Most recent top 3
        most_recent = (
            group_df.sort_values(["PubYear_final", "CitedCount", "PMID"], ascending=[False, False, True])
            .head(3)
        )
        ordered = []
        seen = set()
        for df in (most_cited, most_recent):
            for _, row in df.iterrows():
                pmid = str(row.get("PMID", "")).strip()
                if not pmid or pmid in seen:
                    continue
                seen.add(pmid)
                ordered.append(
                    {
                        "PMID": pmid,
                        "Title": (row.get("Title") if pd.notna(row.get("Title")) else ""),
                        "PubYear": int(row["PubYear_final"]) if pd.notna(row.get("PubYear_final")) else None,
                        "CitedCount": int(row.get("CitedCount", 0) or 0),
                        "Venue": (row.get("Venue") if pd.notna(row.get("Venue")) else ""),
                    }
                )
        return ordered

    # Group by author
    for aid, group in merged.groupby("AID", sort=False):
        papers_list = select_papers_for_group(group)
        begin_year = None
        full_name = ""
        h_index_str = ""
        if aid in authors_df.index:
            row = authors_df.loc[aid]
            by = row.get("BeginYear")
            if pd.notna(by):
                begin_year = int(by)
            fn = row.get("FullName")
            if isinstance(fn, str):
                full_name = fn
            h_index_str = format_h_index(row.get("h_index"))
        aff = most_common_aff.get(aid, "")
        if pd.isna(aff):
            aff = ""

        result[str(aid)] = {
            "features": {
                "BeginYear": begin_year if begin_year is not None else None,
                "FullName": full_name,
                "H-index": h_index_str,
                "Top Cited or Most Recent Papers": papers_list,
                "Affiliation": aff,
            }
        }

    # Ensure also authors with no papers are included
    for aid, row in authors_df.iterrows():
        key = str(aid)
        if key not in result:
            by = row.get("BeginYear")
            begin_year = int(by) if pd.notna(by) else None
            result[key] = {
                "features": {
                    "BeginYear": begin_year,
                    "FullName": row.get("FullName") if isinstance(row.get("FullName"), str) else "",
                    "H-index": format_h_index(row.get("h_index")),
                    "Top Cited or Most Recent Papers": [],
                    "Affiliation": most_common_aff.get(aid, ""),
                }
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def build_without_pandas(paths, output_path):
    import csv
    import gzip

    authors_path = paths["authors"]
    papers_path = paths["papers"]
    paper_author_path = paths["paper_author"]
    affiliations_path = paths["affiliations"]

    # Load authors
    authors = {}
    with gzip.open(authors_path, mode="rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = str(row.get("AID", "")).strip()
            if not aid:
                continue
            begin_year = row.get("BeginYear")
            try:
                begin_year_val = int(begin_year) if begin_year not in (None, "", "NA", "NaN") else None
            except Exception:
                begin_year_val = None
            authors[aid] = {
                "FullName": row.get("FullName", "") or "",
                "BeginYear": begin_year_val,
                "H-index": format_h_index(row.get("h_index")),
            }

    # Load papers: PMID -> metadata
    papers = {}
    with gzip.open(papers_path, mode="rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = str(row.get("PMID", "")).strip()
            if not pmid:
                continue
            py = row.get("PubYear")
            try:
                pub_year = int(py) if py not in (None, "", "NA", "NaN") else None
            except Exception:
                pub_year = None
            try:
                cited = int(float(row.get("citation_count", 0) or 0))
            except Exception:
                cited = 0
            papers[pmid] = {
                "Title": row.get("Title", "") or "",
                "PubYear": pub_year,
                "CitedCount": cited,
                "Venue": row.get("Journal_Title", "") or "",
            }

    # Load affiliations; pick most common per AID
    aid_to_aff_counter: dict[str, Counter] = defaultdict(Counter)
    with gzip.open(affiliations_path, mode="rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = str(row.get("AID", "")).strip()
            if not aid:
                continue
            aff = (row.get("Affiliation") or "").strip()
            if aff:
                aid_to_aff_counter[aid][aff] += 1
    aid_to_aff = {}
    for aid, counter in aid_to_aff_counter.items():
        aff, _ = counter.most_common(1)[0]
        aid_to_aff[aid] = aff

    # Build mapping author -> list of PMIDs (collect from paper_author)
    aid_to_papers = defaultdict(list)
    with gzip.open(paper_author_path, mode="rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = str(row.get("AID", "")).strip()
            pmid = str(row.get("PMID", "")).strip()
            if not aid or not pmid:
                continue
            aid_to_papers[aid].append(pmid)

    # Build result
    result = {}
    for aid, info in authors.items():
        pmids = aid_to_papers.get(aid, [])
        # collect paper dicts with metadata (skip if not in papers table)
        items = []
        for pmid in pmids:
            meta = papers.get(pmid)
            if not meta:
                continue
            items.append({"PMID": pmid, **meta})

        # Compute most cited and most recent
        most_cited = sorted(
            items,
            key=lambda r: (r.get("CitedCount", 0) or 0, r.get("PubYear") or -1, r.get("PMID")),
            reverse=True,
        )[:3]
        most_recent = sorted(
            items,
            key=lambda r: (r.get("PubYear") or -1, r.get("CitedCount", 0) or 0, r.get("PMID")),
            reverse=True,
        )[:3]

        seen = set()
        ordered = []
        for lst in (most_cited, most_recent):
            for it in lst:
                pmid = it["PMID"]
                if pmid in seen:
                    continue
                seen.add(pmid)
                ordered.append(
                    {
                        "PMID": pmid,
                        "Title": it.get("Title", ""),
                        "PubYear": it.get("PubYear"),
                        "CitedCount": int(it.get("CitedCount", 0) or 0),
                        "Venue": it.get("Venue", ""),
                    }
                )

        result[str(aid)] = {
            "features": {
                "BeginYear": info.get("BeginYear"),
                "FullName": info.get("FullName", ""),
                "H-index": info.get("H-index", ""),
                "Top Cited or Most Recent Papers": ordered,
                "Affiliation": aid_to_aff.get(aid, ""),
            }
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    base_dir = "/data/jx4237data/PKG2TKG"
    paths = {
        "authors": os.path.join(base_dir, "deliverable_data/CFDE_Authors.csv.gz"),
        "paper_author": os.path.join(base_dir, "deliverable_data/CFDE_paper_author.csv.gz"),
        "affiliations": os.path.join(base_dir, "deliverable_data/CFDE_Author_Affiliations.csv.gz"),
        "papers": os.path.join(base_dir, "deliverable_data/CFDE_Papers.csv.gz"),
    }
    output_path = os.path.join(base_dir, "cfde_demo/updated_author_nodes_with_papers.json")

    # Prefer pandas for speed/memory; fallback to stdlib
    try:
        import pandas  # noqa: F401

        build_with_pandas(paths, output_path)
    except Exception:
        build_without_pandas(paths, output_path)

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()


