"""
Milestone 12: Topic clustering diagnostics.

The original HDBSCAN run found 0 clusters (100% noise) on the real 128-video
dataset. Rather than blindly tweaking min_cluster_size until something looks
nice, this tests several genuinely different configurations and reports
cluster count + noise percentage for each -- so the choice in
analyze_youtube_text.py can be justified empirically.

Tested axes:
  - Text variant: titles only vs. titles + descriptions
  - Dimensionality reduction: raw normalized embeddings vs. UMAP-reduced
  - min_cluster_size: a range of values

If NOTHING in this grid finds meaningful structure, that itself is the
honest result to report -- this dataset's titles/descriptions may simply
not have strong enough semantic separation for density-based clustering,
which is a real, defensible finding, not a failure to hide.

Run:
    python src/topic_diagnostics.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize as l2_normalize

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR
from analyze_youtube_text import clean_text, force_datetime, embed_texts

MIN_CLUSTER_SIZE_GRID = [3, 4, 5, 6, 8, 10]


def run_hdbscan(embeddings: np.ndarray, min_cluster_size: int) -> tuple:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).sum() / len(labels) * 100
    return n_clusters, noise_pct


def main():
    df = pd.read_csv(RAW_DATA_DIR / "youtube_videos.csv")
    df = force_datetime(df, "published_at")
    df["description"] = df["description"].fillna("")
    df["title"] = df["title"].fillna("")
    df["clean_description"] = df["description"].apply(clean_text)
    df["clean_title"] = df["title"].apply(clean_text)

    titles_only = df["clean_title"]
    titles_and_desc = (df["clean_title"] + ". " + df["clean_title"] + ". " + df["clean_description"]).str.strip()

    print(f"Diagnosing topic clustering on {len(df)} videos...\n")
    print("Embedding both text variants once (titles-only and titles+descriptions)...")

    embeddings_titles = l2_normalize(embed_texts(titles_only))
    embeddings_full = l2_normalize(embed_texts(titles_and_desc))

    try:
        import umap
        HAS_UMAP = True
    except Exception as e:
        # Broad except deliberate -- see analyze_youtube_text.py for why
        # ImportError alone isn't enough (TensorFlow/protobuf conflicts
        # inside umap-learn's optional parametric UMAP feature raise other
        # exception types).
        HAS_UMAP = False
        print(f"UMAP unavailable ({type(e).__name__}: {e}) -- skipping UMAP variants.\n")

    variants = {
        "titles_only, no UMAP": embeddings_titles,
        "titles+desc, no UMAP": embeddings_full,
    }
    if HAS_UMAP:
        n_comp = min(10, len(df) - 2)
        reducer_titles = umap.UMAP(n_components=n_comp, metric="cosine", random_state=42)
        reducer_full = umap.UMAP(n_components=n_comp, metric="cosine", random_state=42)
        variants["titles_only, UMAP"] = reducer_titles.fit_transform(embeddings_titles)
        variants["titles+desc, UMAP"] = reducer_full.fit_transform(embeddings_full)

    print("Running HDBSCAN across all combinations...\n")
    rows = []
    for variant_name, emb in variants.items():
        for min_cluster_size in MIN_CLUSTER_SIZE_GRID:
            n_clusters, noise_pct = run_hdbscan(emb, min_cluster_size)
            rows.append({
                "variant": variant_name,
                "min_cluster_size": min_cluster_size,
                "n_clusters": n_clusters,
                "noise_pct": round(noise_pct, 1),
            })

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False))

    # A "good" result: multiple real clusters, noise well under 50%
    viable = results_df[(results_df["n_clusters"] >= 2) & (results_df["noise_pct"] < 50)]

    if viable.empty:
        print(
            "\nNo configuration in this grid found multiple clusters with "
            "under 50% noise. That's a real result: this sample of "
            "titles/descriptions doesn't have strong enough semantic "
            "separation for density-based clustering to work well. Worth "
            "reporting honestly as 'topic discovery did not find robust "
            "clusters on this dataset' rather than forcing a result."
        )
    else:
        best = viable.sort_values("noise_pct").iloc[0]
        print(
            f"\nBest-performing configuration: {best['variant']}, "
            f"min_cluster_size={int(best['min_cluster_size'])} "
            f"-> {int(best['n_clusters'])} clusters, {best['noise_pct']}% noise."
        )

    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "topic_diagnostics.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()