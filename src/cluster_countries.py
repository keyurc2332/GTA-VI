"""
Milestone 4: Country-level clustering.

Goal: discover whether countries show DIFFERENT SHAPES of interest over time
(e.g. "early spike then decline" vs "steady sustained growth"), not just
different volumes. We normalize each country's series first so raw
popularity differences don't dominate the clustering -- we want to group
by trajectory pattern, not by "which country searches GTA the most."

Run:
    python src/cluster_countries.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TRENDS_COUNTRIES

MIN_K, MAX_K = 2, 8  # candidate range; auto-selected via silhouette score
                      # rather than a fixed guess, now that we have enough
                      # countries (25) to support more than 3 groups


def force_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    return df.dropna(subset=[col])


def load_country_matrix() -> pd.DataFrame:
    """
    Build a wide matrix: rows = countries, columns = weeks, values = interest.
    Averages across the 3 keywords per country first.
    """
    df = pd.read_csv(RAW_DATA_DIR / "trends_combined.csv")
    df = force_datetime(df, "date")

    countries = [c for c in TRENDS_COUNTRIES if c != ""]  # exclude worldwide
    df = df[df["geo"].isin(countries)]

    # average across keywords, per country per week
    weekly = df.groupby(["geo", "date"])["interest"].mean().reset_index()

    # pivot to wide: countries as rows, weeks as columns
    wide = weekly.pivot(index="geo", columns="date", values="interest")
    wide = wide.sort_index(axis=1).fillna(0)
    return wide


def select_best_k(data: np.ndarray, min_k: int, max_k: int) -> tuple:
    """
    Try each candidate k and pick the one with the highest silhouette score,
    rather than assuming a number up front. Returns (best_k, scores_dict).
    """
    n_samples = data.shape[0]
    max_k = min(max_k, n_samples - 1)
    scores = {}
    for k in range(min_k, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(data)
        if len(set(labels)) < 2:
            continue
        scores[k] = silhouette_score(data, labels)
    if not scores:
        return min_k, scores
    best_k = max(scores, key=scores.get)
    return best_k, scores


def label_archetype(centroid: np.ndarray) -> str:
    """
    Classify a cluster's centroid trajectory into a human-readable archetype
    based on WHEN it peaks and how wide that peak is relative to the
    centroid's own range, rather than leaving clusters as bare numbers.
    This is a simple heuristic, not a validated typology -- stated as a
    descriptive label, not a discovery.
    """
    n = len(centroid)
    peak_idx = int(np.argmax(centroid))
    peak_position = peak_idx / n  # 0 = start, 1 = end of timeline

    # Normalize to the centroid's own min-max range, then measure what
    # fraction of the timeline sits above the halfway point of that range.
    # A narrow single spike will have a small fraction near the top; a
    # sustained plateau will have a large fraction near the top. (A raw
    # percentile threshold doesn't work here -- by definition, roughly
    # (100 - percentile)% of any distribution sits above its own
    # percentile, so that comparison is close to constant regardless of
    # actual shape.)
    c_min, c_max = centroid.min(), centroid.max()
    if c_max - c_min < 1e-9:
        return "Flat / no clear pattern"
    normalized = (centroid - c_min) / (c_max - c_min)
    sustained_frac = (normalized >= 0.5).sum() / n

    if sustained_frac > 0.45:
        return "Sustained interest"
    elif peak_position < 0.33:
        return "Early responder"
    elif peak_position > 0.67:
        return "Late riser"
    else:
        return "Mid-cycle spike"


def main():
    wide = load_country_matrix()
    print(f"Loaded interest matrix: {wide.shape[0]} countries x {wide.shape[1]} weeks")
    print(f"Countries: {list(wide.index)}\n")

    if wide.shape[0] < MIN_K + 1:
        print(f"Only {wide.shape[0]} countries available -- not enough to cluster meaningfully.")
        return

    # Normalize each country's row independently (0-1) so we cluster on
    # SHAPE of the trajectory, not absolute search volume
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(wide.T).T  # scale each row (country) separately
    normalized_df = pd.DataFrame(normalized, index=wide.index, columns=wide.columns)

    best_k, silhouette_scores = select_best_k(normalized_df.values, MIN_K, MAX_K)
    print("Silhouette scores by candidate k:")
    for k, score in silhouette_scores.items():
        marker = "  <- selected" if k == best_k else ""
        print(f"  k={k}: {score:.3f}{marker}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized_df.values)

    # Label each cluster with a descriptive archetype based on its centroid
    archetype_labels = {}
    for cluster_id in range(best_k):
        centroid = kmeans.cluster_centers_[cluster_id]
        archetype_labels[cluster_id] = label_archetype(centroid)

    result = pd.DataFrame({"country": wide.index, "cluster": labels})
    result["archetype"] = result["cluster"].map(archetype_labels)
    print(f"\nCluster assignments (k={best_k}, auto-selected via silhouette score):")
    print(result.to_string(index=False))

    # 2D projection for later visualization in the dashboard
    pca = PCA(n_components=2)
    coords = pca.fit_transform(normalized_df.values)
    result["pca_x"] = coords[:, 0]
    result["pca_y"] = coords[:, 1]

    out_path = PROCESSED_DATA_DIR / "country_clusters.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    # Also save the normalized trajectories themselves for dashboard plotting
    traj_path = PROCESSED_DATA_DIR / "country_trajectories.csv"
    normalized_df.reset_index().to_csv(traj_path, index=False)
    print(f"Saved -> {traj_path}")

    print(f"\nClustered on {wide.shape[0]} countries, auto-selected {best_k} groups "
          f"via silhouette score rather than an assumed number.")


if __name__ == "__main__":
    main()
