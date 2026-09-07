"""
Milestone 5: NLP layer on YouTube text data.

Since Reddit access was unavailable, this substitutes YouTube titles +
descriptions as our textual signal for "what are people talking about"
and "what's the tone." Three things happen here:

  1. TOPIC DISCOVERY: sentence-transformer embeddings + HDBSCAN clustering.
     HDBSCAN (not KMeans) because we don't actually know how many distinct
     GTA VI content themes exist ahead of time -- forcing a fixed number
     of clusters would be a modeling assumption dressed up as a discovery.
     HDBSCAN finds natural density-based clusters and explicitly labels
     videos that don't fit any clear group as noise, rather than forcing
     every video into one of N buckets. TF-IDF is used ONLY afterward, to
     extract human-readable top terms per cluster for interpretation.
  2. SENTIMENT: VADER (primary) + a transformer-based social-media
     sentiment model (comparison). Rather than picking one and hoping it's
     right, both are computed and their agreement rate is reported --
     exactly what you'd want to see before trusting either one.
  3. AGREEMENT CHECK: what fraction of videos do VADER and the transformer
     model actually agree on? Low agreement would mean sentiment here is
     shakier than either model alone suggests.

First run downloads two small pretrained models from Hugging Face --
needs internet once, then both are cached locally.

Honesty note: video titles/descriptions are written by creators trying to
maximize clicks, not neutral discussion -- this measures "how creators
frame GTA VI content," which is a different (narrower) signal than genuine
audience sentiment. That distinction should be explicit in the writeup.

Run:
    python src/analyze_youtube_text.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

sys.path.append(str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

TOP_TERMS_PER_CLUSTER = 8
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, well-suited to short text
TRANSFORMER_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# min_cluster_size: smallest group HDBSCAN will call a real cluster, rather
# than folding it into noise. Set to 4 based on topic_diagnostics.py's
# empirical grid search on this dataset (~130 videos) -- re-run that
# diagnostic and reconsider this value if the YouTube dataset size changes
# substantially (e.g. after collecting with more search queries).
HDBSCAN_MIN_CLUSTER_SIZE = 4
# UMAP dimensionality to reduce to before clustering. Sentence embeddings
# are high-dimensional (384-d here), and HDBSCAN's density estimates
# degrade badly in high dimensions -- distances between points start
# looking artificially similar (the "curse of dimensionality"), so density
# based clustering often finds nothing at all, even when real structure
# exists. Reducing to a small number of UMAP dimensions first is the
# standard fix (this is what BERTopic does by default) and directly
# addresses the all-noise result seen on the raw embeddings.
UMAP_N_COMPONENTS = 10

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
# Common creator boilerplate that pollutes every description regardless of
# actual video content -- stripping this so clustering reflects the video's
# actual subject matter, not template text every channel pastes in.
BOILERPLATE_PATTERNS = [
    r"subscribe\s*(to\s*(my|our|the)?\s*channel)?",
    r"follow\s*(me|us)?\s*on\s*(instagram|twitter|facebook|tiktok|discord)?",
    r"(instagram|twitter|facebook|tiktok|discord)\s*:?.*",
    r"like\s*(and|&)?\s*subscribe",
    r"check\s*out\s*(my|our)?\s*(channel|other videos)?",
    r"business\s*(inquiries|email)\s*:?.*",
    r"#\w+",  # hashtags
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    text = URL_PATTERN.sub(" ", text)
    text = BOILERPLATE_RE.sub(" ", text)
    text = re.sub(r"@\w+", " ", text)          # @handles
    text = re.sub(r"[:;•|]+", " ", text)        # leftover punctuation from stripped boilerplate
    text = re.sub(r"\s+", " ", text).strip()
    return text


def force_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_localize(None)
    return df.dropna(subset=[col])


def embed_texts(texts: pd.Series) -> np.ndarray:
    print(f"Loading sentence embedding model ({EMBEDDING_MODEL})... "
          f"first run downloads it, may take ~30-60s.")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    return embeddings


def discover_topics(texts: pd.Series) -> tuple:
    # Step 1: embed, then L2-normalize -- sentence-transformer embeddings
    # are trained for cosine similarity, and normalizing to unit length
    # makes Euclidean distance on the normalized vectors mathematically
    # equivalent to cosine distance, which is the more appropriate metric
    # here than raw Euclidean on unnormalized vectors.
    embeddings = embed_texts(texts)
    embeddings = l2_normalize(embeddings)

    # Step 2: reduce dimensionality with UMAP before clustering. HDBSCAN's
    # density estimates are unreliable in high dimensions -- this is the
    # direct fix for an all-noise result, not just a tuning knob.
    try:
        import umap
        n_components = min(UMAP_N_COMPONENTS, len(texts) - 2, embeddings.shape[1])
        reducer = umap.UMAP(
            n_components=max(2, n_components),
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)
        print(f"Reduced embeddings from {embeddings.shape[1]}-d to {reduced.shape[1]}-d via UMAP before clustering.")
    except Exception as e:
        # Broad except is deliberate here: umap-learn's optional parametric
        # UMAP feature imports TensorFlow, which can fail with unrelated
        # environment issues (protobuf version conflicts are common) even
        # when umap-learn itself is installed fine. That's not an ImportError
        # specifically, so a narrower except would let it crash the whole
        # pipeline over a feature we don't even use. Fall back cleanly.
        print(f"UMAP unavailable ({type(e).__name__}: {e}) -- clustering on "
              f"normalized embeddings directly instead. Results may be "
              f"noisier in high dimensions, but the pipeline continues.")
        reduced = embeddings

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = n_noise / len(labels) * 100
    print(f"HDBSCAN found {n_clusters} natural clusters "
          f"({n_noise}/{len(labels)} videos = {noise_pct:.0f}% didn't fit any cluster cleanly)")

    if n_clusters == 0:
        print(
            "No natural clusters found even after UMAP + normalization. "
            "This is a real, reportable result -- it means this sample of "
            "titles/descriptions doesn't have strong enough semantic "
            "grouping for density-based clustering to separate, not a bug "
            "to keep tuning away. Run topic_diagnostics.py to see whether "
            "any parameter combination finds structure, and report "
            "honestly whichever way that comes out."
        )

    # Step 2: use TF-IDF purely to extract human-readable top terms per
    # cluster for interpretation -- this does not affect the clustering
    vectorizer = TfidfVectorizer(
        max_features=300, stop_words="english", ngram_range=(1, 2), min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    cluster_labels = {}
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        if mask.sum() == 0:
            continue
        cluster_tfidf_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).flatten()
        top_indices = cluster_tfidf_mean.argsort()[-TOP_TERMS_PER_CLUSTER:][::-1]
        top_terms = [terms[i] for i in top_indices]
        if cluster_id == -1:
            cluster_labels[cluster_id] = "UNCLUSTERED (" + ", ".join(top_terms) + ")"
        else:
            cluster_labels[cluster_id] = ", ".join(top_terms)

    return labels, cluster_labels


def score_vader_sentiment(texts: pd.Series) -> pd.Series:
    analyzer = SentimentIntensityAnalyzer()
    return texts.apply(lambda t: analyzer.polarity_scores(t)["compound"])


def score_transformer_sentiment(texts: pd.Series) -> pd.Series:
    print(f"Loading transformer sentiment model ({TRANSFORMER_SENTIMENT_MODEL})... "
          f"first run downloads it, may take ~30-60s.")
    classifier = pipeline("sentiment-analysis", model=TRANSFORMER_SENTIMENT_MODEL)
    # Truncate to the model's max length; titles/descriptions are short anyway
    results = classifier(texts.tolist(), truncation=True, max_length=128)
    # This model outputs labels like "positive"/"neutral"/"negative" directly
    return pd.Series([r["label"].lower() for r in results], index=texts.index)


def main():
    df = pd.read_csv(RAW_DATA_DIR / "youtube_videos.csv")
    df = force_datetime(df, "published_at")

    df["description"] = df["description"].fillna("")
    df["title"] = df["title"].fillna("")

    # Clean boilerplate, then weight title more heavily than description by
    # repeating it -- titles are almost always more on-topic and less
    # polluted by channel templates than descriptions are.
    df["clean_description"] = df["description"].apply(clean_text)
    df["clean_title"] = df["title"].apply(clean_text)
    df["text"] = (
        df["clean_title"] + ". " + df["clean_title"] + ". " + df["clean_description"]
    ).str.strip()
    # Drop rows where cleaning left nothing usable
    df = df[df["text"].str.len() > 3].reset_index(drop=True)

    print(f"Analyzing {len(df)} videos...")

    print("Discovering topics via sentence embeddings + HDBSCAN (TF-IDF used only for cluster labels)...")
    # Titles-only for clustering, not titles+descriptions -- topic_diagnostics.py
    # empirically found titles-only clusters noticeably better on this data
    # (titles+desc collapsed to 0 clusters at min_cluster_size >= 5, while
    # titles-only kept finding real structure). Descriptions apparently add
    # more noise (boilerplate, unrelated links/text) than signal here.
    # Sentiment still uses the fuller text below -- that's a different task
    # the diagnostic didn't test, so no evidence to change it there.
    labels, cluster_terms = discover_topics(df["clean_title"])
    df["topic_cluster"] = labels
    df["topic_terms"] = df["topic_cluster"].map(cluster_terms)

    print("\nDiscovered topic clusters:")
    for cid, terms in cluster_terms.items():
        count = (df["topic_cluster"] == cid).sum()
        print(f"  Cluster {cid} ({count} videos): {terms}")

    print("\nScoring sentiment with VADER (primary)...")
    df["vader_compound"] = score_vader_sentiment(df["text"])
    df["vader_label"] = pd.cut(
        df["vader_compound"],
        bins=[-1.01, -0.05, 0.05, 1.01],
        labels=["negative", "neutral", "positive"],
    ).astype(str)

    print("Scoring sentiment with transformer model (comparison)...")
    df["transformer_label"] = score_transformer_sentiment(df["text"])

    df["sentiment_agree"] = df["vader_label"] == df["transformer_label"]
    agreement_rate = df["sentiment_agree"].mean()

    # Keep "sentiment_label" as the primary label used elsewhere (VADER),
    # for backward compatibility with the dashboard's existing sections
    df["sentiment_label"] = df["vader_label"]
    df["sentiment_compound"] = df["vader_compound"]

    print(f"\nVADER vs transformer agreement: {agreement_rate:.1%}")
    print("\nOverall sentiment distribution (VADER):")
    print(df["sentiment_label"].value_counts().to_string())
    print("\nOverall sentiment distribution (transformer):")
    print(df["transformer_label"].value_counts().to_string())

    # Video-level output
    video_out_cols = [
        "video_id", "title", "published_at", "view_count",
        "topic_cluster", "topic_terms",
        "vader_compound", "vader_label", "transformer_label",
        "sentiment_agree", "sentiment_label", "sentiment_compound",
    ]
    video_out_path = PROCESSED_DATA_DIR / "youtube_topics_sentiment.csv"
    df[video_out_cols].to_csv(video_out_path, index=False)
    print(f"\nSaved -> {video_out_path}")

    # Sentiment agreement summary -- a single honest number for the dashboard
    agreement_summary = pd.DataFrame([{
        "agreement_rate": round(agreement_rate, 4),
        "total_videos": len(df),
        "agreeing_videos": int(df["sentiment_agree"].sum()),
    }])
    agreement_path = PROCESSED_DATA_DIR / "sentiment_agreement.csv"
    agreement_summary.to_csv(agreement_path, index=False)
    print(f"Saved -> {agreement_path}")

    # Weekly aggregation, to merge with the Hype Index timeline later
    weekly = df.set_index("published_at").resample("W").agg(
        avg_sentiment=("sentiment_compound", "mean"),
        video_count=("video_id", "count"),
    ).reset_index()
    weekly = weekly.rename(columns={"published_at": "date"})

    # Dominant topic per week (most common cluster among that week's videos)
    dominant_topic = (
        df.set_index("published_at")
        .resample("W")["topic_cluster"]
        .agg(lambda x: x.mode().iloc[0] if len(x) else np.nan)
        .reset_index()
        .rename(columns={"published_at": "date", "topic_cluster": "dominant_topic_cluster"})
    )
    weekly = weekly.merge(dominant_topic, on="date", how="left")
    weekly["dominant_topic_terms"] = weekly["dominant_topic_cluster"].map(cluster_terms)

    weekly_out_path = PROCESSED_DATA_DIR / "youtube_weekly_sentiment.csv"
    weekly.to_csv(weekly_out_path, index=False)
    print(f"Saved -> {weekly_out_path}")

    # Save the cluster label lookup separately too, for dashboard use
    cluster_lookup = pd.DataFrame(
        [{"cluster": k, "top_terms": v} for k, v in cluster_terms.items()]
    )
    cluster_lookup_path = PROCESSED_DATA_DIR / "topic_cluster_lookup.csv"
    cluster_lookup.to_csv(cluster_lookup_path, index=False)
    print(f"Saved -> {cluster_lookup_path}")


if __name__ == "__main__":
    main()