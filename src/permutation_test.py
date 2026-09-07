"""
Milestone 13: Permutation / null test.

This is the single most important addition for scientific credibility.
Every evaluation so far checks: "did detected spikes land near real
events?" But there's a subtle risk in that framing -- if events happen
often enough, some detected spikes will coincidentally land near an event
by pure chance, and it would be easy to overstate how meaningful that is.

This test asks the sharper question: would a detector matching RANDOM
dates to the same spikes score about as well as matching the REAL event
dates? If yes, the detector isn't actually finding anything
event-specific -- it's just finding weeks that are somewhat spread out,
and events are common enough that some will always be nearby.

Method:
  1. Compute the observed F1 using the real, verified event dates.
  2. Generate N random "fake event" date sets (same count as real events,
     drawn uniformly across the covered time range).
  3. Compute F1 for each random set against the SAME detected spikes.
  4. Compare: how often does a random event set score as well as or
     better than the real one? That fraction is the p-value.

A low p-value (e.g. < 0.05) means the real events aligning with detected
spikes is unlikely to be chance. A high p-value means the alignment isn't
much better than randomly-placed dates would achieve -- an honest, useful
result even if it's not the one you'd hope for.

Run:
    python src/permutation_test.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR
from detect_spikes import KNOWN_EVENTS, EVENT_MATCH_WINDOW_DAYS, ROLLING_WINDOW, Z_SCORE_THRESHOLD
from evaluate_detection import evaluate, compute_rolling_zscore

N_PERMUTATIONS = 1000
RANDOM_SEED = 42


def generate_random_event_dates(date_range_start, date_range_end, n_events, rng):
    """Draw n_events random dates uniformly across the covered timeline."""
    total_days = (date_range_end - date_range_start).days
    random_offsets = rng.integers(0, total_days, size=n_events)
    return pd.to_datetime([date_range_start + pd.Timedelta(days=int(d)) for d in random_offsets])


def main():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    events_df = pd.DataFrame(KNOWN_EVENTS)
    events_df["date"] = pd.to_datetime(events_df["date"])
    n_events = len(events_df)

    z = compute_rolling_zscore(df["hype_index_equal"], ROLLING_WINDOW)
    detected_spikes = df["date"][z >= Z_SCORE_THRESHOLD]

    observed_scores = evaluate(detected_spikes, events_df, EVENT_MATCH_WINDOW_DAYS)
    observed_f1 = observed_scores["f1"]

    print(f"Observed F1 (real event dates): {observed_f1:.3f}")
    print(f"Running {N_PERMUTATIONS} permutations with random event dates...\n")

    rng = np.random.default_rng(RANDOM_SEED)
    date_start, date_end = df["date"].min(), df["date"].max()

    null_f1_scores = []
    for _ in range(N_PERMUTATIONS):
        random_dates_df = pd.DataFrame({
            "date": generate_random_event_dates(date_start, date_end, n_events, rng)
        })
        scores = evaluate(detected_spikes, random_dates_df, EVENT_MATCH_WINDOW_DAYS)
        null_f1_scores.append(scores["f1"])

    null_f1_scores = np.array(null_f1_scores)
    null_mean = null_f1_scores.mean()
    null_std = null_f1_scores.std()

    # p-value: fraction of random permutations that scored >= the real events
    p_value = (null_f1_scores >= observed_f1).mean()

    print(f"Null distribution (random event dates): mean F1 = {null_mean:.3f}, "
          f"std = {null_std:.3f}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"({int((null_f1_scores >= observed_f1).sum())} / {N_PERMUTATIONS} random "
          f"permutations scored >= the real events)\n")

    if p_value < 0.05:
        print(
            f"The real events align with detected spikes better than random "
            f"chance would produce (p={p_value:.4f} < 0.05). The detector "
            f"appears to be finding something genuinely event-related, not "
            f"just landing near events by coincidence."
        )
    elif p_value < 0.20:
        print(
            f"Borderline result (p={p_value:.4f}). The real events score "
            f"somewhat better than random, but not decisively so. Worth "
            f"reporting honestly as suggestive rather than conclusive."
        )
    else:
        print(
            f"The real events do NOT align with detected spikes meaningfully "
            f"better than random dates would (p={p_value:.4f}). This is an "
            f"important, honest result -- it suggests the detector may be "
            f"finding generically 'busy' weeks rather than weeks specifically "
            f"tied to the events in the registry. This doesn't invalidate the "
            f"project, but it's a real limitation worth stating plainly rather "
            f"than glossing over."
        )

    result_summary = pd.DataFrame([{
        "observed_f1": round(observed_f1, 4),
        "null_mean_f1": round(null_mean, 4),
        "null_std_f1": round(null_std, 4),
        "p_value": round(p_value, 4),
        "n_permutations": N_PERMUTATIONS,
        "n_events": n_events,
    }])
    out_path = PROCESSED_DATA_DIR / "permutation_test.csv"
    result_summary.to_csv(out_path, index=False)

    null_dist_path = PROCESSED_DATA_DIR / "permutation_null_distribution.csv"
    pd.DataFrame({"null_f1": null_f1_scores}).to_csv(null_dist_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {null_dist_path}")


if __name__ == "__main__":
    main()
