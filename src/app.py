"""
GTA VI Hype Intelligence — the final dashboard.

Ties together every processed output from the collection and analysis
scripts into one Streamlit app.

Run:
    streamlit run src/app.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from config import PROCESSED_DATA_DIR

st.set_page_config(page_title="GTA VI Hype Intelligence", layout="wide", page_icon="🌴")

GAME_LAUNCH_DATE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Palette (used both in custom CSS and Plotly chart theming, so charts and
# chrome match exactly instead of Plotly's default clashing with the page).
# Kept to three curated accents rather than a rainbow -- one does the heavy
# lifting (pink), two support it (cyan, gold), everything else is neutral.
# ---------------------------------------------------------------------------

BG = "#08080f"
BG_DEEP = "#050508"
PANEL = "#131320"
PANEL_BORDER = "#26263c"
PINK = "#ff3d81"
CYAN = "#2de3c8"
GOLD = "#ffb84d"
TEXT = "#f2f2f5"
MUTED = "#84849c"
FAINT = "#4a4a63"

CHART_COLORWAY = [PINK, CYAN, GOLD, "#9d9db8", "#5c5c78"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="Space Grotesk, sans-serif", size=13),
    colorway=CHART_COLORWAY,
    xaxis=dict(gridcolor=PANEL_BORDER, zerolinecolor=PANEL_BORDER),
    yaxis=dict(gridcolor=PANEL_BORDER, zerolinecolor=PANEL_BORDER),
    margin=dict(t=60, l=10, r=10, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Custom CSS — condensed display font for headlines, clean sans for body,
# panel-style cards with a colored left edge and a soft tinted glow instead
# of generic rounded SaaS cards, restyled sidebar with live mission stats.
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
}}

.stApp {{
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(255,61,129,0.10), transparent),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(45,227,200,0.06), transparent),
        linear-gradient(180deg, {BG} 0%, {BG_DEEP} 100%);
}}

h1 {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.4rem;
    letter-spacing: 0.02em;
    line-height: 1;
    background: linear-gradient(90deg, {PINK}, {GOLD});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}}

h2, h3 {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: {TEXT};
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {BG_DEEP};
    border-right: 1px solid {PANEL_BORDER};
}}

[data-testid="stSidebar"] h1 {{
    font-size: 1.7rem;
    margin-bottom: 0;
}}

.stRadio > label {{
    color: {MUTED};
}}

[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 2px 0;
}}

/* Mission stats strip in the sidebar */
.mission-strip {{
    background: linear-gradient(180deg, {PANEL} 0%, {BG_DEEP} 100%);
    border: 1px solid {PANEL_BORDER};
    border-radius: 6px;
    padding: 0.9rem 1rem;
    margin: 0.8rem 0 1rem 0;
}}

.mission-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 3px 0;
    border-bottom: 1px solid {PANEL_BORDER};
}}

.mission-row:last-child {{
    border-bottom: none;
}}

.mission-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.06em;
    color: {FAINT};
    text-transform: uppercase;
}}

.mission-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: {CYAN};
    font-weight: 500;
}}

.mission-value.pink {{
    color: {PINK};
}}

/* Metrics */
[data-testid="stMetric"] {{
    background-color: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-left: 3px solid {PINK};
    border-radius: 6px;
    padding: 1rem;
    box-shadow: 0 4px 20px -8px rgba(255,61,129,0.25), 0 1px 2px rgba(0,0,0,0.4);
}}

[data-testid="stMetricLabel"] {{
    color: {MUTED};
}}

/* Panels */
.hype-panel {{
    background-color: {PANEL};
    border: 1px solid {PANEL_BORDER};
    border-left: 3px solid {CYAN};
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 24px -10px rgba(45,227,200,0.15), 0 1px 2px rgba(0,0,0,0.3);
}}

.hype-panel.warn {{
    border-left-color: {GOLD};
    box-shadow: 0 4px 24px -10px rgba(255,184,77,0.15), 0 1px 2px rgba(0,0,0,0.3);
}}

.hype-tag {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: {CYAN};
    margin-bottom: 0.6rem;
}}

.hype-tag::before {{
    content: "";
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: {CYAN};
    box-shadow: 0 0 8px {CYAN};
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {PANEL_BORDER};
    border-radius: 6px;
}}

hr {{
    border-color: {PANEL_BORDER};
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_hype_index():
    path = PROCESSED_DATA_DIR / "hype_index_with_spikes.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_spike_attribution():
    path = PROCESSED_DATA_DIR / "spike_attribution.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["spike_week", "event_date"])


@st.cache_data
def load_country_clusters():
    clusters_path = PROCESSED_DATA_DIR / "country_clusters.csv"
    traj_path = PROCESSED_DATA_DIR / "country_trajectories.csv"
    if not clusters_path.exists() or not traj_path.exists():
        return None, None
    clusters = pd.read_csv(clusters_path)
    traj = pd.read_csv(traj_path, index_col=0)
    traj.columns = pd.to_datetime(traj.columns)
    return clusters, traj


@st.cache_data
def load_youtube_topics():
    video_path = PROCESSED_DATA_DIR / "youtube_topics_sentiment.csv"
    weekly_path = PROCESSED_DATA_DIR / "youtube_weekly_sentiment.csv"
    lookup_path = PROCESSED_DATA_DIR / "topic_cluster_lookup.csv"
    if not all(p.exists() for p in [video_path, weekly_path, lookup_path]):
        return None, None, None
    videos = pd.read_csv(video_path, parse_dates=["published_at"])
    weekly = pd.read_csv(weekly_path, parse_dates=["date"])
    lookup = pd.read_csv(lookup_path)
    return videos, weekly, lookup


@st.cache_data
def load_evaluation():
    path = PROCESSED_DATA_DIR / "detection_evaluation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_event_impact():
    path = PROCESSED_DATA_DIR / "event_impact.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["event_date"])


@st.cache_data
def load_lag_analysis():
    path = PROCESSED_DATA_DIR / "lag_analysis.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_index_robustness():
    agreement_path = PROCESSED_DATA_DIR / "index_robustness_agreement.csv"
    eval_path = PROCESSED_DATA_DIR / "index_robustness_evaluation.csv"
    if not agreement_path.exists() or not eval_path.exists():
        return None, None
    return pd.read_csv(agreement_path), pd.read_csv(eval_path)


@st.cache_data
def load_spike_sensitivity():
    path = PROCESSED_DATA_DIR / "spike_sensitivity.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_topic_diagnostics():
    path = PROCESSED_DATA_DIR / "topic_diagnostics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_permutation_test():
    path = PROCESSED_DATA_DIR / "permutation_test.csv"
    null_path = PROCESSED_DATA_DIR / "permutation_null_distribution.csv"
    if not path.exists() or not null_path.exists():
        return None, None
    return pd.read_csv(path), pd.read_csv(null_path)


@st.cache_data
def load_sentiment_disagreement():
    examples_path = PROCESSED_DATA_DIR / "sentiment_disagreement_examples.csv"
    summary_path = PROCESSED_DATA_DIR / "sentiment_disagreement_summary.csv"
    if not examples_path.exists() or not summary_path.exists():
        return None, None
    return pd.read_csv(examples_path), pd.read_csv(summary_path)


@st.cache_data
def load_bootstrap_ci():
    path = PROCESSED_DATA_DIR / "bootstrap_ci.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_event_holdout():
    summary_path = PROCESSED_DATA_DIR / "event_holdout.csv"
    grid_path = PROCESSED_DATA_DIR / "event_holdout_dev_grid.csv"
    if not summary_path.exists() or not grid_path.exists():
        return None, None
    return pd.read_csv(summary_path), pd.read_csv(grid_path)


@st.cache_data
def load_loo():
    path = PROCESSED_DATA_DIR / "leave_one_event_out.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_holdout_by_type():
    path = PROCESSED_DATA_DIR / "holdout_by_type.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_walk_forward():
    path = PROCESSED_DATA_DIR / "walk_forward_evaluation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["test_event_date"])


@st.cache_data
def load_confusion_breakdown():
    hits_path = PROCESSED_DATA_DIR / "confusion_hits.csv"
    misses_path = PROCESSED_DATA_DIR / "confusion_misses.csv"
    fp_path = PROCESSED_DATA_DIR / "confusion_false_positives.csv"
    if not all(p.exists() for p in [hits_path, misses_path, fp_path]):
        return None, None, None
    return pd.read_csv(hits_path), pd.read_csv(misses_path), pd.read_csv(fp_path)


@st.cache_data
def load_signal_ablation():
    path = PROCESSED_DATA_DIR / "signal_ablation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_youtube_signal_ablation():
    path = PROCESSED_DATA_DIR / "youtube_signal_ablation.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_event_response_curves():
    by_type_path = PROCESSED_DATA_DIR / "event_response_by_type.csv"
    if not by_type_path.exists():
        return None
    return pd.read_csv(by_type_path)


def missing_data_notice(name: str):
    st.markdown(
        f'<div class="hype-panel warn">Run the script that builds '
        f'<b>{name}</b> first, then refresh this page.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown("# GTA VI\n## HYPE INTEL")

# --- Mission stats: real numbers computed from the data, plus a live
# countdown to the actual Nov 19, 2026 release date, since that's a real
# fixed point this whole project is oriented around.
_hype_df_for_stats = load_hype_index()
days_to_launch = (GAME_LAUNCH_DATE - datetime.now(timezone.utc)).days

if _hype_df_for_stats is not None:
    _weeks_tracked = len(_hype_df_for_stats)
    _spikes_found = int(_hype_df_for_stats["is_spike"].sum()) if "is_spike" in _hype_df_for_stats.columns else None
else:
    _weeks_tracked = None
    _spikes_found = None

stats_rows = f"""
  <div class="mission-row">
    <span class="mission-label">Launch T-minus</span>
    <span class="mission-value pink">{days_to_launch}d</span>
  </div>
"""
if _weeks_tracked is not None:
    stats_rows += f"""
  <div class="mission-row">
    <span class="mission-label">Weeks tracked</span>
    <span class="mission-value">{_weeks_tracked}</span>
  </div>
"""
if _spikes_found is not None:
    stats_rows += f"""
  <div class="mission-row">
    <span class="mission-label">Spikes detected</span>
    <span class="mission-value">{_spikes_found}</span>
  </div>
"""

st.sidebar.markdown(f'<div class="mission-strip">{stats_rows}</div>', unsafe_allow_html=True)

section = st.sidebar.radio(
    "Section",
    [
        "Global Hype",
        "Spike Attribution",
        "Baseline vs Combined",
        "Holdout Validation",
        "Signal Ablation",
        "Index Robustness",
        "Detector Sensitivity",
        "Event Impact",
        "Event Response Shapes",
        "Signal Lag",
        "The Conversation",
        "Sentiment",
        "The World",
        "Findings",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<span style="color:{MUTED}; font-size:0.85rem;">'
    f'Measuring how a video game turns into a global cultural moment, '
    f'using public search, viewing, and discussion signals.</span>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Section — Global Hype
# ---------------------------------------------------------------------------

if section == "Global Hype":
    st.markdown('<span class="hype-tag">SIGNAL OVERVIEW</span>', unsafe_allow_html=True)
    st.title("Global Hype")
    st.markdown(
        "I built a **Hype Index** by combining Google Trends search interest, "
        "Wikipedia pageviews, and YouTube upload activity into a single "
        "weekly score. Nothing here is one platform's opinion — it's three "
        "independent signals agreeing (or not)."
    )

    df = load_hype_index()
    if df is None:
        missing_data_notice("hype_index_with_spikes.csv")
    else:
        weight_mode = st.radio(
            "Index weighting", ["Equal-weight", "PCA-weighted"], horizontal=True
        )
        col = "hype_index_equal" if weight_mode == "Equal-weight" else "hype_index_pca"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], mode="lines", name="Hype Index",
            line=dict(color=PINK, width=2.5),
            fill="tozeroy", fillcolor="rgba(255,61,129,0.08)",
        ))
        fig.update_layout(title=f"Hype Index over time — {weight_mode}", height=440)
        st.plotly_chart(style_fig(fig), width="stretch")

        st.markdown("#### The three signals on their own")
        components_fig = px.line(
            df, x="date", y=["trends_signal", "wiki_signal", "youtube_signal"],
            labels={"value": "Normalized signal (0–1)", "date": "", "variable": "Signal"},
        )
        st.plotly_chart(style_fig(components_fig), width="stretch")

        st.markdown(
            f'<div class="hype-panel">Equal-weight treats all three signals '
            f'as equally important. PCA-weighted instead derives weights from '
            f'whatever variation the three signals actually share in common. '
            f'I compare both rather than trusting either one blindly — and '
            f'they don\'t fully agree: correlated, but their highest-attention '
            f'weeks only overlap about half the time. That disagreement is a '
            f'real limit of the index, not just a footnote.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section — Spike Attribution
# ---------------------------------------------------------------------------

elif section == "Spike Attribution":
    st.markdown('<span class="hype-tag">ANOMALY DETECTION</span>', unsafe_allow_html=True)
    st.title("Spike Attribution")
    st.markdown(
        "I let the data flag unusual weeks first, using a rolling z-score "
        "on the Hype Index — then checked which of those weeks lined up "
        "with a real GTA VI event. That order matters: it keeps the claim "
        "honest as *'this spike coincided with X'*, not *'X caused this.'*"
    )

    hype_df = load_hype_index()
    attribution_df = load_spike_attribution()
    hits_df, misses_df, fp_df = load_confusion_breakdown()

    if hype_df is None or attribution_df is None:
        missing_data_notice("spike detection output")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hype_df["date"], y=hype_df["hype_index_equal"],
            mode="lines", name="Hype Index", line=dict(color=CYAN, width=2),
        ))

        if hits_df is not None and not hits_df.empty:
            hit_dates = pd.to_datetime(hits_df["matched_spike_week"])
            hit_values = hype_df.set_index("date").reindex(hit_dates)["hype_index_equal"]
            fig.add_trace(go.Scatter(
                x=hit_dates, y=hit_values, mode="markers", name="Hit (matched a known event)",
                marker=dict(color="#2de3c8", size=13, symbol="star", line=dict(color=TEXT, width=1)),
            ))
        if misses_df is not None and not misses_df.empty:
            miss_dates = pd.to_datetime(misses_df["event_date"])
            miss_values = hype_df.set_index("date").reindex(miss_dates, method="nearest")["hype_index_equal"]
            fig.add_trace(go.Scatter(
                x=miss_dates, y=miss_values, mode="markers", name="Miss (event, no spike caught it)",
                marker=dict(color="#ff3d81", size=11, symbol="x", line=dict(color=TEXT, width=1)),
            ))
        if fp_df is not None and not fp_df.empty:
            fp_dates = pd.to_datetime(fp_df["spike_week"])
            fp_values = hype_df.set_index("date").reindex(fp_dates)["hype_index_equal"]
            fig.add_trace(go.Scatter(
                x=fp_dates, y=fp_values, mode="markers",
                name="False positive (spike, no known event)",
                marker=dict(color=GOLD, size=9, symbol="circle-open", line=dict(width=2)),
            ))

        fig.update_layout(title="Hits, misses, and false positives on the Hype Index", height=460)
        st.plotly_chart(style_fig(fig), width="stretch")
        st.markdown(
            f'<div class="hype-panel">Reading this chart: cyan stars are '
            f'known events the detector caught, pink X marks are known '
            f'events it missed entirely, and gold circles are detected '
            f'spikes that match no known event — some of those may be real '
            f'GTA VI moments not yet in the 21-event registry, not '
            f'necessarily noise.</div>',
            unsafe_allow_html=True,
        )

        matched = attribution_df[attribution_df["matched_event"] != "UNEXPLAINED"]
        unexplained = attribution_df[attribution_df["matched_event"] == "UNEXPLAINED"]

        col1, col2 = st.columns(2)
        col1.metric("Matched to a known event", len(matched))
        col2.metric("Unexplained", len(unexplained))

        st.markdown("#### Matched to a known event")
        st.dataframe(matched, width="stretch")

        st.markdown("#### Unexplained — worth a manual look")
        st.dataframe(unexplained, width="stretch")
        st.markdown(
            f'<div class="hype-panel warn">Not every unexplained spike is a '
            f'real discovery. Some are just noise from the rolling-window '
            f'method, especially early in the series when there isn\'t much '
            f'data yet to build a stable baseline.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### What got missed entirely")
    st.markdown(
        "The tables above show detected spikes and where they landed. This "
        "shows the other side: known events with NO detected spike nearby "
        "at all — a genuine miss, not just an unexplained extra spike."
    )
    hits_df, misses_df, fp_df = load_confusion_breakdown()
    if misses_df is None:
        missing_data_notice("confusion_breakdown.py output")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Events caught", len(hits_df))
        col2.metric("Events missed", len(misses_df))
        col3.metric("False-positive spikes", len(fp_df))

        st.markdown("Missed events by type:")
        st.dataframe(misses_df["event_type"].value_counts().rename("count"), width="stretch")

        st.markdown("Full miss list:")
        st.dataframe(misses_df, width="stretch")

        st.markdown(
            f'<div class="hype-panel warn">A "false positive" here doesn\'t '
            f'necessarily mean noise — it could be a real GTA VI event that '
            f'simply isn\'t in the 21-event registry yet. Worth treating '
            f'these as investigation leads, the same way the unexplained '
            f'spikes above are.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section — Baseline vs Combined
# ---------------------------------------------------------------------------

elif section == "Baseline vs Combined":
    st.markdown('<span class="hype-tag">EVALUATION</span>', unsafe_allow_html=True)
    st.title("Baseline vs Combined")
    st.markdown(
        "Before trusting the Hype Index, I checked whether combining three "
        "signals actually detects real events better than Google Trends "
        "alone — or whether the extra complexity bought nothing."
    )

    eval_df = load_evaluation()
    if eval_df is None:
        missing_data_notice("evaluate_detection.py output")
    else:
        st.dataframe(eval_df, width="stretch")

        fig = px.bar(
            eval_df, x="method", y=["precision", "recall", "f1"],
            barmode="group",
            labels={"value": "Score", "method": "", "variable": "Metric"},
            title="Detection quality: baseline vs combined signal",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style_fig(fig), width="stretch")

        baseline_f1 = eval_df.loc[eval_df["method"].str.contains("Baseline"), "f1"].values[0]
        combined_f1 = eval_df.loc[eval_df["method"].str.contains("Combined"), "f1"].values[0]

        if combined_f1 > baseline_f1:
            verdict = ("Combining three signals into the Hype Index caught real "
                       "events more reliably than Trends alone on this data.")
        elif combined_f1 < baseline_f1:
            verdict = ("The single-signal baseline actually matched or beat the "
                       "combined index here — worth being honest about rather "
                       "than assuming more signals is automatically better.")
        else:
            verdict = "Both methods performed identically on this event set."

        st.markdown(f'<div class="hype-panel">{verdict}</div>', unsafe_allow_html=True)
        events_total = int(eval_df["events_total"].iloc[0]) if "events_total" in eval_df.columns else None
        events_note = f"With {events_total} verified events" if events_total else "With a modest verified-event list"
        st.markdown(
            f'<div class="hype-panel warn">{events_note}, these '
            f'numbers are directional, not statistically powerful. A larger '
            f'verified-event list would sharpen this further.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### How much does that F1 actually wobble?")
    st.markdown(
        "A point-estimate F1 hides how much it would change with a "
        "slightly different sample of events. This resamples the event "
        "list itself 2,000 times to build a 95% confidence interval "
        "around each method's score."
    )
    ci_df = load_bootstrap_ci()
    if ci_df is None:
        missing_data_notice("bootstrap_ci.py output")
    else:
        fig = go.Figure()
        for _, row in ci_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["method"]], y=[row["f1_mean"]],
                error_y=dict(
                    type="data",
                    array=[row["f1_ci_upper"] - row["f1_mean"]],
                    arrayminus=[row["f1_mean"] - row["f1_ci_lower"]],
                ),
                marker_color=PINK if "Combined" in row["method"] else CYAN,
                showlegend=False,
            ))
        fig.update_layout(title="F1 with 95% bootstrap confidence intervals", height=380)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style_fig(fig), width="stretch")

        overlap = not (
            ci_df.loc[ci_df["method"].str.contains("Combined"), "f1_ci_lower"].values[0] >
            ci_df.loc[ci_df["method"].str.contains("Baseline"), "f1_ci_upper"].values[0]
            or
            ci_df.loc[ci_df["method"].str.contains("Baseline"), "f1_ci_lower"].values[0] >
            ci_df.loc[ci_df["method"].str.contains("Combined"), "f1_ci_upper"].values[0]
        )
        if overlap:
            ci_verdict = (
                "The confidence intervals overlap. A higher point-estimate F1 "
                "for the combined index doesn't mean it's decisively better — "
                "with this event sample size, the gap could plausibly be noise."
            )
        else:
            ci_verdict = (
                "The confidence intervals don't overlap — the combined "
                "method's advantage looks like a real difference, not just "
                "small-sample noise."
            )
        st.markdown(f'<div class="hype-panel warn">{ci_verdict}</div>', unsafe_allow_html=True)

    st.markdown("#### Is this better than chance?")
    st.markdown(
        "A subtler risk than sample size: with enough spikes and events on "
        "the calendar, some will land near each other by pure coincidence. "
        "This permutation test checks whether the real event dates align "
        "with detected spikes meaningfully better than **1,000 sets of "
        "randomly placed dates** would."
    )

    perm_df, null_dist_df = load_permutation_test()
    if perm_df is None:
        missing_data_notice("permutation_test.py output")
    else:
        row = perm_df.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Observed F1", f"{row['observed_f1']:.3f}")
        col2.metric("Random-chance F1", f"{row['null_mean_f1']:.3f}")
        col3.metric("p-value", f"{row['p_value']:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=null_dist_df["null_f1"], nbinsx=30, marker_color=CYAN,
            name="Random-date F1 scores",
        ))
        fig.add_vline(
            x=row["observed_f1"], line_color=PINK, line_width=3,
            annotation_text="Real events", annotation_position="top",
        )
        fig.update_layout(
            title="Real events vs. 1,000 random-date permutations",
            xaxis_title="F1 score", yaxis_title="Count", height=400,
        )
        st.plotly_chart(style_fig(fig), width="stretch")

        p_value = row["p_value"]
        if p_value < 0.05:
            verdict = (f"p={p_value:.4f} — the real events align with detected "
                      f"spikes better than random chance would produce. The "
                      f"detector appears to be finding something genuinely "
                      f"event-related.")
        elif p_value < 0.20:
            verdict = (f"p={p_value:.4f} — a borderline result. The real events "
                      f"score somewhat better than random, but not decisively.")
        else:
            verdict = (f"p={p_value:.4f} — the real events do NOT align "
                      f"meaningfully better than random dates would. This is an "
                      f"important limitation: the detector may be finding "
                      f"generically busy weeks rather than weeks specifically "
                      f"tied to the events in the registry.")
        st.markdown(f'<div class="hype-panel warn">{verdict}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section — Holdout Validation
# ---------------------------------------------------------------------------

elif section == "Holdout Validation":
    st.markdown('<span class="hype-tag">NO PEEKING</span>', unsafe_allow_html=True)
    st.title("Holdout Validation")
    st.markdown(
        "Every parameter choice so far was picked by checking which one "
        "scored best against the full event list — which is tuning on the "
        "same events used to evaluate the result. This splits the events "
        "chronologically: earlier events for tuning, the most recent ones "
        "held back and never touched until the very last step."
    )

    holdout_summary, dev_grid = load_event_holdout()
    if holdout_summary is None:
        missing_data_notice("event_holdout.py output")
    else:
        row = holdout_summary.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Dev events", int(row["n_dev_events"]))
        col2.metric("Holdout events", int(row["n_holdout_events"]))
        col3.metric("Chosen params", f"win={int(row['best_window'])}, z={row['best_threshold']}")

        st.markdown("#### Parameters tuned on dev events only")
        st.dataframe(dev_grid, width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("Dev F1 (tuning set)", f"{row['dev_f1']:.3f}")
        col2.metric("Holdout F1 (honest)", f"{row['holdout_f1']:.3f}")
        col3.metric("If tuned on everything", f"{row['full_set_tuned_f1']:.3f}")

        gap = row["tuning_gap"]
        if gap > 0.05:
            verdict = (
                f"There's a meaningful gap ({gap:.3f}) between the honest "
                f"holdout result and what tuning on the full event set would "
                f"have shown. That gap is the optimistic bias every earlier "
                f"F1 number in this project carries — the holdout figure "
                f"above is the more trustworthy one."
            )
        else:
            verdict = (
                f"The gap between the honest holdout result and full-set "
                f"tuning is small ({gap:.3f}) — the detector's performance "
                f"doesn't look like it was just an artifact of tuning on "
                f"the same events used to evaluate it."
            )
        st.markdown(f'<div class="hype-panel warn">{verdict}</div>', unsafe_allow_html=True)

    st.markdown("#### Was that just one unlucky split?")
    st.markdown(
        "A single train/test division is itself a small sample. Leave-one-event-out "
        "cross-validation uses the data more efficiently: for each of the 21 "
        "events, tune on the other 20, then check whether that one held-out "
        "event gets caught. Repeated 21 times."
    )
    loo_df = load_loo()
    if loo_df is None:
        missing_data_notice("leave_one_event_out.py output")
    else:
        catch_rate = loo_df["held_out_event_caught"].mean()
        col1, col2 = st.columns(2)
        col1.metric("LOO catch rate", f"{catch_rate:.1%}")
        col2.metric("Events caught", f"{int(loo_df['held_out_event_caught'].sum())}/{len(loo_df)}")

        by_type = loo_df.groupby("event_type")["held_out_event_caught"].agg(["sum", "count"])
        by_type["rate"] = (by_type["sum"] / by_type["count"]).round(3)
        st.markdown("Catch rate by event type:")
        st.dataframe(by_type, width="stretch")

        if catch_rate < 0.4:
            loo_verdict = (
                f"A {catch_rate:.1%} catch rate across {len(loo_df)} independent "
                f"trials confirms this isn't an artifact of one unlucky split — "
                f"the generalization gap is real and consistent."
            )
        else:
            loo_verdict = (
                f"A {catch_rate:.1%} catch rate is better than the single "
                f"split suggested — that specific division may have been "
                f"unusually difficult rather than fully representative."
            )
        st.markdown(f'<div class="hype-panel warn">{loo_verdict}</div>', unsafe_allow_html=True)

    st.markdown("#### Does it generalize across event type, not just across time?")
    st.markdown(
        "A different question: tune only on predictable, scripted event "
        "types (official announcements, marketing, controversy), then test "
        "on messier types the detector never saw during tuning (leaks, "
        "rumors)."
    )
    type_holdout_df = load_holdout_by_type()
    if type_holdout_df is None:
        missing_data_notice("holdout_by_type.py output")
    else:
        row = type_holdout_df.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Dev types F1", f"{row['dev_f1']:.3f}")
        col2.metric("Holdout types F1", f"{row['holdout_f1']:.3f}")
        col3.metric("Gap", f"{row['gap']:.3f}")

        gap = row["gap"]
        if gap > 0.15:
            type_verdict = (
                "Performance drops moving from scripted announcement types "
                "to messier leak/rumor types — the detector may be partly "
                "keyed to the specific rhythm of official Rockstar PR beats "
                "rather than a fully general attention shock."
            )
        else:
            type_verdict = (
                "Performance holds up moving from official/marketing/"
                "controversy events to leaks/rumors — interesting contrast "
                "to the time-based holdout above, since it suggests the "
                "earlier collapse was about something specific to that "
                "later time period, not about event category at all."
            )
        st.markdown(f'<div class="hype-panel">{type_verdict}</div>', unsafe_allow_html=True)

    st.markdown("#### The strictest test: walk-forward evaluation")
    st.markdown(
        "Leave-one-out has a subtle unrealism: when testing on an early "
        "event, it still tunes using LATER events — future information "
        "no real system would have. Walk-forward fixes this: for each "
        "event in chronological order, tune only on events that happened "
        "before it, then check if that next event gets caught. This is "
        "the closest thing in the project to 'would this have actually "
        "worked in production.'"
    )
    wf_df = load_walk_forward()
    if wf_df is None:
        missing_data_notice("walk_forward_evaluation.py output")
    else:
        catch_rate = wf_df["caught"].mean()
        col1, col2 = st.columns(2)
        col1.metric("Walk-forward catch rate", f"{catch_rate:.1%}")
        col2.metric("Events tested", f"{int(wf_df['caught'].sum())}/{len(wf_df)}")

        st.dataframe(
            wf_df[["test_event", "test_event_date", "event_type",
                   "n_past_events_available", "caught"]],
            width="stretch",
        )

        st.markdown(
            f'<div class="hype-panel warn">This is the strictest '
            f'generalization test in the project, and the lowest score — '
            f'meaningfully below both the chronological holdout and '
            f'leave-one-out. That gap between tests is itself informative: '
            f'leave-one-out\'s more optimistic number came partly from '
            f'letting future events inform tuning decisions about the '
            f'past, which no real deployed system could ever do. The '
            f'walk-forward number is the one I\'d trust most if asked '
            f'"would this actually work going forward."</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section — Signal Ablation
# ---------------------------------------------------------------------------

elif section == "Signal Ablation":
    st.markdown('<span class="hype-tag">WHICH SIGNAL MATTERS</span>', unsafe_allow_html=True)
    st.title("Signal Ablation")
    st.markdown(
        "Every prior comparison was 'Trends alone vs. all three combined' — "
        "which skips a real question: does every signal actually help, or "
        "is one of them dead weight? This tests all 7 possible "
        "combinations of Trends, Wikipedia, and YouTube."
    )

    ablation_df = load_signal_ablation()
    if ablation_df is None:
        missing_data_notice("signal_ablation.py output")
    else:
        fig = px.bar(
            ablation_df.sort_values("f1"), x="f1", y="signals", orientation="h",
            labels={"f1": "F1 score", "signals": ""},
            title="Detection F1 by signal combination",
        )
        st.plotly_chart(style_fig(fig), width="stretch")
        st.dataframe(ablation_df, width="stretch")

        best = ablation_df.iloc[0]
        all_three = ablation_df[ablation_df["n_signals"] == 3].iloc[0]

        if best["n_signals"] < 3:
            verdict = (
                f'The best-performing combination ("{best["signals"]}", '
                f'F1={best["f1"]:.3f}) doesn\'t use all three signals. Using '
                f'every available signal isn\'t automatically better than a '
                f'more selective combination on this event set.'
            )
        else:
            verdict = (
                f'Using all three signals together performs best '
                f'(F1={all_three["f1"]:.3f}) — combining signals earns its '
                f'complexity here rather than just adding it for its own sake.'
            )
        st.markdown(f'<div class="hype-panel">{verdict}</div>', unsafe_allow_html=True)

    st.markdown("#### Does splitting YouTube into two signals help?")
    st.markdown(
        "The blended YouTube signal above mixes upload activity with "
        "lifetime view counts — two conceptually different things averaged "
        "into one number. This tests them separately: creator activity "
        "(uploads/week, unique channels) vs audience engagement (views, "
        "likes, log-scaled so one viral video can't dominate)."
    )
    yt_ablation_df = load_youtube_signal_ablation()
    if yt_ablation_df is None:
        missing_data_notice("youtube_signal_ablation.py output")
    else:
        st.dataframe(yt_ablation_df, width="stretch")

        creator_row = yt_ablation_df[yt_ablation_df["signals"] == "YouTube (creator activity)"]
        audience_row = yt_ablation_df[yt_ablation_df["signals"] == "YouTube (audience engagement)"]
        if not creator_row.empty and not audience_row.empty:
            c_f1 = creator_row["f1"].iloc[0]
            a_f1 = audience_row["f1"].iloc[0]
            if abs(c_f1 - a_f1) > 0.05:
                better = "creator activity" if c_f1 > a_f1 else "audience engagement"
                yt_verdict = (
                    f"Splitting reveals a real difference: {better} alone "
                    f"(F1={max(c_f1, a_f1):.3f}) is more informative than the "
                    f"other (F1={min(c_f1, a_f1):.3f}). The original blended "
                    f"YouTube signal was likely diluting a more useful "
                    f"sub-signal by averaging it with a weaker one."
                )
            else:
                yt_verdict = (
                    f"Creator activity (F1={c_f1:.3f}) and audience "
                    f"engagement (F1={a_f1:.3f}) perform similarly once "
                    f"separated — the blended signal wasn't hiding a "
                    f"meaningfully stronger sub-signal."
                )
            st.markdown(f'<div class="hype-panel warn">{yt_verdict}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section — Index Robustness
# ---------------------------------------------------------------------------

elif section == "Index Robustness":
    st.markdown('<span class="hype-tag">SENSITIVITY</span>', unsafe_allow_html=True)
    st.title("Index Robustness")
    st.markdown(
        "The Hype Index combines three signals, but nothing says they "
        "should be weighted equally. Rather than picking one weighting and "
        "hoping it's right, I compared three methods: equal-weight, "
        "PCA-weighted, and a rank-based average that's robust to any one "
        "signal having a wider numeric range than the others."
    )

    agreement_df, eval_df = load_index_robustness()
    if agreement_df is None:
        missing_data_notice("index_robustness.py output")
    else:
        st.markdown("#### How much do the three methods agree?")
        st.dataframe(agreement_df, width="stretch")

        avg_overlap = agreement_df["top10_overlap"].apply(
            lambda x: int(x.split("/")[0])
        ).mean()

        if avg_overlap >= 7:
            verdict = (f"Average top-10 overlap: {avg_overlap:.1f}/10 — the "
                      f"methods broadly agree on which weeks mattered most.")
        else:
            verdict = (f"Average top-10 overlap: {avg_overlap:.1f}/10 — "
                      f"meaningful disagreement. Which weeks look like the "
                      f"biggest hype moments genuinely depends on the "
                      f"aggregation method, which is a real limitation of "
                      f"the index, not a footnote.")
        st.markdown(f'<div class="hype-panel warn">{verdict}</div>', unsafe_allow_html=True)

        st.markdown("#### Which method actually detects real events best?")
        st.dataframe(eval_df, width="stretch")

        fig = px.bar(
            eval_df, x="method", y="f1",
            labels={"f1": "F1 score", "method": ""},
            title="Event-detection F1 by aggregation method",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(style_fig(fig), width="stretch")

        best = eval_df.loc[eval_df["f1"].idxmax(), "method"]
        st.markdown(
            f'<div class="hype-panel">{best} performed best on this event '
            f'set. That doesn\'t make it objectively correct — with only a '
            f'modest number of verified events, this is directional '
            f'evidence, not proof.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section — Detector Sensitivity
# ---------------------------------------------------------------------------

elif section == "Detector Sensitivity":
    st.markdown('<span class="hype-tag">PARAMETER SWEEP</span>', unsafe_allow_html=True)
    st.title("Detector Sensitivity")
    st.markdown(
        "The spike detector uses a rolling window and a z-score threshold — "
        "two numbers I originally picked because they looked reasonable, "
        "not because they were derived from anything. This grid-searches "
        "both and checks event-detection quality at every combination."
    )

    sens_df = load_spike_sensitivity()
    if sens_df is None:
        missing_data_notice("spike_sensitivity.py output")
    else:
        heatmap_data = sens_df.pivot(index="window", columns="threshold", values="f1")

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"{c}" for c in heatmap_data.columns],
            y=[f"{r}" for r in heatmap_data.index],
            colorscale=[[0, PANEL], [0.5, CYAN], [1, PINK]],
            text=heatmap_data.round(3).values,
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="F1 score by window size and z-threshold",
            xaxis_title="Z-score threshold", yaxis_title="Rolling window (weeks)",
            height=400,
        )
        st.plotly_chart(style_fig(fig), width="stretch")

        f1_range = sens_df["f1"].max() - sens_df["f1"].min()
        best_row = sens_df.loc[sens_df["f1"].idxmax()]

        if f1_range < 0.15:
            verdict = (f"F1 stays fairly stable across the whole grid (range: "
                      f"{f1_range:.3f}) — the originally chosen parameters "
                      f"weren't a lucky pick that only looks good at that "
                      f"exact setting.")
        else:
            verdict = (f"F1 varies meaningfully across the grid (range: "
                      f"{f1_range:.3f}). Best found: window="
                      f"{int(best_row['window'])}, threshold="
                      f"{best_row['threshold']}. The chosen defaults should "
                      f"be reported as one reasonable choice among several, "
                      f"not the objectively correct one.")

        st.markdown(f'<div class="hype-panel warn">{verdict}</div>', unsafe_allow_html=True)
        st.dataframe(sens_df, width="stretch")


# ---------------------------------------------------------------------------
# Section — Event Impact
# ---------------------------------------------------------------------------

elif section == "Event Impact":
    st.markdown('<span class="hype-tag">CAUSAL CAUTION</span>', unsafe_allow_html=True)
    st.title("Event Impact")
    st.markdown(
        "For each event, I compared the 3 weeks before to the 3 weeks after "
        "and bootstrapped the difference 2,000 times to get a plausible "
        "range instead of one precise-sounding number. This is an "
        "**observed change coinciding with the event**, not a proven "
        "causal effect of it — other things happen in the same weeks too."
    )

    impact_df = load_event_impact()
    if impact_df is None:
        missing_data_notice("event_impact.py output")
    else:
        event_options = impact_df["event"].unique().tolist()
        selected_event = st.selectbox("Event", event_options)

        subset = impact_df[impact_df["event"] == selected_event].dropna(subset=["observed_change"])

        if subset.empty:
            st.info("Not enough data around this event to estimate a change.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=subset["signal"], y=subset["observed_change"],
                error_y=dict(
                    type="data",
                    array=subset["ci_95_upper"] - subset["observed_change"],
                    arrayminus=subset["observed_change"] - subset["ci_95_lower"],
                ),
                marker_color=PINK,
            ))
            fig.update_layout(
                title=f"Observed change, 3 weeks before -> after: {selected_event}",
                yaxis_title="Change (normalized signal)", height=420,
            )
            fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
            st.plotly_chart(style_fig(fig), width="stretch")

            st.dataframe(
                subset[["signal", "before_mean", "after_mean", "observed_change",
                        "ci_95_lower", "ci_95_upper"]],
                width="stretch",
            )

        st.markdown(
            f'<div class="hype-panel warn">A wide confidence interval means '
            f'the estimate isn\'t stable — usually because that window only '
            f'has a few weeks of data. A tight interval that doesn\'t cross '
            f'zero is the stronger signal.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section — Signal Lag
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Section — Event Response Shapes
# ---------------------------------------------------------------------------

elif section == "Event Response Shapes":
    st.markdown('<span class="hype-tag">SHAPE, NOT JUST SIZE</span>', unsafe_allow_html=True)
    st.title("Event Response Shapes")
    st.markdown(
        "Event Impact asks how much attention changed. This asks something "
        "different: does the SHAPE of that response differ by event type — "
        "does a trailer create a sharp, short-lived spike while a delay "
        "creates a slower, more sustained response? Each event's trajectory "
        "is normalized to its own pre-event baseline (1.0 = no change) so "
        "events of different sizes can be compared on shape alone."
    )

    response_df = load_event_response_curves()
    if response_df is None:
        missing_data_notice("event_response_curves.py output")
    else:
        types = response_df["type"].unique().tolist()
        fig = go.Figure()
        for event_type in types:
            subset = response_df[response_df["type"] == event_type].sort_values("offset_weeks")
            fig.add_trace(go.Scatter(
                x=subset["offset_weeks"], y=subset["relative_hype"],
                mode="lines+markers", name=event_type,
            ))
        fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
        fig.add_hline(y=1.0, line_dash="dot", line_color=MUTED)
        fig.update_layout(
            title="Average attention shape by event type (weeks relative to event)",
            xaxis_title="Weeks from event", yaxis_title="Relative to pre-event baseline",
            height=460,
        )
        st.plotly_chart(style_fig(fig), width="stretch")

        st.markdown(
            f'<div class="hype-panel warn">Some event types here have only '
            f'one or two examples — their "average shape" is really just '
            f'that one event\'s shape, not a validated pattern for the '
            f'category in general. Read these as descriptive leads worth '
            f'checking on more events, not confirmed archetypes.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section — Signal Lag
# ---------------------------------------------------------------------------

elif section == "Signal Lag":
    st.markdown('<span class="hype-tag">PROPAGATION</span>', unsafe_allow_html=True)
    st.title("Signal Lag")
    st.markdown(
        "Does search curiosity, Wikipedia lookups, or YouTube activity tend "
        "to move first? I tested cross-correlation at different weekly lags "
        "between each pair of signals to see which one tends to lead."
    )

    lag_df = load_lag_analysis()
    if lag_df is None:
        missing_data_notice("lag_analysis.py output")
    else:
        pairs = lag_df["pair"].unique().tolist()
        selected_pair = st.selectbox("Signal pair", pairs)
        subset = lag_df[lag_df["pair"] == selected_pair]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=subset["lag_weeks"], y=subset["correlation"],
            mode="lines+markers", line=dict(color=CYAN, width=2),
            marker=dict(size=8),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
        fig.update_layout(
            title=f"Cross-correlation by lag: {selected_pair}",
            xaxis_title="Lag (weeks)", yaxis_title="Correlation", height=420,
        )
        st.plotly_chart(style_fig(fig), width="stretch")

        st.markdown(
            f'<div class="hype-panel warn">A lag of 0 at weekly resolution '
            f'doesn\'t mean nothing led anything — it means any real ordering '
            f'happens faster than a week, which this data can\'t resolve. '
            f'These are observed patterns, not a causal mechanism, and '
            f'correlations near zero mean the two signals just aren\'t '
            f'related at any tested lag.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section — The Conversation
# ---------------------------------------------------------------------------

elif section == "The Conversation":
    st.markdown('<span class="hype-tag">TOPIC DISCOVERY</span>', unsafe_allow_html=True)
    st.title("The Conversation")
    st.markdown(
        "I couldn't get approved API access to Reddit in time for this "
        "build, so this layer runs on YouTube titles and descriptions "
        "instead. Topics were discovered automatically — sentence "
        "embeddings plus clustering — then labeled afterward using each "
        "cluster's top terms. I never defined the categories up front."
    )

    videos, weekly, lookup = load_youtube_topics()
    if videos is None:
        missing_data_notice("YouTube topic analysis")
    else:
        st.markdown("#### What the clusters turned out to be")
        st.dataframe(lookup, width="stretch")

        cluster_counts = videos["topic_cluster"].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index.astype(str), y=cluster_counts.values,
            labels={"x": "Cluster", "y": "Videos"},
            title="Videos per topic cluster",
        )
        st.plotly_chart(style_fig(fig), width="stretch")

        st.markdown("#### Dominant topic, week by week")
        weekly_display = weekly.dropna(subset=["dominant_topic_terms"])
        st.dataframe(
            weekly_display[["date", "dominant_topic_terms", "video_count"]],
            width="stretch",
        )

        st.markdown(
            f'<div class="hype-panel">Titles and descriptions reflect how '
            f'creators frame content to get clicks, not neutral audience '
            f'discussion — a narrower signal than Reddit would\'ve given me. '
            f'This also isn\'t "the conversation" on YouTube — it\'s topic '
            f'patterns in the {len(videos)} videos my search queries actually '
            f'returned, which is a sample, not a census. A different set of '
            f'queries would pull a different sample.</div>',
            unsafe_allow_html=True,
        )

        diag_df = load_topic_diagnostics()
        if diag_df is not None:
            st.markdown("#### Why these particular clustering settings")
            st.markdown(
                "HDBSCAN doesn't assume a fixed number of topics — but its "
                "results depend on real choices: which text to embed, "
                "whether to reduce dimensionality first, and how large a "
                "group has to be to count as a real cluster. I tested a grid "
                "of these rather than picking one and hoping it worked."
            )
            st.dataframe(diag_df, width="stretch")
            viable = diag_df[(diag_df["n_clusters"] >= 2) & (diag_df["noise_pct"] < 50)]
            if viable.empty:
                st.markdown(
                    f'<div class="hype-panel warn">No configuration in this '
                    f'grid found multiple clusters with under 50% noise. '
                    f'That\'s the honest result — this sample doesn\'t have '
                    f'strong enough semantic separation for density-based '
                    f'clustering, and the topic groupings above should be '
                    f'read with that in mind.</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Section — Sentiment
# ---------------------------------------------------------------------------

elif section == "Sentiment":
    st.markdown('<span class="hype-tag">TONE</span>', unsafe_allow_html=True)
    st.title("Sentiment")
    st.markdown(
        "Scored with VADER, a lexicon-based analyzer built for short, "
        "informal, social-style text — a better fit for YouTube titles "
        "than a general-purpose model trained on formal writing."
    )

    videos, weekly, lookup = load_youtube_topics()
    if videos is None:
        missing_data_notice("YouTube sentiment analysis")
    else:
        counts = videos["sentiment_label"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", int(counts.get("positive", 0)))
        col2.metric("Neutral", int(counts.get("neutral", 0)))
        col3.metric("Negative", int(counts.get("negative", 0)))

        fig = px.pie(
            names=counts.index, values=counts.values,
            title="Overall sentiment split",
            color=counts.index,
            color_discrete_map={"positive": CYAN, "neutral": GOLD, "negative": PINK},
            hole=0.45,
        )
        st.plotly_chart(style_fig(fig), width="stretch")

        st.markdown("#### Average sentiment, week by week")
        trend_fig = px.line(
            weekly, x="date", y="avg_sentiment",
            labels={"avg_sentiment": "Avg. sentiment", "date": ""},
        )
        trend_fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
        st.plotly_chart(style_fig(trend_fig), width="stretch")

        st.markdown(
            f'<div class="hype-panel warn">This measures how creators frame '
            f'their content, not raw audience opinion — I\'m treating it as '
            f'a proxy, not ground truth.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("#### VADER vs. a transformer model — where do they disagree?")
        st.markdown(
            "Rather than reporting one sentiment model's numbers and hoping "
            "they're right, I ran a second model (a transformer trained on "
            "social-media text) alongside VADER and checked how often they "
            "actually agree."
        )

        examples_df, summary_df = load_sentiment_disagreement()
        if examples_df is None:
            missing_data_notice("sentiment_disagreement.py output")
        else:
            summary = summary_df.iloc[0]
            col1, col2 = st.columns(2)
            col1.metric("Model agreement rate", f"{summary['agreement_rate']:.1%}")
            col2.metric("Videos disagreeing", f"{len(examples_df)}+ examples saved")

            st.markdown(
                f'<div class="hype-panel warn">Agreement is far from perfect. '
                f'That\'s not a failure to hide — it\'s a real signal that '
                f'sentiment on short, informal gaming titles is genuinely '
                f'harder to pin down than a single accuracy number would '
                f'suggest.</div>',
                unsafe_allow_html=True,
            )

            st.markdown("Concrete disagreement examples (read a few of these directly):")
            st.dataframe(
                examples_df[["title", "vader_label", "transformer_label"]].head(10),
                width="stretch",
            )


# ---------------------------------------------------------------------------
# Section — The World
# ---------------------------------------------------------------------------

elif section == "The World":
    st.markdown('<span class="hype-tag">GEOGRAPHY</span>', unsafe_allow_html=True)
    st.title("The World")
    st.markdown(
        "I clustered countries by the **shape** of their search-interest "
        "trajectory, not raw volume — each country's series is normalized "
        "on its own first, so a smaller country with a sharp spike can "
        "land in the same group as a bigger one with the same pattern."
    )

    clusters, traj = load_country_clusters()
    if clusters is None:
        missing_data_notice("country clustering output")
    else:
        display_cols = ["country", "cluster"] + (["archetype"] if "archetype" in clusters.columns else [])
        st.markdown("#### Cluster assignments")
        st.dataframe(clusters[display_cols], width="stretch")

        traj_long = traj.T.reset_index().melt(id_vars="index", var_name="country", value_name="interest")
        traj_long = traj_long.rename(columns={"index": "date"})
        traj_long = traj_long.merge(clusters[["country", "cluster"]], on="country")
        traj_long["cluster"] = traj_long["cluster"].astype(str)

        fig = px.line(
            traj_long, x="date", y="interest", color="country", line_dash="cluster",
            labels={"interest": "Normalized interest (0–1)", "date": ""},
            title="Search-interest trajectory by country",
        )
        st.plotly_chart(style_fig(fig), width="stretch")

        n_countries = len(clusters)
        n_clusters_found = clusters["cluster"].nunique()
        st.markdown(
            f'<div class="hype-panel warn">Clustered on {n_countries} countries '
            f'into {n_clusters_found} groups, with the cluster count itself '
            f'chosen by silhouette score rather than assumed up front. More '
            f'countries would still sharpen this further, but {n_countries} '
            f'is enough to treat the groupings as a real starting signal, not '
            f'just an exploratory sketch.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Section — Findings
# ---------------------------------------------------------------------------

elif section == "Findings":
    st.markdown('<span class="hype-tag">WRITE-UP</span>', unsafe_allow_html=True)
    st.title("Findings")

    st.markdown("""
    ### Does combining signals actually help?

    Yes, but weakly, and not by adding everything. Google Trends alone hit
    F1=0.508 detecting real GTA VI events against my 19-event evaluation
    set (2 of the 21 registry events predate the signal data and are
    excluded — the detector has nothing to catch them with). Combining all
    three signals improved that to F1=0.572. But the ablation study —
    testing every single signal and every pair, not just the two endpoints
    — found something I didn't expect: **Trends + Wikipedia alone scored
    F1=0.599, higher than all three signals combined.** Adding YouTube
    didn't help and may have actively hurt. I'm keeping all three in the
    index for conceptual completeness (search, information-seeking, and
    creator activity are genuinely different things worth measuring), but
    the evidence says YouTube's contribution to *detection* specifically
    is questionable — likely because its "attention" signal is really
    cumulative lifetime views mixed with upload timing, not true weekly
    attention.

    I tested that hypothesis directly by splitting YouTube into two
    signals — creator activity (uploads/week, unique channels) and
    audience engagement (views and likes, log-scaled so one viral video
    can't dominate) — instead of one blended number. This was a
    diagnostic, not a fix: it explains *why* YouTube wasn't helping rather
    than producing a better detector than Trends + Wikipedia. The result
    confirmed the diagnosis: **creator activity alone scored F1=0.518,
    nearly matching Trends alone (0.508), while audience engagement alone
    scored just F1=0.171.** The original blended signal was genuinely
    diluting a decent sub-signal by averaging it with a nearly useless
    one. Even with this finer split, Trends + Wikipedia (F1=0.599) still
    beat every combination that included YouTube in any form — "more
    signals isn't automatically better" held up under closer scrutiny,
    not just at the coarse
    three-signal level.

    Bootstrapping 2,000 resamples of the event list gave F1=0.447
    [0.325, 0.537] for Trends alone versus F1=0.512 [0.384, 0.595] for the
    combined index. **Those intervals overlap.** So while combining signals
    looked better every single time I ran this, I can't call that
    difference statistically decisive with only 19 events — it's a
    real, consistent, but not yet proven advantage.

    ### Is the event alignment better than chance?

    Yes, decisively. A permutation test — comparing the real event dates
    against 1,000 sets of randomly placed fake dates, scored against the
    same detected spikes — gave an observed F1 of 0.572 against a
    random-chance mean of 0.296 (**p=0.003**). Only 3 of 1,000 random
    permutations scored as well as the real events did. The detector is
    finding something genuinely tied to real GTA VI events, not just
    landing near a busy calendar by coincidence.

    ### Does it generalize to events it wasn't tuned on?

    This is the most important — and most humbling — result in the
    project, and it came from doing exactly what I should have: testing
    generalization properly instead of reporting the flattering number.

    A chronological holdout (tune on the 12 earliest evaluable events,
    freeze parameters, test on the 7 most recent) collapsed from a dev F1
    of 0.500 to a **holdout F1 of 0.076**. That's a huge gap, and it means
    every F1 number elsewhere in this project carries some optimism from
    being tuned on the same events used to score it.

    But a single split is itself a small, noisy sample, so I ran
    leave-one-event-out cross-validation — 19 separate trials, each one
    tuning on the other 18 events and testing on the 1 left out. That gave
    a **68.4% catch rate**, meaningfully better than the single split
    suggested. So the chronological collapse wasn't purely representative
    of general performance — it was made worse by the specific 7 holdout
    events landing in an unusually event-dense 10-week stretch (cover art,
    promo update, pre-orders, a Netflix announcement, two leak waves, and
    the Extended Look all within about ten weeks), which a detector tuned
    on sparser earlier history wasn't built to resolve.

    A third test — holding out entire event *types* rather than a time
    window (tune on official/marketing/controversy, test on never-seen
    leak/rumor events) — collapsed to **F1=0.000**, identical to before
    the fix. That's a sharper, more specific finding than "it doesn't
    generalize": **the detector generalizes reasonably to a new individual
    event of a familiar type, but fails almost completely when asked to
    recognize an entire category it has zero examples of.** Rumors were
    also the weakest category in leave-one-out (33.3% catch rate, the
    lowest of all five types, unchanged by the fix) — consistent with this
    same pattern, since rumors don't share the clean single reference date
    that makes official announcements easy to detect.

    ### What different event types look like

    Response curves (attention 4 weeks before to 4 weeks after each event,
    normalized to that event's own pre-event baseline) showed real shape
    differences. Leaks spiked sharply (peaking at 4.5x baseline by week 2)
    and crashed hard by week 4 (down to 0.07x). Marketing events (cover
    art, pre-orders) produced a flatter, more sustained plateau — still
    elevated (1.6-2.9x) four weeks out — rather than a single sharp peak.
    Official announcements produced the largest peaks by far (7.8x at week
    1), though that average is dominated by a couple of unusually large
    events — with some event types having only 1-3 examples, I'm reading
    these as descriptive leads, not confirmed archetypes.

    ### A genuinely nice secondary finding: the three signals don't move together

    Google Trends tended to lead both Wikipedia and YouTube by about a
    week (r=0.59 and r=0.65 respectively). But Wikipedia and YouTube moved
    together in the *same* week, with no lead at all (r=0.79 — the
    strongest correlation of the three pairs). That's a specific,
    non-obvious asymmetry: search interest looks like it comes first,
    then information-seeking and creator activity respond together rather
    than in their own sequence. I wouldn't build the whole project around
    this, but it's a real, interpretable pattern in how a public-attention
    ecosystem moves — not a bare set of three lag numbers, an actual
    finding about structure. Read as an observed pattern, not a causal
    claim — people apparently search before they read or watch, which is
    intuitive, but weekly resolution can't rule out a different true
    ordering happening faster than a week.

    ### What's actually being discussed

    With a small YouTube sample (~130 videos), topic clustering found
    nothing — 100% noise. With a larger, more targeted sample (388 videos
    across 10 event-specific search queries), real structure emerged: 9
    interpretable clusters (leaked gameplay, cover art, pre-orders,
    delay/2026 news, release-date speculation, trailer reactions, trailer
    comparisons) covering about a third of the videos, with two-thirds
    still unclustered. A diagnostic grid search found an alternate
    configuration with much lower noise (18.6%) but only 2 broad clusters
    — a real tradeoff between granularity and confidence I'm stating
    rather than picking silently.

    Sentiment is genuinely uncertain: VADER and a transformer model agree
    on only about 43% of videos. The most common disagreement is VADER
    reading a title as positive while the transformer calls it neutral —
    consistent with VADER being more sensitive to exclamation marks and
    caps than the transformer model, not obviously a mixed-sentiment
    problem (a contrast-word check like "but"/"however" found no
    meaningful difference between agreeing and disagreeing videos).

    ### Geography

    25 countries auto-clustered into 5 groups by trajectory shape
    (silhouette-selected, not assumed). But my simple archetype labels
    (early responder / late riser / sustained / mid-cycle) collapsed to
    "late riser" for every single country — not a bug, but a real
    limitation of that heuristic: the Extended Look in August 2026 was by
    far the single biggest spike for essentially every country, so
    everyone's *global* peak falls late regardless of real differences
    earlier in their trajectories. The 5 numeric clusters likely capture
    real distinctions my simple labels can't express.

    ### What the Hype Index does NOT measure

    **Hype Index ≠ popularity.** It's a composite proxy for observable
    public attention (search, information-seeking, and creator activity),
    not a measurement of how much people actually like GTA VI. A delay
    announcement and a trailer can move the index by similar amounts even
    though one is bad news — the index measures attention magnitude, not
    sentiment direction.

    **Spike ≠ causality.** A detected spike means an unusual change in the
    measured signals that coincides temporally with a known event. It does
    not establish that the event caused the change — other things happen
    in the same weeks too. (The dashboard section that checks this is
    called "Spike Attribution," deliberately not "What Caused the Spikes.")

    **Event detection ≠ importance detection.** The detector is evaluated
    against my 21-event registry (19 of which fall inside the observable
    signal window). It cannot tell you whether an undetected week was
    actually unimportant, only that nothing in that week matched a
    documented event within the tolerance window.

    ### Where this falls short

    I didn't get Reddit API access in time, so the topic and sentiment
    layers run on YouTube titles and descriptions instead — a narrower
    signal measuring how creators pitch content, not what audiences
    actually think. The YouTube dataset is a search-result sample, not a
    census — a different set of queries would return a different sample.
    YouTube's "attention" signal is upload volume plus eventual lifetime
    views of that week's uploads, not real day-by-day view velocity, since
    the API only returns cumulative counts at collection time — this
    conflates creator activity with audience reach in a way a cleaner
    signal design would separate. Two of the 21 registered events predate
    the signal data entirely (Feb and Sept 2022) and are automatically
    excluded from every evaluation number above — a real bug I caught and
    fixed mid-project, not something designed in from the start. Even at
    19 evaluable events, every precision/recall/F1 number here is
    directional, not statistically decisive on its own — which is exactly
    why the permutation test, bootstrap CIs, and three separate holdout
    designs matter more than any single F1 score.

    None of that makes the project wrong. It means every number here comes
    with a specific, known boundary, and the generalization tests above are
    the most useful result precisely because they found a real limitation
    instead of a flattering one.
    """)


