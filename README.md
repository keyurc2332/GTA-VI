# GTA VI Hype Intelligence

**🔴 [Live demo](https://gta-vi.streamlit.app/)** — explore all 14 dashboard
sections yourself, no setup needed.

**I tested whether independent public signals — search interest,
information-seeking, and creator activity — can be combined into a
measurable proxy for public attention, and then tested where that measure
succeeds and fails at aligning with real-world GTA VI events.**

This is a retrospective event-alignment experiment, not a predictive hype
detector. I built it to find out — rather than assume — whether the
resulting measure detects real events better than a single signal would,
beats random chance, and holds up on events it never saw during tuning.

This is a weekend data-science project, not a production system. Several
results below are genuinely humbling rather than flattering. I kept them
in because they're the most useful part of the project.

## Headline results

*(from a real, corrected run — 2 events predating the signal data are now
excluded from evaluation; re-run the pipeline yourself to reproduce these
on fresh data, numbers will shift slightly as the dataset grows)*

**Does combining signals help?** Weakly, and not by using everything.
Trends alone: F1=0.508. All three signals combined: F1=0.572. But
**Trends + Wikipedia alone scored F1=0.599** — higher than all three
combined. Adding YouTube didn't help. Bootstrapped 95% confidence
intervals for Trends-only [0.325, 0.537] and combined [0.384, 0.595]
**overlap**, so "combining helps" is a consistent but not yet
statistically decisive finding at this sample size.

**Is the event alignment better than chance?** Yes, clearly. A
permutation test against 1,000 random-date trials gave **p=0.003** — the
real events align with detected spikes far better than chance would
produce (only 3 of 1,000 random trials scored as well).

**Does it generalize?** This is the project's most important result, and
it's not a good one — which is exactly why it matters. A chronological
holdout (tune on the 12 earliest evaluable events, test on the 7 most
recent) collapsed from dev F1=0.500 to **holdout F1=0.076**.
Leave-one-event-out cross-validation (19 independent trials) painted a
more nuanced picture: **68.4% catch rate** — better than the single split
suggested, meaning that split was partly just an unusually hard division.
But a third test — holding out entire event *types* (tune on
official/marketing/controversy, test on never-seen leak/rumor events) —
collapsed to **F1=0.000**. Put together: the detector generalizes
reasonably to a new individual event of a familiar type, but fails almost
completely on an entire category it has zero examples of.

**Does generalization hold under the strictest possible test?** Even
harder than any of the above: walk-forward evaluation, where each event is
tuned on strictly using only events that happened *before* it — no future
information allowed, unlike leave-one-out. This gave the lowest catch rate
of any generalization test in the project, well below leave-one-out's
68.4%, which suggests LOO's more optimistic number was partly inflated by
letting future events inform tuning decisions about the past.

**Does splitting YouTube reveal why it wasn't helping?** Yes — this
confirmed a real, specific diagnosis rather than a vague suspicion, but
it did not produce a better overall detector than Trends + Wikipedia.
**Creator activity alone scored F1=0.518** (nearly matching Trends alone
at 0.508), while **audience engagement alone scored just F1=0.171**. The
original blended YouTube signal was diluting a genuinely useful
sub-signal by averaging it with a nearly useless one — that's the
diagnostic insight. Even with this finer split, Trends + Wikipedia
(F1=0.599) still beat every combination that included YouTube in any
form.

**A genuinely nice secondary finding: the three signals don't move
together.** Google Trends tended to lead both Wikipedia and YouTube by
about a week (r=0.59 and r=0.65). But Wikipedia and YouTube moved
together in the *same* week, with no lead at all (r=0.79 — the strongest
correlation of the three pairs). That's a specific, non-obvious asymmetry
worth its own visualization in the dashboard's "Signal Lag" section, not
just a footnote — read as an observed pattern, not a causal claim.

## Architecture

```
                         GTA VI
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      Google Trends    Wikipedia       YouTube
      (search)       (pageviews)   (upload + view proxy)
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                  Signal normalization
                           │
                           ▼
              ┌────────────────────────┐
              │       Hype Index       │
              │ equal / PCA / rank-agg │
              └────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
  Spike detection    Event attribution    Signal lag
  (rolling z-score)  (verified registry)  (cross-correlation)
       │                   │
       ▼                   ▼
  Baseline vs.        Permutation test
  combined signal     (real vs. random
  (F1 comparison)      event alignment)
       │
       ▼
  Signal ablation (every signal + every pair)
       │
       ▼
  Detector sensitivity (window x threshold grid)
       │
       ▼
  Generalization testing:
    - Chronological holdout (earlier -> later)
    - Leave-one-event-out (19 independent trials)
    - Holdout by event type (official/marketing/controversy -> leak/rumor)
       │
       ▼
  Event impact (bootstrapped 95% CIs) + response curves by event type

YouTube text                          Google Trends (per-country)
     │                                        │
     ├── Sentence embeddings (MiniLM)         └── Trajectory clustering
     ├── L2-normalize + UMAP (with fallback)      (silhouette-selected k,
     ├── HDBSCAN (no fixed cluster count)          archetype labeling)
     ├── TF-IDF (interpretation only)
     ├── VADER + transformer sentiment
     └── Disagreement investigation

                           │
                           ▼
                    Streamlit dashboard (14 sections)
```

## Data sources

| Source | Signal | Auth needed | Coverage |
|---|---|---|---|
| Google Trends | Search curiosity | No | 3 keywords × 25 countries |
| YouTube Data API v3 | Creator/viewing attention proxy | Yes (free key) | 10 event-targeted queries |
| Wikipedia Pageviews | Information-seeking | No | Daily, full date range |

**Known limitation on YouTube:** the API returns *total* view count at
collection time, not day-by-day growth. The weekly signal is upload volume
plus eventual views of that week's uploads — this conflates creator
activity with lifetime audience reach, which a cleaner signal design would
separate. The ablation study suggests this may be why YouTube doesn't
improve detection despite adding real information.

**Known limitation on Reddit:** API access wasn't approved in time, so the
NLP/sentiment layer runs on YouTube titles and descriptions instead — a
narrower signal measuring creator framing, not audience discussion.

## What the Hype Index does NOT measure

**Hype Index ≠ popularity.** It's a composite proxy for observable public
attention (search, information-seeking, creator activity) — not a
measurement of how much people like GTA VI. A delay announcement and a
trailer can move the index by similar amounts even though one is bad news.
The index measures attention *magnitude*, not sentiment *direction* — the
event-response analysis confirmed this directly (see below).

**Spike ≠ causality.** A detected spike means an unusual change in the
measured signals coinciding temporally with a known event — never that
the event caused the change. Other things happen in the same weeks too.

**Event detection ≠ importance detection.** The detector is evaluated
against a 21-event registry I compiled (19 of which fall inside the
observable signal window — see below). It cannot tell you whether an
undetected week was genuinely unimportant, only that nothing in it matched
a documented event within the tolerance window.

## Event registry

`data/events.csv` — 21 independently documented GTA VI events, each with a
type (official/leak/rumor/controversy/marketing), a named source, and a
confidence rating (low/medium/high). I stopped at 21 rather than padding
to a round number with weakly-sourced rumors.

**Two of those 21 (Rockstar's Feb 2022 development confirmation and the
Sept 2022 leak) fall before the signal data's start date (Jan 2023) and
are automatically excluded from evaluation** — the detector has zero
Trends/Wikipedia/YouTube data to catch them with, so including them would
guarantee two misses regardless of detector quality. They're kept in the
registry file for historical context, but every F1/precision/recall number
in this project is computed against the **19 events actually inside the
observable signal window**.

## Methodology notes

**Spike detection first, attribution second.** The detector flags
statistically unusual weeks *before* checking which known events they
might correspond to — never the reverse.

**Every aggregation and parameter choice is compared, not asserted.**
Three Hype Index weightings (equal/PCA/rank), a full signal ablation
(every signal and pair, not just the two endpoints), and a parameter
sensitivity grid (window × threshold) are all tested explicitly rather
than picking one and presenting it as correct.

**Why the default detector uses window=8/threshold=1.5 even though the
sensitivity grid found window=8/threshold=2.0 scores higher (F1=0.595 vs
0.572).** The 1.5 threshold was the original exploratory choice, used
consistently across every other experiment in this project (baseline
comparison, permutation test, holdout tests, ablation) — changing it now
to whichever value scores best on the full event set would itself be a
form of tuning on the evaluation data, the exact problem the holdout
tests exist to catch. The sensitivity grid's job is to show that
performance depends on this choice, not to justify silently swapping in
whichever number looks best in hindsight.

**Generalization is tested three different ways, not assumed from one
flattering F1.** A single train/test split is itself a small, noisy
sample with only 19 evaluable events — using three different splitting
strategies (chronological, leave-one-out, by-category) is what actually
revealed *where* the detector's competence boundary is, rather than just
*whether* one exists.

**Event alignment is checked against random chance**, not assumed
meaningful just because spikes landed near events.

**Uncertainty is quantified**, not implied — bootstrapped confidence
intervals on F1, not bare point estimates.

## Setup

```bash
pip install -r requirements.txt
export YOUTUBE_API_KEY=your_key_here
```

## Run order

*(Not needed just to explore the results — the [live demo](https://gta-vi.streamlit.app/)
already has everything precomputed. This is for reproducing the pipeline
yourself or regenerating the data.)*

```bash
# 1. Collect raw data
python src/collect_google_trends.py    # ~15+ min across 25 countries
python src/collect_wikipedia.py
python src/collect_youtube.py

# 2. Hype Index + detection + evaluation
python src/build_hype_index.py
python src/detect_spikes.py
python src/evaluate_detection.py
python src/index_robustness.py
python src/spike_sensitivity.py
python src/permutation_test.py

# 3. Generalization testing
python src/event_holdout.py
python src/leave_one_event_out.py
python src/holdout_by_type.py

# 4. Ablation, impact, uncertainty
python src/signal_ablation.py
python src/bootstrap_ci.py
python src/event_impact.py
python src/event_response_curves.py
python src/lag_analysis.py

# 5. Geography
python src/cluster_countries.py

# 6. YouTube NLP (downloads 2 models on first run)
python src/analyze_youtube_text.py
python src/topic_diagnostics.py
python src/sentiment_disagreement.py

# 7. Dashboard
streamlit run src/app.py
```

Each script is idempotent and writes to `data/raw/` or `data/processed/`.

## Limitations, stated plainly

- **The core detector doesn't generalize to unseen event categories.**
  This is the single most important limitation, discovered through
  deliberate testing rather than avoided.
- **Reddit wasn't available** — text/sentiment analysis runs on YouTube
  titles/descriptions, a narrower signal than genuine audience discussion.
- **YouTube's "attention" signal conflates creator activity with lifetime
  view counts**, not real-time velocity, since the API only returns
  cumulative totals at collection time.
- **The YouTube dataset is a search-result sample**, not a census — a
  different set of queries would return a different sample.
- **19 evaluable events (of 21 total in the registry) is a modest
  evaluation set** — every F1/precision/recall number is directional at
  that sample size, which is exactly why the permutation test and
  bootstrap CIs matter more than any single score.
- **Topic clustering finds real structure only with a large enough
  sample** — 130 videos produced 100% noise; 388 videos produced 9
  interpretable clusters covering a third of the data. Reported honestly
  either way, not tuned until it looked presentable.
- **Sentiment models agree on only ~43% of videos** — treated as a
  genuine finding about the difficulty of the task, not averaged away.
- **Country archetype labels collapsed to one category** because a single
  dominant global event (the Extended Look) sits at the same relative
  timeline position for every country — a real limitation of a simple
  heuristic, not a data error.
- **Correlation and event-coincidence are not causation**, anywhere in
  this project.

## What this demonstrates

Time-series analysis, multi-source signal fusion with an honest ablation
study, statistical anomaly detection validated against a random-chance
null, three independent generalization tests that found and characterized
a real limitation, bootstrapped uncertainty quantification, semantic NLP
with sample-size-dependent honest failure reporting, sentiment model
comparison, geographic trajectory clustering — and, throughout,
investigating a project's own weak points instead of hiding them.