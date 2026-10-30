"""Judge the Model: a Streamlit app for a clinical lead deciding whether a classifier is fit for a job.

Run locally:  bash run_app.sh  (it runs streamlit run streamlit_app.py)
Every number comes from m11.py applied to the files in results/; the app never retrains a model.
Counts shown as dots (100 patients) are the test counts from m11 scaled to 100 and rounded.
"""

import math
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st

import m11

alt.data_transformers.disable_max_rows()

TRAIN = "Training"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]

POSITIVE = {
    "readmission": "readmitted to hospital within 30 days of discharge",
    "diabetes": "has a doctor diagnosis of diabetes, or an HbA1c of 6.5% or more",
}
GROUP = {"readmission": "patients", "diabetes": "adults"}
SENS_TARGET = {"diabetes": 0.90}  # the screening team's target; readmission has no fixed target

INK, CREAM, SAND, TEAL, BLUE, ORANGE, GREY = "#111111", "#F6F3EE", "#EDE8DF", "#1F7A8C", "#2E4A9E", "#E8683A", "#9A948A"

st.set_page_config(page_title="Judge the Model", page_icon="◖", layout="wide", initial_sidebar_state="collapsed")


# ---- look and feel ------------------------------------------------------------------
def blob_uri(fill: str, lobes: int, amp: float, rot: float = 0.0) -> str:
    """A scalloped blob as a CSS data-URI background (pure decoration)."""
    pts = []
    for i in range(240):
        a = 2 * math.pi * i / 240 + rot
        r = 43 + amp * abs(math.sin(lobes * a / 2))
        pts.append(f"{50 + r * math.cos(a):.2f},{50 + r * math.sin(a):.2f}")
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><polygon points='{' '.join(pts)}' fill='{fill}'/></svg>"
    return f'url("data:image/svg+xml,{quote(svg)}")'


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, td, th { font-family: 'Inter', sans-serif; color: #111; }
h1, h2, h3, .serif { font-family: 'Instrument Serif', 'Times New Roman', serif !important; font-weight: 400 !important; color: #111; }
.stApp { background: #F6F3EE; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stToolbar"], header { display: none !important; }
.block-container { padding: 1.2rem 2rem 4rem; max-width: 1280px; }
h2 { font-size: 2.6rem !important; line-height: 1.05; text-transform: uppercase; letter-spacing: 0.01em; margin-top: 1.6rem; }
h3 { font-size: 1.9rem !important; line-height: 1.1; margin-top: 1.4rem; }
a { color: #1F7A8C; }

/* top bar + hero */
.topbar { display: flex; justify-content: space-between; align-items: center; border: 1px solid #111; background: #F6F3EE; padding: 0.7rem 1.1rem; }
.brand { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 1.15rem; letter-spacing: -0.01em; }
.brand b { display: inline-block; width: 1.05rem; height: 1.05rem; border-radius: 50% 0 0 50%; background: #111; margin-right: 0.5rem; vertical-align: -2px; }
.topmeta { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; }
.hero { display: grid; grid-template-columns: 1fr 1.25fr 1fr; border: 1px solid #111; border-top: none; margin-bottom: 1.1rem; }
.hcell { position: relative; min-height: 340px; border-right: 1px solid #111; padding: 1.6rem; display: flex; flex-direction: column; justify-content: center; }
.hcell:last-child { border-right: none; }
.hero p.hx { font-family: 'Instrument Serif', serif !important; font-size: 4.6rem !important; line-height: 0.95; text-transform: uppercase; text-align: center; margin: 0; }
.hero p.hx em { text-transform: none; font-size: 4.2rem !important; }
.blob { background-repeat: no-repeat; background-position: center; background-size: contain; color: #fff; text-align: center; align-items: center; }
.hero .blob p { color: #fff !important; font-family: 'Instrument Serif', serif !important; font-size: 2rem !important; line-height: 1.12; margin: 0 auto; max-width: 17rem; }
.blob small { color: #fff; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.8rem; }
.toc a { display: block; text-decoration: none; color: #111; border-bottom: 1px solid #111; padding: 0.8rem 0; font-family: 'Instrument Serif', serif; font-size: 1.5rem; }
.toc a span { font-family: 'Inter', sans-serif; font-size: 0.75rem; margin-right: 0.7rem; }
.toc a:last-child { border-bottom: none; }
@media (max-width: 900px) { .hero { grid-template-columns: 1fr; } .hcell { border-right: none; border-bottom: 1px solid #111; } .hx { font-size: 3.4rem; } }

/* nav + controls (radios styled as hairline cells) */
.st-key-nav [role="radiogroup"], .st-key-dataset [role="radiogroup"], .st-key-model [role="radiogroup"],
.st-key-sortby [role="radiogroup"], .st-key-question [role="radiogroup"] { gap: 0; flex-wrap: wrap; border: 1px solid #111; background: #F6F3EE; width: 100%; }
.st-key-nav label, .st-key-dataset label, .st-key-model label, .st-key-sortby label, .st-key-question label
  { margin: 0 !important; padding: 0.55rem 1rem; border-right: 1px solid #111; flex: 1 1 auto; justify-content: center; cursor: pointer; transition: background .15s ease; }
.st-key-nav label { padding: 0.9rem 1rem; }
.st-key-nav label:last-child, .st-key-dataset label:last-child, .st-key-model label:last-child, .st-key-sortby label:last-child, .st-key-question label:last-child { border-right: none; }
[data-testid="stRadioOption"] > div > div:first-child { display: none !important; }
[data-testid="stRadioOption"] > div { justify-content: center; }
.st-key-nav label:hover, .st-key-dataset label:hover, .st-key-model label:hover, .st-key-sortby label:hover, .st-key-question label:hover { background: #EDE8DF; }
.st-key-nav label:has(input:checked), .st-key-dataset label:has(input:checked), .st-key-model label:has(input:checked),
.st-key-sortby label:has(input:checked), .st-key-question label:has(input:checked) { background: #111; }
.st-key-nav label:has(input:checked) p, .st-key-dataset label:has(input:checked) p, .st-key-model label:has(input:checked) p,
.st-key-sortby label:has(input:checked) p, .st-key-question label:has(input:checked) p { color: #F6F3EE !important; }
.st-key-nav label p { font-family: 'Instrument Serif', serif; font-size: 1.35rem; text-transform: uppercase; }
.ctl-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0.9rem 0 0.25rem; }

/* cards, metrics, tables */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 0; border: 1px solid #111 !important; background: #F6F3EE; }
[data-testid="stMetric"] { background: #F6F3EE; border: 1px solid #111; padding: 1rem 1.2rem; transition: background .15s ease, transform .15s ease; }
[data-testid="stMetric"]:hover { background: #EDE8DF; transform: translateY(-2px); }
[data-testid="stMetricValue"] { font-family: 'Instrument Serif', serif; font-size: 3rem; color: #1F7A8C; }
[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.75rem; }
[data-testid="stExpander"] { border: 1px solid #111; border-radius: 0; background: #F6F3EE; }
[data-testid="stExpander"] summary:hover { background: #EDE8DF; }
[data-testid="stTable"] table { border-collapse: collapse; width: 100%; border: 1px solid #111; }
[data-testid="stTable"] th { background: #EDE8DF; font-family: 'Instrument Serif', serif; font-size: 1.05rem; font-weight: 400; text-transform: uppercase; letter-spacing: 0.04em; }
[data-testid="stTable"] td, [data-testid="stTable"] th { border: 1px solid #111 !important; padding: 0.55rem 0.8rem; }
[data-testid="stTable"] tr:hover td { filter: brightness(0.97); }
[data-testid="stAlert"] { border-radius: 0; border: 1px solid #111; background: #EDE8DF; }
[data-testid="stAlert"] p { color: #111; }
.stButton > button { border-radius: 0; background: #111; color: #F6F3EE; border: 1px solid #111; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.78rem; padding: 0.55rem 1rem; transition: transform .15s ease, background .15s ease; }
.stButton > button:hover { background: #1F7A8C; border-color: #1F7A8C; color: #fff; transform: translateY(-2px); }
.stButton > button p { color: inherit !important; }
[data-testid="stSlider"] [role="slider"] { background: #1F7A8C; }

/* recommendation cards */
.rec { border: 1px solid #111; padding: 1.2rem 1.4rem; height: 100%; background: #F6F3EE; }
.rec.on { background: #EDE8DF; box-shadow: 6px 6px 0 #1F7A8C; }
.rec h4 { font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 1.7rem; margin: 0 0 0.5rem; text-transform: uppercase; }
.rec p { font-size: 0.95rem; line-height: 1.55; margin: 0; }

/* 100-patient dot grid */
.waffle { display: grid; grid-template-columns: repeat(10, 1.7rem); gap: 0.35rem; margin: 0.4rem 0 0.6rem; }
.dot { width: 1.7rem; height: 1.7rem; border-radius: 50%; border: 1px solid #111; transition: transform .15s ease, opacity .2s ease; }
.dot:hover { transform: scale(1.25); }
.dot.tp { background: #1F7A8C; } .dot.tn { background: #D9E6E8; } .dot.fn { background: #E8683A; } .dot.fp { background: #F6C9B8; }
.dot.dim { opacity: 0.14; }
.legend span { display: inline-block; margin-right: 1.1rem; font-size: 0.82rem; }
.legend i { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 50%; border: 1px solid #111; margin-right: 0.35rem; vertical-align: -1px; }
.legend .tp i { background: #1F7A8C; } .legend .tn i { background: #D9E6E8; } .legend .fn i { background: #E8683A; } .legend .fp i { background: #F6C9B8; }
</style>
"""
CSS += ("<style>.blob-teal { background-image: %s; } .blob-blue { background-image: %s; }</style>"
        % (blob_uri(TEAL, 9, 6.5), blob_uri(BLUE, 7, 6.5, 0.4)))
st.markdown(CSS, unsafe_allow_html=True)


def style_chart(ch):
    return (ch.configure(background="transparent").configure_view(strokeWidth=0)
            .configure_axis(labelColor="#111", titleColor="#111", gridColor="#DDD6C9", domainColor="#111", tickColor="#111")
            .configure_legend(labelColor="#111", titleColor="#111", orient="top", symbolStrokeColor="#111"))


# ---- shared helpers -------------------------------------------------------------------
@st.cache_data
def load(name: str):
    return m11.load(name)


def pct(v, d: int = 1) -> str:
    return m11.fmt_pct(v, d)


def per100(v) -> str:
    return m11.fmt_num(v, 1)


def matrix(c: dict, short: bool = False):
    """Course layout: rows = what actually happened, columns = what the model said, positives first.
    Correct cells (TP, TN) are teal-tinted and mistakes (FN, FP) are orange-tinted."""
    df = pd.DataFrame(
        [[f"TP = {c['tp']:,}", f"FN = {c['fn']:,}"], [f"FP = {c['fp']:,}", f"TN = {c['tn']:,}"]],
        index=["Actual +", "Actual −"] if short else ["Actual: positive", "Actual: negative"],
        columns=["Pred. +", "Pred. −"] if short else ["Predicted: positive (flagged)", "Predicted: negative (not flagged)"],
    )
    good, bad = "background-color: rgba(31,122,140,0.25)", "background-color: rgba(232,104,58,0.30)"
    return df.style.map(lambda v: good if v[:2] in ("TP", "TN") else bad)


def control_label(text: str) -> None:
    st.markdown(f'<div class="ctl-label">{text}</div>', unsafe_allow_html=True)


def how_to_read(text: str) -> None:
    with st.expander("How to read this page"):
        st.markdown(text)


def glossary() -> None:
    with st.expander("Plain-language glossary"):
        st.markdown(
            "- **Positive class:** the outcome we are trying to find (readmitted within 30 days, or diabetes).\n"
            "- **Probability:** the model's score for each patient, from 0 to 1.\n"
            "- **Threshold:** the cut-off. A patient is *flagged* when their probability is at least the threshold.\n"
            "- **TP / FN / FP / TN:** flagged and had the outcome / not flagged but had it (missed) / flagged but did not have it (false alarm) / not flagged and did not.\n"
            "- **Sensitivity (recall):** of the patients who had the outcome, the share we flagged.\n"
            "- **Specificity:** of the patients who did not, the share we left alone.\n"
            "- **Precision:** of the patients we flagged, the share who had the outcome.\n"
            "- **Accuracy:** the share of all patients the model got right. It can look high even when the model finds no one.\n"
            "- **Prevalence:** the share of all patients who have the outcome.\n"
            "- **AUROC:** how well a model *ranks* patients: the chance a patient with the outcome gets a higher probability than one without. 0.50 is chance, 1.00 is perfect. It is not accuracy and does not pick a threshold.\n"
            "- **Training / Test:** training patients are the ones a model learned from; test patients are held back and never seen. Judge a model on the test patients.")


def to_100(c: dict) -> dict:
    """The four test counts scaled to 100 dots (largest-remainder rounding)."""
    keys = ["tp", "fn", "fp", "tn"]
    n = sum(c[k] for k in keys)
    if n == 0:
        return {k: 0 for k in keys}
    raw = {k: 100 * c[k] / n for k in keys}
    out = {k: int(raw[k]) for k in keys}
    for k in sorted(keys, key=lambda k: raw[k] - out[k], reverse=True)[: 100 - sum(out.values())]:
        out[k] += 1
    return out


def waffle(c: dict, focus: set | None = None) -> None:
    d = to_100(c)
    dots = []
    for k in ("tp", "fn", "fp", "tn"):
        dim = " dim" if focus is not None and k not in focus else ""
        dots += [f'<div class="dot {k}{dim}" title="{k.upper()}"></div>'] * d[k]
    st.markdown('<div class="waffle">' + "".join(dots) + "</div>"
                '<div class="legend"><span class="tp"><i></i>TP caught</span><span class="fn"><i></i>FN missed</span>'
                '<span class="fp"><i></i>FP false alarm</span><span class="tn"><i></i>TN correctly left alone</span></div>',
                unsafe_allow_html=True)


@st.cache_data
def grid(key: str, model: str) -> list[dict]:
    """Metrics at every threshold 0.01 to 0.99 (test set), all from m11."""
    p, y = m11.preds(load(key), model)
    out = []
    for t in range(1, 100):
        c = m11.confusion(p, y, t / 100)
        out.append({"t": t / 100, "c": c, "m": m11.metrics(c)})
    return out


# ---- header, recommendations, controls --------------------------------------------------
def header() -> None:
    st.markdown(
        f"""<div class="topbar"><span class="brand"><b></b>Judge the Model</span>
        <span class="topmeta">{len(m11.DATASETS)} clinical jobs · {len(m11.MODELS)} models incl. a baseline · every table labelled Training or Test</span></div>
        <div class="hero">
          <div class="hcell"><p class="hx">Judge<br><em>the</em><br>Model</p></div>
          <div class="hcell blob blob-teal"><p>Which classifier would you trust for this job, at what threshold, and with what limits?</p></div>
          <div class="hcell"><div class="toc"><a><span>01</span>Beyond accuracy</a><a><span>02</span>A threshold for the job</a><a><span>03</span>Training vs. test</a></div></div>
        </div>""", unsafe_allow_html=True)


def recommendations(active: str) -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    st.markdown("### Recommendations")
    if not any(recs.values()):
        st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run `python -m analysis`, then commit and push.")
        return
    cols = st.columns(len(m11.DATASETS))
    for col, k in zip(cols, m11.DATASETS):
        with col:
            st.markdown(f'<div class="rec{" on" if k == active else ""}"><h4>{m11.DATASET_LABEL[k]}</h4>'
                        f'<p>{recs.get(k) or "<i>not written yet</i>"}</p></div>', unsafe_allow_html=True)


# ---- 1. Beyond Accuracy ----------------------------------------------------------------
QUESTIONS = {
    "Show all four groups": (None, "all"),
    "Who did we catch?  (sensitivity)": ({"tp", "fn"}, "sensitivity"),
    "Who did we leave alone?  (specificity)": ({"tn", "fp"}, "specificity"),
    "Whom we flagged, who really had it?  (precision)": ({"tp", "fp"}, "precision"),
}


def metric_rows(c: dict) -> list[tuple[str, str, str]]:
    """(name, value, 'of the patients who…' denominator in words) for the three headline metrics."""
    m = m11.metrics(c)
    return [
        ("Sensitivity (recall, TPR)", pct(m["sensitivity"]),
         f"TP ÷ (TP + FN) = {c['tp']:,} ÷ {c['tp'] + c['fn']:,}. Of the patients who had the outcome, the share the model flagged."),
        ("Specificity", pct(m["specificity"]),
         f"TN ÷ (TN + FP) = {c['tn']:,} ÷ {c['tn'] + c['fp']:,}. Of the patients who did not have the outcome, the share the model did not flag."),
        ("Precision", pct(m["precision"]),
         f"TP ÷ (TP + FP) = {c['tp']:,} ÷ {c['tp'] + c['fp']:,}. Of the patients the model flagged, the share who had the outcome."),
    ]


def beyond_accuracy(res: dict, model: str, key: str) -> None:
    p, y = m11.preds(res, model)
    c = m11.confusion(p, y, 0.5)
    m = m11.metrics(c)
    maj = m11.majority_accuracy(y)
    who = GROUP[key]

    how_to_read("1. Pick a **dataset** and a **model** at the top.\n"
                "2. Read the **positive class**: the outcome we are trying to find.\n"
                "3. Compare the model's accuracy with the 'most common class for everyone' accuracy. If they are close, accuracy is not telling you the model helps.\n"
                "4. Read the confusion matrix as four groups of patients, then tap a question under it to light up the dots that answer it.\n"
                "5. Scroll to the table that compares all five models at the same 0.50 threshold, and re-sort it by any metric.")
    st.markdown(f"**Positive class:** a patient counts as positive if they were **{POSITIVE[key]}**. "
                f"All numbers on this page are **{TEST}** results: {m['n']:,} {who} the model never saw while learning, at threshold 0.50.")

    st.markdown("### Why accuracy alone can mislead")
    a, b, d = st.columns(3)
    a.metric(f"Prevalence ({TEST})", pct(m["prevalence"]), help=f"Share of {who} who are positive: {m['positives']:,} of {m['n']:,}.")
    b.metric(f"Baseline accuracy ({TEST})", pct(maj),
             help="Baseline accuracy: the accuracy of “the most common class for everyone”, giving every patient the most common answer with no model at all.")
    d.metric(f"{m11.MODEL_LABEL[model]} accuracy ({TEST})", pct(m["accuracy"]),
             help="(TP + TN) ÷ all patients: the share of patients the model got right.")
    st.caption(f"Prevalence is {m['positives']:,} positive out of {m['n']:,} {who}. Baseline accuracy, the accuracy of “the most common class for everyone”, is "
               f"{pct(maj)}: that rule is right for that share with no model at all, so a model's accuracy only means something next to that number. "
               f"{m11.MODEL_LABEL[model]} catches {c['tp']:,} of the {m['positives']:,} positive {who}; "
               "compare how many positives each approach finds, not just how many it gets right.")

    st.markdown(f"### {m11.MODEL_LABEL[model]}: confusion matrix at 0.50 ({TEST})")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.caption(f"{TEST} · rows = what actually happened, columns = what the model said. Positive means {POSITIVE[key]}.")
        st.table(matrix(c))
        st.markdown(
            f"- **TP (true positive):** {c['tp']:,} {who} flagged who had the outcome.\n"
            f"- **FN (false negative):** {c['fn']:,} {who} not flagged who had the outcome (missed).\n"
            f"- **FP (false positive):** {c['fp']:,} {who} flagged who did not have the outcome (false alarms).\n"
            f"- **TN (true negative):** {c['tn']:,} {who} not flagged who did not have the outcome.")
    with right:
        st.markdown("**Picture 100 of these patients**")
        st.caption(f"{TEST} · 0.50 · the four counts scaled to 100 dots. Tap a question to light up the answer.")
        q = st.radio("Question", list(QUESTIONS), key="question", label_visibility="collapsed")
        focus, kind = QUESTIONS[q]
        waffle(c, focus)
        rows = {r[0].split(" ")[0].lower(): r for r in metric_rows(c)}
        if kind in rows:
            st.info(f"**{rows[kind][0]} = {rows[kind][1]}.** {rows[kind][2]}")
        else:
            st.info(f"Of the {m['positives']:,} {who} who had the outcome, {m11.MODEL_LABEL[model]} flagged {c['tp']:,} and missed {c['fn']:,}. "
                    f"Of the {m['negatives']:,} who did not, it left {c['tn']:,} alone and flagged {c['fp']:,} by mistake ({TEST}, threshold 0.50).")

    st.markdown(f"### Sensitivity, specificity and precision ({TEST}, 0.50)")
    st.table(pd.DataFrame([{"Metric": n, "Value": v, "Of the patients who…": w} for n, v, w in metric_rows(c)]).set_index("Metric"))
    if m["flagged"] == 0:
        st.caption("This model flagged no one at 0.50, so precision is 0 ÷ 0, which is undefined (not 0).")

    st.markdown(f"### All five models at 0.50 ({TEST})")
    st.caption(f"{TEST} · same {m['n']:,} {who} for every model. Undefined means a 0 ÷ 0 (for example, precision when no one is flagged).")
    control_label("Sort the table by")
    sort = st.radio("Sort", ["Model order", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1"], key="sortby", label_visibility="collapsed")
    entries = {}
    for name in m11.MODELS:
        if name in res["models"]:
            pp, yy = m11.preds(res, name)
            cc = m11.confusion(pp, yy, 0.5)
            entries[name] = (cc, m11.metrics(cc))
    order = list(entries)
    if sort != "Model order":
        order.sort(key=lambda n: (entries[n][1][sort.lower()] is None, -(entries[n][1][sort.lower()] or 0)))
    rows = []
    for name in order:
        mm = entries[name][1]
        rows.append({"Model": m11.MODEL_LABEL[name], "Accuracy": pct(mm["accuracy"]), "Sensitivity (recall, TPR)": pct(mm["sensitivity"]),
                     "Specificity": pct(mm["specificity"]), "FPR (1 − specificity)": pct(mm["fpr"]),
                     "Precision": pct(mm["precision"]), "F1": m11.fmt_num(mm["f1"])})
    for name in m11.MODELS:
        if name not in entries:
            rows.append({"Model": m11.MODEL_LABEL[name] + " (not trained yet)"})
    st.table(pd.DataFrame(rows).set_index("Model").fillna("–"))
    st.caption("Accuracy ranks differently from F1 and sensitivity. Read the whole row, not one column.")
    cols = st.columns(len(order))
    for col, name in zip(cols, order):
        with col:
            st.markdown(f"**{m11.MODEL_LABEL[name]}**")
            st.caption(f"{TEST} · 0.50 · rows actual, columns predicted")
            st.table(matrix(entries[name][0], short=True))


# ---- 2. A Threshold for the Job ---------------------------------------------------------
def delta(new, old, as_pct: bool = True) -> str:
    if new is None or old is None:
        return "undefined"
    d = (new - old) * (100 if as_pct else 1)
    return f"{d:+.1f}" + (" pts" if as_pct else "")


def set_thr(skey: str, value: float) -> None:
    st.session_state[skey] = value


def strip(p, y, label: str, color: str, thr: float):
    df = pd.DataFrame([{"Probability": pi, "jit": ((i * 37) % 100) / 100} for i, (pi, yi) in enumerate(zip(p, y)) if (yi == 1) == (label == "Had the outcome")])
    pts = alt.Chart(df).mark_circle(size=22, opacity=0.5, color=color).encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title=f"Model probability ({TEST})"),
        y=alt.Y("jit:Q", axis=None, scale=alt.Scale(domain=[0, 1])), tooltip=[alt.Tooltip("Probability:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"Probability": [thr]})).mark_rule(color=INK, strokeWidth=2).encode(x="Probability:Q")
    return (pts + rule).properties(height=80, title=f"{label} ({len(df):,} {TEST.lower()} patients)")


def threshold(res: dict, model: str, key: str) -> None:
    p, y = m11.preds(res, model)
    who = GROUP[key]
    G = grid(key, model)
    st.markdown(f"All numbers on this page are **{TEST}** results ({len(y):,} {who} the model never saw while learning). "
                "The **threshold** is the probability above which we flag a patient. It is a decision made separately from the model: "
                "moving it changes who is flagged, never the probabilities or the ranking.")
    how_to_read("1. Choose a **dataset** and **model** at the top.\n"
                "2. Move the **threshold slider**, or tap a shortcut. A lower threshold flags more patients: it finds more positives but raises false alarms.\n"
                "3. Watch the **dots**: patients to the right of the black line are flagged.\n"
                "4. Compare the **0.50** column with the **slider** column. Flagged per 100 shows the workload.\n"
                "5. The **ROC curve** and the line chart show every threshold at once. The ranking (AUROC) never changes.")
    if key == "diabetes":
        st.info("**The diabetes job:** finding positives matters most. The screening team wants a sensitivity of at least 0.90.")
    else:
        st.info("**The readmission job:** there is no fixed target. See what each threshold gains and gives up, then choose.")

    skey = f"thr_{key}_{model}"
    st.session_state.setdefault(skey, 0.50)
    control_label("Shortcuts")
    b1, b2, b3, _ = st.columns([1, 1, 1, 1])
    b1.button("Reset to 0.50", on_click=set_thr, args=(skey, 0.50), key=f"b1_{key}_{model}")
    f1s = [(g["m"]["f1"], g["t"]) for g in G if g["m"]["f1"] is not None]
    if f1s:
        b2.button(f"Best F1 (threshold {max(f1s)[1]:.2f})", on_click=set_thr, args=(skey, max(f1s)[1]), key=f"b2_{key}_{model}")
    if key in SENS_TARGET:
        ok = [g["t"] for g in G if (g["m"]["sensitivity"] or 0) >= SENS_TARGET[key]]
        if ok:
            b3.button(f"Reach {SENS_TARGET[key]:.2f} sensitivity ({max(ok):.2f})", on_click=set_thr, args=(skey, max(ok)), key=f"b3_{key}_{model}")
        else:
            st.caption(f"No threshold from 0.01 to 0.99 reaches a sensitivity of {SENS_TARGET[key]:.2f} for this model.")
    thr = st.slider("Threshold (flag a patient when their probability is at least this)", 0.01, 0.99, step=0.01, key=skey)

    c0, c1 = m11.confusion(p, y, 0.5), m11.confusion(p, y, thr)
    m0, m1 = m11.metrics(c0), m11.metrics(c1)

    st.markdown(f"### Where the patients fall ({TEST})")
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]} · each dot is one patient. The black line is the threshold {thr:.2f}: dots on or to its right are flagged.")
    st.altair_chart(style_chart(alt.vconcat(strip(p, y, "Had the outcome", ORANGE, thr), strip(p, y, "Did not have the outcome", BLUE, thr))), width="stretch")

    st.markdown(f"### Side by side ({TEST}): 0.50 vs. {thr:.2f}")
    l, r = st.columns(2)
    for col, c, label in ((l, c0, "0.50 (reference)"), (r, c1, f"{thr:.2f} (slider)")):
        with col:
            st.markdown(f"**Threshold {label}**")
            st.caption(f"{TEST} · {m11.MODEL_LABEL[model]} · rows actual, columns predicted")
            st.table(matrix(c))

    REF, NEW = "Threshold 0.50 (reference)", f"Threshold {thr:.2f} (slider)"

    def row(name, k, as_pct=True):
        f = pct if as_pct else per100
        return {"Metric": name, REF: f(m0[k]), NEW: f(m1[k]), "Change": delta(m1[k], m0[k], as_pct)}

    cmp = [row("Flagged per 100 patients", "flagged_per_100", False), row("Sensitivity (recall, TPR)", "sensitivity"),
           row("Specificity", "specificity"), row("FPR (1 − specificity)", "fpr"), row("Precision", "precision"),
           row("Accuracy", "accuracy"), row("F1", "f1")]
    cmp += [{"Metric": "Positives found (TP)", REF: f"{c0['tp']:,}", NEW: f"{c1['tp']:,}", "Change": f"{c1['tp'] - c0['tp']:+,}"},
            {"Metric": "False alarms (FP)", REF: f"{c0['fp']:,}", NEW: f"{c1['fp']:,}", "Change": f"{c1['fp'] - c0['fp']:+,}"}]
    st.caption(f"{TEST} · Flagged per 100 = of every 100 {who}, how many the model flags (changes in percentage points, pts).")
    st.table(pd.DataFrame(cmp).set_index("Metric"))

    nflag0, nflag1 = m0["flagged_per_100"], m1["flagged_per_100"]
    if m0["n"] and nflag0 is not None and nflag1 is not None:
        extra_fp = 100 * (c1["fp"] - c0["fp"]) / m1["n"]
        extra_tp = 100 * (c1["tp"] - c0["tp"]) / m1["n"]
        st.success(f"Compared with 0.50, per 100 {who} screened: **{nflag1 - nflag0:+.1f} flagged**, "
                   f"**{extra_tp:+.1f} more positives found**, **{extra_fp:+.1f} more false alarms** ({TEST}).")
    if key in SENS_TARGET:
        s = m1["sensitivity"]
        met = s is not None and s >= SENS_TARGET[key]
        st.markdown(f"Sensitivity target {SENS_TARGET[key]:.2f}: {'**met**' if met else '**not met**'} at {thr:.2f} (sensitivity {pct(s)}, {TEST}).")
        st.caption("This threshold was picked by looking at these test patients, so on new patients sensitivity will usually be slightly lower (peeking).")

    st.markdown(f"### Every threshold at once ({TEST})")
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]} · thresholds 0.01 to 0.99. Gaps in the precision line are thresholds where no one is flagged (precision undefined).")
    curve = pd.DataFrame([{"Threshold": g["t"], "Sensitivity (%)": None if g["m"]["sensitivity"] is None else 100 * g["m"]["sensitivity"],
                           "Precision (%)": None if g["m"]["precision"] is None else 100 * g["m"]["precision"],
                           "Flagged per 100": g["m"]["flagged_per_100"]} for g in G])
    long = curve.melt("Threshold", var_name="Measure", value_name="Value").dropna()
    line = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("Threshold:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Value:Q", title="Percent / per 100"),
        color=alt.Color("Measure:N", scale=alt.Scale(domain=["Sensitivity (%)", "Precision (%)", "Flagged per 100"], range=[TEAL, BLUE, ORANGE])),
        tooltip=["Threshold", "Measure", alt.Tooltip("Value:Q", format=".1f")])
    mark = alt.Chart(pd.DataFrame({"Threshold": [thr]})).mark_rule(strokeDash=[4, 4], color=INK).encode(x="Threshold:Q")
    st.altair_chart(style_chart(line + mark), width="stretch")
    st.caption("The dashed line is your slider's threshold.")

    st.markdown(f"### ROC curve ({TEST})")
    rc = m11.roc(p, y)
    roc_df = pd.DataFrame(list(zip(rc["fpr"], rc["tpr"])), columns=["False positive rate", "Sensitivity"])
    here = pd.DataFrame({"False positive rate": [m1["fpr"]], "Sensitivity": [m1["sensitivity"]]}).dropna()
    st.caption(f"{TEST} · each point on the curve is one threshold. The orange dot is your slider's threshold {thr:.2f}; the diagonal is a coin flip.")
    base = alt.Chart(roc_df).mark_line(color=TEAL, strokeWidth=2.5).encode(
        x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Sensitivity:Q", scale=alt.Scale(domain=[0, 1])))
    diag = alt.Chart(pd.DataFrame({"False positive rate": [0, 1], "Sensitivity": [0, 1]})).mark_line(color=GREY, strokeDash=[4, 4]).encode(
        x="False positive rate:Q", y="Sensitivity:Q")
    dot = alt.Chart(here).mark_circle(size=170, color=ORANGE, opacity=1).encode(x="False positive rate:Q", y="Sensitivity:Q")
    st.altair_chart(style_chart(base + diag + dot).properties(height=320), width="stretch")

    st.markdown("### What the slider does not change")
    st.markdown(f"{TEST} AUROC for {m11.MODEL_LABEL[model]}: **{m11.fmt_num(rc['auroc'])}**. It is the same at every threshold. "
                "AUROC describes ranking (how often a patient with the outcome gets a higher probability than one without); "
                "it is not accuracy, and it does not choose a threshold.")


# ---- 3. Training vs. Test ----------------------------------------------------------------
def train_vs_test(res: dict, model: str, key: str) -> None:
    s = m11.auroc_summary(res)
    who = GROUP[key]
    n_tr, n_te = res["train"]["n"], res["test"]["n"]
    st.markdown("A model is judged on patients it never saw. AUROC describes **ranking**: how often a patient with the outcome gets a higher "
                "probability than one without (0.50 is no better than chance, 1.00 is perfect). It is not accuracy.")
    how_to_read("1. Choose a **dataset** at the top.\n"
                "2. In the table, compare each model's **Training** AUROC (patients it learned from) with its **Test** AUROC (patients it never saw).\n"
                "3. A big **gap** means the model memorised its training patients. Trust the **Test** column.\n"
                "4. The chart shows the same thing: a grey dot for Training, a teal dot for Test.\n"
                "5. Try the guessing game at the bottom.")
    a, b = st.columns(2)
    a.metric(f"{TRAIN} patients", f"{n_tr:,}", help=f"{res['train']['n_pos']:,} positive. Each model learned from these {who}.")
    b.metric(f"{TEST} patients", f"{n_te:,}", help=f"{res['test']['n_pos']:,} positive. Each model never saw these {who} while learning.")
    st.caption(f"{TRAIN} AUROC rests on {n_tr:,} {who} ({res['train']['n_pos']:,} positive); {TEST} AUROC rests on {n_te:,} {who} ({res['test']['n_pos']:,} positive). "
               "With fewer patients, an AUROC can move by chance.")

    order = sorted(s, key=lambda m: (s[m]["test"] is None, -(s[m]["test"] or 0)))
    rows = [{"Model": m11.MODEL_LABEL[m] + (" ◀ selected" if m == model else ""), f"{TRAIN} AUROC": m11.fmt_num(s[m]["train"]),
             f"{TEST} AUROC": m11.fmt_num(s[m]["test"]), "Gap (Training − Test)": m11.fmt_num(s[m]["gap"])} for m in order]
    st.markdown("### Training and test AUROC side by side")
    st.caption(f"{TRAIN} and {TEST} · sorted by {TEST} AUROC, because that is the one to expect on new patients. "
               f"{TRAIN} AUROC flatters every model: it is measured on the patients the model learned from.")
    st.table(pd.DataFrame(rows).set_index("Model"))

    st.markdown("### Training vs. test, model by model")
    st.caption(f"{TRAIN} (grey) and {TEST} (teal) AUROC for each model. The longer the line between the dots, the bigger the gap.")
    pts = pd.DataFrame([{"Model": m11.MODEL_LABEL[m], "Partition": part, "AUROC": s[m][k], "Training": s[m]["train"], "Test": s[m]["test"]}
                        for m in order if s[m]["gap"] is not None for part, k in ((TRAIN, "train"), (TEST, "test"))])
    if len(pts):
        names = [m11.MODEL_LABEL[m] for m in order]
        base = alt.Chart(pts).encode(y=alt.Y("Model:N", sort=names, title=None, axis=alt.Axis(labelLimit=260)))
        rule = base.mark_rule(color=INK, strokeWidth=2).encode(x=alt.X("Training:Q", scale=alt.Scale(domain=[0.4, 1.0]), title="AUROC"), x2="Test:Q")
        dots = base.mark_circle(size=190, opacity=1, stroke=INK, strokeWidth=1).encode(
            x="AUROC:Q", color=alt.Color("Partition:N", scale=alt.Scale(domain=[TRAIN, TEST], range=[GREY, TEAL])),
            tooltip=["Model", "Partition", alt.Tooltip("AUROC:Q", format=".3f")])
        st.altair_chart(style_chart(rule + dots), width="stretch")

    if model in s and s[model]["gap"] is not None:
        v = s[model]
        st.info(f"**{m11.MODEL_LABEL[model]}:** {TRAIN} AUROC {m11.fmt_num(v['train'])}, {TEST} AUROC {m11.fmt_num(v['test'])}, gap {m11.fmt_num(v['gap'])}. "
                + ("On patients it had already seen it ranked far better than on new patients, a sign it memorised them. "
                   if v["gap"] >= 0.1 else "The gap is small, so it ranked training and new patients about equally well. ")
                + f"Expect the {TEST} AUROC on new patients, not the {TRAIN} one.")
    st.caption("A test AUROC a little above the training AUROC can happen by chance when the test set is small.")

    fitted = [m for m in m11.FITTED if m in s and s[m]["test"] is not None]
    if fitted:
        st.markdown("### Guessing game")
        st.caption(f"Only the {TRAIN} AUROC tempts you. Which model would you expect to rank new patients best?")
        guess = st.radio("Your guess", fitted, index=None, format_func=lambda m: m11.MODEL_LABEL[m], horizontal=True, key="guess")
        if guess:
            best_tr = max(fitted, key=lambda m: s[m]["train"])
            best_te = max(fitted, key=lambda m: s[m]["test"])
            rank_te = sorted(fitted, key=lambda m: -s[m]["test"]).index(guess) + 1
            st.success(f"Best on {TRAIN} patients: **{m11.MODEL_LABEL[best_tr]}** ({m11.fmt_num(s[best_tr]['train'])}). "
                       f"Best on {TEST} patients: **{m11.MODEL_LABEL[best_te]}** ({m11.fmt_num(s[best_te]['test'])}). "
                       f"Your pick, {m11.MODEL_LABEL[guess]}, ranks #{rank_te} of {len(fitted)} on {TEST} patients.")


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

# ---- page ----------------------------------------------------------------------------------
header()
control_label("Section")
section = st.radio("Section", SECTIONS, horizontal=True, key="nav", label_visibility="collapsed")
c1, c2 = st.columns([3, 5])
with c1:
    control_label("Dataset")
    key = st.radio("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get, horizontal=True, key="dataset", label_visibility="collapsed")
res = load(key)
with c2:
    control_label("Model")
    model = st.radio("Model", m11.MODELS, horizontal=True, key="model", label_visibility="collapsed",
                     format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
recommendations(key)
glossary()
st.markdown(f"## {section}")
st.caption(f"{m11.DATASET_LABEL[key]}: {res['train']['n']:,} training rows ({res['train']['n_pos']:,} positive), "
           f"{res['test']['n']:,} test rows ({res['test']['n_pos']:,} positive).")
if model in res["models"] or section == SECTIONS[2]:
    RENDER[section](res, model, key)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
