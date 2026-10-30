"""Judge the Model: a Streamlit app for a clinical lead deciding whether a classifier is fit for a job.

Run locally:  bash run_app.sh  (it runs streamlit run streamlit_app.py)
Every number comes from m11.py applied to the files in results/; the app never retrains a model.
The 100-dot picture scales the test counts from m11 to 100 dots (rounded).
"""

import math
from urllib.parse import quote

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

INK, TEAL, BLUE, ORANGE, GREY = "#111111", "#1F7A8C", "#2E4A9E", "#E8683A", "#9A948A"

st.set_page_config(page_title="Judge the Model", page_icon="◖", layout="wide", initial_sidebar_state="expanded")


# ---- look and feel ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label, td, th { font-family: 'Inter', sans-serif; }
h1, h2, h3, .serif { font-family: 'Instrument Serif', 'Times New Roman', serif !important; font-weight: 400 !important; letter-spacing: -0.005em; }
.stApp { background: #F6F3EE; }
[data-testid="stToolbar"], header { display: none !important; }
[data-testid="stSidebar"] { background: #EDE8DF; border-right: 1px solid rgba(17,17,17,0.12); }
.block-container { padding: 1.6rem 2.2rem 4rem; max-width: 1050px; }
h2 { font-size: 2.3rem !important; line-height: 1.1; margin-top: 0.8rem; }
h3 { font-size: 1.6rem !important; line-height: 1.15; margin-top: 1.8rem; }

.hero { position: relative; overflow: hidden; background: #EDE8DF; border: 1px solid rgba(17,17,17,0.15); border-radius: 20px; padding: 1.4rem 2rem 2.8rem; margin-bottom: 1rem; }
.hero-blob { position: absolute; right: 2rem; top: 50%; width: 110px; height: 110px; transform: translateY(-50%); background-repeat: no-repeat; background-size: contain; }
.hero p.hero-title { font-family: 'Instrument Serif', serif !important; font-size: 3rem !important; line-height: 1; margin: 0; position: relative; }
.hero p.hero-title em { color: #1F7A8C; }
.hero p.hero-sub { position: relative; max-width: 34rem; font-size: 1rem !important; margin: 0.5rem 0 0; color: #3b3b3b; }

[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; border: 1px solid rgba(17,17,17,0.15) !important; background: #FBF9F5; }
[data-testid="stMetric"] { background: #FBF9F5; border: 1px solid rgba(17,17,17,0.15); border-radius: 16px; padding: 0.9rem 1.1rem; transition: transform .15s ease, border-color .15s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: #1F7A8C; }
[data-testid="stMetricValue"] { font-family: 'Instrument Serif', serif; font-size: 2.6rem; color: #1F7A8C; }
[data-testid="stExpander"] { border: 1px solid rgba(17,17,17,0.15); border-radius: 14px; background: #FBF9F5; }
[data-testid="stTable"] table { border-collapse: separate; border-spacing: 0; width: 100%; border: 1px solid rgba(17,17,17,0.2); border-radius: 12px; overflow: hidden; }
[data-testid="stTable"] th { background: #EDE8DF; font-weight: 600; font-size: 0.85rem; }
[data-testid="stTable"] td, [data-testid="stTable"] th { border-color: rgba(17,17,17,0.1) !important; padding: 0.5rem 0.8rem; }
[data-testid="stAlert"] { border-radius: 14px; }
.stButton > button { border-radius: 999px; background: #111; color: #F6F3EE; border: none; font-weight: 500; padding: 0.4rem 1.1rem; transition: transform .15s ease, background .15s ease; }
.stButton > button:hover { background: #1F7A8C; color: #fff; transform: translateY(-1px); }
.stButton > button p { color: inherit !important; }
.stTabs [data-baseweb="tab"] { font-weight: 500; }
.stTabs [aria-selected="true"] { color: #1F7A8C; }

.waffle { display: grid; grid-template-columns: repeat(10, 1.5rem); gap: 0.28rem; margin: 0.4rem 0 0.5rem; }
.dot { width: 1.5rem; height: 1.5rem; border-radius: 50%; border: 1px solid rgba(17,17,17,0.45); transition: transform .15s ease, opacity .2s ease; }
.dot:hover { transform: scale(1.2); }
.dot.tp { background: #1F7A8C; } .dot.tn { background: #D9E6E8; } .dot.fn { background: #E8683A; } .dot.fp { background: #F6C9B8; }
.dot.dim { opacity: 0.14; }
.legend span { display: inline-block; margin-right: 0.9rem; font-size: 0.78rem; }
.legend i { display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 50%; border: 1px solid rgba(17,17,17,0.45); margin-right: 0.3rem; vertical-align: -1px; }
.legend .tp i { background: #1F7A8C; } .legend .tn i { background: #D9E6E8; } .legend .fn i { background: #E8683A; } .legend .fp i { background: #F6C9B8; }
</style>
"""


def svg_uri(svg: str) -> str:
    return 'url("data:image/svg+xml,%s")' % quote(svg)


def hero_art() -> str:
    """Teal scalloped medallion with a white heartbeat line and a medical cross (decoration only)."""
    pts = " ".join(f"{50 + (43 + 6.5 * abs(math.sin(9 * a / 2))) * math.cos(a):.2f},{50 + (43 + 6.5 * abs(math.sin(9 * a / 2))) * math.sin(a):.2f}"
                   for a in (2 * math.pi * i / 240 for i in range(240)))
    return svg_uri(
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><polygon points='{pts}' fill='{TEAL}'/>"
        "<rect x='45' y='20' width='10' height='22' rx='2.5' fill='white'/><rect x='39' y='26' width='22' height='10' rx='2.5' fill='white'/>"
        "<polyline points='16,62 34,62 41,62 46,48 53,78 59,56 64,62 84,62' fill='none' stroke='white' stroke-width='3.4' stroke-linecap='round' stroke-linejoin='round'/></svg>")


def icon_art(n: int) -> str:
    """Small section icons: 1 confusion-matrix tiles, 2 threshold slider, 3 training vs. test bars."""
    if n == 0:
        body = (f"<rect x='8' y='8' width='23' height='23' rx='6' fill='{TEAL}'/><rect x='33' y='8' width='23' height='23' rx='6' fill='{ORANGE}'/>"
                f"<rect x='8' y='33' width='23' height='23' rx='6' fill='#F6C9B8'/><rect x='33' y='33' width='23' height='23' rx='6' fill='#BFD9DE'/>")
    elif n == 1:
        body = (f"<rect x='6' y='29' width='52' height='7' rx='3.5' fill='#D9D3C7'/><rect x='6' y='29' width='30' height='7' rx='3.5' fill='{TEAL}'/>"
                f"<circle cx='36' cy='32.5' r='9' fill='white' stroke='{INK}' stroke-width='2.5'/><circle cx='14' cy='14' r='4' fill='{ORANGE}'/>"
                f"<circle cx='24' cy='12' r='4' fill='{ORANGE}'/><circle cx='46' cy='50' r='4' fill='{BLUE}'/><circle cx='54' cy='47' r='4' fill='{BLUE}'/>")
    else:
        body = (f"<rect x='10' y='10' width='18' height='46' rx='5' fill='{GREY}'/><rect x='36' y='24' width='18' height='32' rx='5' fill='{TEAL}'/>"
                f"<line x1='6' y1='10' x2='58' y2='10' stroke='{INK}' stroke-width='2' stroke-dasharray='3 3'/>"
                f"<line x1='32' y1='10' x2='32' y2='24' stroke='{ORANGE}' stroke-width='3' stroke-linecap='round'/>")
    return svg_uri(f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>{body}</svg>")


ECG_TILE = svg_uri("<svg xmlns='http://www.w3.org/2000/svg' width='140' height='40' viewBox='0 0 140 40'>"
                   f"<polyline points='0,22 46,22 52,8 60,34 66,16 72,22 140,22' fill='none' stroke='{TEAL}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/></svg>")

CSS += ("<style>.hero-blob { background-image: %s; }"
        ".hero::after { content: ''; position: absolute; left: 0; right: 0; bottom: 6px; height: 36px; background-image: %s; background-repeat: repeat-x; opacity: .35; }"
        ".sec { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }"
        ".sec-icon { flex: none; width: 64px; height: 64px; border-radius: 18px; background-color: #EDE8DF; border: 1px solid rgba(17,17,17,0.15); background-size: 44px; background-repeat: no-repeat; background-position: center; }"
        ".sec-icon.i0 { background-image: %s; } .sec-icon.i1 { background-image: %s; } .sec-icon.i2 { background-image: %s; }"
        ".side-logo { display: flex; align-items: center; gap: .7rem; padding: .2rem 0 1rem; }"
        ".side-logo .art { width: 46px; height: 46px; background-image: %s; background-size: contain; background-repeat: no-repeat; }"
        ".side-logo b { font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 1.5rem; line-height: 1; }"
        "</style>") % (hero_art(), ECG_TILE, icon_art(0), icon_art(1), icon_art(2), hero_art())
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


def matrix(c: dict, short: bool = False):
    """Course layout: rows = what actually happened, columns = what the model said, positives first."""
    df = pd.DataFrame(
        [[f"TP = {c['tp']:,}", f"FN = {c['fn']:,}"], [f"FP = {c['fp']:,}", f"TN = {c['tn']:,}"]],
        index=["Actual +", "Actual −"] if short else ["Actual: positive", "Actual: negative"],
        columns=["Pred. +", "Pred. −"] if short else ["Predicted: positive (flagged)", "Predicted: negative (not flagged)"],
    )
    good, bad = "background-color: rgba(31,122,140,0.25)", "background-color: rgba(232,104,58,0.30)"
    return df.style.map(lambda v: good if v[:2] in ("TP", "TN") else bad)


def sidebar_help() -> None:
    with st.sidebar.expander("Help: what the words mean"):
        st.markdown(
            "- **Positive:** has the outcome we want to find.\n"
            "- **Flagged:** the model's probability is at least the **threshold**.\n"
            "- **TP / FN / FP / TN:** caught / missed / false alarm / correctly left alone.\n"
            "- **Sensitivity:** of those who had the outcome, the share flagged.\n"
            "- **Specificity:** of those who did not, the share not flagged.\n"
            "- **Precision:** of those flagged, the share who had it.\n"
            "- **Prevalence:** the share of all patients who have the outcome.\n"
            "- **AUROC:** how well a model *ranks* patients (0.50 is chance, 1.00 is perfect). It is not accuracy.\n"
            "- **Training / Test:** patients a model learned from / patients it never saw. Trust Test.")


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
                '<span class="fp"><i></i>FP false alarm</span><span class="tn"><i></i>TN left alone</span></div>',
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


# ---- header and recommendations ---------------------------------------------------------
def header() -> None:
    st.markdown(
        """<div class="hero"><div class="hero-blob"></div>
        <p class="hero-title">Judge the <em>model</em></p>
        <p class="hero-sub">Which classifier would you trust for this job, at what threshold, and with what limits?</p></div>""",
        unsafe_allow_html=True)


def recommendations(active: str) -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.values()):
            st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run `python -m analysis`, then commit and push.")
            return
        st.markdown(f"**{m11.DATASET_LABEL[active]}:** {recs.get(active) or '_not written yet_'}")
        other = [k for k in m11.DATASETS if k != active]
        with st.expander(f"See the {m11.DATASET_LABEL[other[0]].lower()} recommendation"):
            st.markdown(recs.get(other[0]) or "_not written yet_")


# ---- 1. Beyond Accuracy ----------------------------------------------------------------
QUESTIONS = {
    "All four groups": (None, "all"),
    "Who did we catch?": ({"tp", "fn"}, "sensitivity"),
    "Who did we leave alone?": ({"tn", "fp"}, "specificity"),
    "Of those flagged, who had it?": ({"tp", "fp"}, "precision"),
}


def metric_rows(c: dict, who: str) -> list[tuple[str, str, str]]:
    """(name, value, 'of the patients who…' denominator in words) for the three headline metrics."""
    m = m11.metrics(c)
    return [
        ("Sensitivity (recall, TPR)", pct(m["sensitivity"]),
         f"Of the {c['tp'] + c['fn']:,} {who} who had the outcome, {c['tp']:,} were flagged (TP ÷ (TP + FN))."),
        ("Specificity", pct(m["specificity"]),
         f"Of the {c['tn'] + c['fp']:,} {who} who did not have the outcome, {c['tn']:,} were not flagged (TN ÷ (TN + FP))."),
        ("Precision", pct(m["precision"]),
         f"Of the {c['tp'] + c['fp']:,} {who} who were flagged, {c['tp']:,} had the outcome (TP ÷ (TP + FP))."),
    ]


def beyond_accuracy(res: dict, model: str, key: str) -> None:
    p, y = m11.preds(res, model)
    c = m11.confusion(p, y, 0.5)
    m = m11.metrics(c)
    maj = m11.majority_accuracy(y)
    who = GROUP[key]

    st.markdown(f"**Positive class:** {who} who were **{POSITIVE[key]}**. "
                f"Everything here is **{TEST}** data ({m['n']:,} {who}) at threshold 0.50.")

    a, b, d = st.columns(3)
    a.metric(f"Prevalence ({TEST})", pct(m["prevalence"]), help=f"{m['positives']:,} of {m['n']:,} {who} are positive.")
    b.metric(f"Baseline accuracy ({TEST})", pct(maj),
             help="Baseline accuracy: the accuracy of “the most common class for everyone”, with no model at all.")
    d.metric(f"{m11.MODEL_LABEL[model]} accuracy ({TEST})", pct(m["accuracy"]), help="(TP + TN) ÷ all patients.")
    st.caption(f"Baseline accuracy is the accuracy of “the most common class for everyone”. A model has to do better than that to help: "
               f"{m11.MODEL_LABEL[model]} catches {c['tp']:,} of the {m['positives']:,} positive {who}.")

    st.markdown(f"### Confusion matrix at 0.50 ({TEST})")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.caption(f"{TEST} · {m11.MODEL_LABEL[model]} · rows = actual, columns = predicted")
        st.table(matrix(c))
    with right:
        st.caption(f"{TEST} · 100 {who} drawn from the counts. Pick a question:")
        q = st.radio("Question", list(QUESTIONS), key="question", label_visibility="collapsed")
        focus, kind = QUESTIONS[q]
        waffle(c, focus)

    st.markdown(f"### Sensitivity, specificity and precision ({TEST}, 0.50)")
    rows = metric_rows(c, who)
    st.table(pd.DataFrame([{"Metric": n, "Value": v, "Of the patients who…": w} for n, v, w in rows]).set_index("Metric"))
    if m["flagged"] == 0:
        st.caption("No one was flagged, so precision is 0 ÷ 0: undefined, not 0.")

    st.markdown(f"### All five models at 0.50 ({TEST})")
    sort = st.radio("Sort by", ["Model order", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1"], key="sortby", horizontal=True)
    entries = {}
    for name in m11.MODELS:
        if name in res["models"]:
            pp, yy = m11.preds(res, name)
            cc = m11.confusion(pp, yy, 0.5)
            entries[name] = (cc, m11.metrics(cc))
    order = list(entries)
    if sort != "Model order":
        order.sort(key=lambda n: (entries[n][1][sort.lower()] is None, -(entries[n][1][sort.lower()] or 0)))
    table = []
    for name in order:
        mm = entries[name][1]
        table.append({"Model": m11.MODEL_LABEL[name], "Accuracy": pct(mm["accuracy"]), "Sensitivity (recall, TPR)": pct(mm["sensitivity"]),
                      "Specificity": pct(mm["specificity"]), "FPR (1 − specificity)": pct(mm["fpr"]),
                      "Precision": pct(mm["precision"]), "F1": m11.fmt_num(mm["f1"])})
    for name in m11.MODELS:
        if name not in entries:
            table.append({"Model": m11.MODEL_LABEL[name] + " (not trained yet)"})
    st.caption(f"{TEST} · same {m['n']:,} {who} for every model · “undefined” means 0 ÷ 0")
    st.table(pd.DataFrame(table).set_index("Model").fillna("n/a"))
    cols = st.columns(len(order))
    for col, name in zip(cols, order):
        with col:
            st.markdown(f"**{m11.MODEL_LABEL[name]}**")
            st.caption(f"{TEST} · 0.50")
            st.table(matrix(entries[name][0], short=True))


# ---- 2. A Threshold for the Job ---------------------------------------------------------
def delta(new, old, as_pct: bool = True) -> str:
    if new is None or old is None:
        return "undefined"
    d = (new - old) * (100 if as_pct else 1)
    return f"{d:+.1f}" + (" pts" if as_pct else "")


def set_thr(skey: str, value: float) -> None:
    st.session_state[skey] = value
    st.session_state[skey + "_n"] = st.session_state.get(skey + "_n", 0) + 1


live_threshold = components.declare_component("live_threshold", path=str(Path(__file__).parent / "live_threshold"))


def live_payload(G: list[dict], key: str, who: str, model: str) -> dict:
    """Everything the live slider shows, for every threshold 0.01 to 0.99, computed here with m11.
    The slider in the browser only looks these up, so the numbers update while you drag."""
    ref = G[49]  # threshold 0.50
    c0, m0 = ref["c"], ref["m"]
    rows = {}
    for g in G:
        c1, m1 = g["c"], g["m"]

        def r(name, k, as_pct=True):
            f = pct if as_pct else (lambda v: m11.fmt_num(v, 1))
            return [name, f(m0[k]), f(m1[k]), delta(m1[k], m0[k], as_pct)]

        cmp = [r("Flagged per 100", "flagged_per_100", False), r("Sensitivity (recall, TPR)", "sensitivity"), r("Specificity", "specificity"),
               r("Precision", "precision"), r("Accuracy", "accuracy"), r("F1", "f1")]
        n0, n1 = m0["flagged_per_100"], m1["flagged_per_100"]
        summary = ""
        if m0["n"] and n0 is not None and n1 is not None:
            summary = (f"Per 100 {who} screened, compared with 0.50: <b>{n1 - n0:+.1f} flagged</b>, "
                       f"<b>{100 * (c1['tp'] - c0['tp']) / m1['n']:+.1f} more positives found</b>, "
                       f"<b>{100 * (c1['fp'] - c0['fp']) / m1['n']:+.1f} more false alarms</b> ({TEST}).")
        target = ""
        if key in SENS_TARGET:
            sv = m1["sensitivity"]
            met = sv is not None and sv >= SENS_TARGET[key]
            target = (f"Sensitivity target {SENS_TARGET[key]:.2f}: {'met' if met else 'not met'} at {g['t']:.2f} (sensitivity {pct(sv)}, {TEST}). "
                      "The threshold was chosen on these test patients, so expect slightly lower on new ones.")
        rows[str(round(g["t"] * 100))] = {"t": f"{g['t']:.2f}", "tv": g["t"], "c": c1, "cmp": cmp, "summary": summary, "target": target,
                                         "dots": to_100(c1)}
    series = [{"sensp": None if g["m"]["sensitivity"] is None else 100 * g["m"]["sensitivity"],  # percent, for the line chart
               "prec": None if g["m"]["precision"] is None else 100 * g["m"]["precision"],
               "flag": g["m"]["flagged_per_100"],
               "sens": g["m"]["sensitivity"], "fpr": g["m"]["fpr"]}  # 0-1, for the ROC curve
              for g in G]
    return {"rows": rows, "ref": c0, "series": series, "labels": {
        "slider": "Threshold: flag a patient when their probability is at least this",
        "capRef": f"{TEST} · {m11.MODEL_LABEL[model]} · threshold 0.50",
        "capNew": f"{TEST} · {m11.MODEL_LABEL[model]} · threshold {{t}} (slider)",
        "capTable": f"{TEST} · Flagged per 100 = of every 100 {who}, how many the model flags",
        "capDots": f"{TEST} · 100 {who} at threshold {{t}}",
        "capChart": f"{TEST} · every threshold", "capRoc": f"{TEST} · ROC curve",
        "row1": "Actual: positive", "row2": "Actual: negative",
        "col1": "Predicted: positive (flagged)", "col2": "Predicted: negative (not flagged)"}}


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
    if key == "diabetes":
        st.info(f"**The job:** finding positives matters most; the screening team wants a sensitivity of at least {SENS_TARGET[key]:.2f}. "
                f"Everything here is {TEST} data ({len(y):,} {who}).")
    else:
        st.info(f"**The job:** no fixed target; see what each threshold gains and gives up. Everything here is {TEST} data ({len(y):,} {who}).")

    skey = f"thr_{key}_{model}"
    st.session_state.setdefault(skey, 0.50)
    st.session_state.setdefault(skey + "_n", 0)
    b1, b2, b3, _ = st.columns([1, 1, 1.4, 0.6])
    b1.button("Reset to 0.50", on_click=set_thr, args=(skey, 0.50), key=f"b1_{key}_{model}")
    f1s = [(g["m"]["f1"], g["t"]) for g in G if g["m"]["f1"] is not None]
    if f1s:
        b2.button(f"Best F1 ({max(f1s)[1]:.2f})", on_click=set_thr, args=(skey, max(f1s)[1]), key=f"b2_{key}_{model}")
    if key in SENS_TARGET:
        ok = [g["t"] for g in G if (g["m"]["sensitivity"] or 0) >= SENS_TARGET[key]]
        if ok:
            b3.button(f"Reach {SENS_TARGET[key]:.2f} sensitivity ({max(ok):.2f})", on_click=set_thr, args=(skey, max(ok)), key=f"b3_{key}_{model}")
        else:
            st.caption(f"No threshold from 0.01 to 0.99 reaches a sensitivity of {SENS_TARGET[key]:.2f} for this model.")

    nonce = st.session_state[skey + "_n"]
    val = live_threshold(**live_payload(G, key, who, model), initial=st.session_state[skey], nonce=nonce,
                         key=f"live_{key}_{model}", default={"thr": st.session_state[skey], "nonce": nonce})
    if val and val.get("nonce") == nonce:
        st.session_state[skey] = val["thr"]
    thr = st.session_state[skey]

    rc = m11.roc(p, y)
    st.markdown("### Bigger charts")
    st.caption("These larger versions update when you let go of the slider.")
    tab1, tab2, tab3 = st.tabs(["Patient dots", "Every threshold", "ROC curve"])
    with tab1:
        st.caption(f"{TEST} · each dot is one patient; dots on or right of the black line (threshold {thr:.2f}) are flagged.")
        st.altair_chart(style_chart(alt.vconcat(strip(p, y, "Had the outcome", ORANGE, thr), strip(p, y, "Did not have the outcome", BLUE, thr))), width="stretch")
    with tab2:
        st.caption(f"{TEST} · thresholds 0.01 to 0.99; the dashed line is your slider. A gap in precision means no one was flagged (undefined).")
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
    with tab3:
        st.caption(f"{TEST} · each point is one threshold; the orange dot is your slider; the dashed diagonal is a coin flip.")
        roc_df = pd.DataFrame(list(zip(rc["fpr"], rc["tpr"])), columns=["False positive rate", "Sensitivity"])
        m1 = G[round(thr * 100) - 1]["m"]
        here = pd.DataFrame({"False positive rate": [m1["fpr"]], "Sensitivity": [m1["sensitivity"]]}).dropna()
        base = alt.Chart(roc_df).mark_line(color=TEAL, strokeWidth=2.5).encode(
            x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Sensitivity:Q", scale=alt.Scale(domain=[0, 1])))
        diag = alt.Chart(pd.DataFrame({"False positive rate": [0, 1], "Sensitivity": [0, 1]})).mark_line(color=GREY, strokeDash=[4, 4]).encode(
            x="False positive rate:Q", y="Sensitivity:Q")
        dot = alt.Chart(here).mark_circle(size=170, color=ORANGE, opacity=1).encode(x="False positive rate:Q", y="Sensitivity:Q")
        st.altair_chart(style_chart(base + diag + dot).properties(height=300), width="stretch")

    st.caption(f"{TEST} AUROC for {m11.MODEL_LABEL[model]}: {m11.fmt_num(rc['auroc'])}. It does not change with the threshold: "
               "AUROC describes how well the model ranks patients, not accuracy.")


# ---- 3. Training vs. Test ----------------------------------------------------------------
def train_vs_test(res: dict, model: str, key: str) -> None:
    s = m11.auroc_summary(res)
    who = GROUP[key]
    n_tr, n_te = res["train"]["n"], res["test"]["n"]
    st.caption("AUROC describes ranking: how often a patient with the outcome gets a higher probability than one without. It is not accuracy.")
    a, b = st.columns(2)
    a.metric(f"{TRAIN} patients", f"{n_tr:,}", help=f"{res['train']['n_pos']:,} positive. The patients each model learned from.")
    b.metric(f"{TEST} patients", f"{n_te:,}", help=f"{res['test']['n_pos']:,} positive. Patients no model saw while learning.")
    st.caption(f"{TRAIN} AUROC rests on {n_tr:,} {who} ({res['train']['n_pos']:,} positive); {TEST} AUROC on {n_te:,} {who} ({res['test']['n_pos']:,} positive).")

    order = sorted(s, key=lambda m: (s[m]["test"] is None, -(s[m]["test"] or 0)))
    rows = [{"Model": m11.MODEL_LABEL[m], f"{TRAIN} AUROC": m11.fmt_num(s[m]["train"]),
             f"{TEST} AUROC": m11.fmt_num(s[m]["test"]), "Gap (Training − Test)": m11.fmt_num(s[m]["gap"])} for m in order]
    st.markdown("### Training and test AUROC side by side")
    st.caption(f"{TRAIN} and {TEST} · sorted by {TEST} AUROC, the one to expect on new patients. A big gap means the model memorised its training patients.")
    st.table(pd.DataFrame(rows).set_index("Model"))

    fitted = [m for m in m11.FITTED if m in s and s[m]["test"] is not None]
    if fitted:
        best_tr = max(fitted, key=lambda m: s[m]["train"])
        best_te = max(fitted, key=lambda m: s[m]["test"])
        same = (f"and it is still best on {TEST} patients ({m11.fmt_num(s[best_te]['test'])})" if best_tr == best_te
                else f"but on {TEST} patients {m11.MODEL_LABEL[best_te]} is best ({m11.fmt_num(s[best_te]['test'])})")
        st.info(f"Best on {TRAIN} patients: **{m11.MODEL_LABEL[best_tr]}** ({m11.fmt_num(s[best_tr]['train'])}), {same}.")

    pts = pd.DataFrame([{"Model": m11.MODEL_LABEL[m], "Partition": part, "AUROC": s[m][k], "Training": s[m]["train"], "Test": s[m]["test"]}
                        for m in order if s[m]["gap"] is not None for part, k in ((TRAIN, "train"), (TEST, "test"))])
    if len(pts):
        st.caption(f"{TRAIN} (grey) and {TEST} (teal) AUROC · the longer the line, the bigger the gap")
        names = [m11.MODEL_LABEL[m] for m in order]
        base = alt.Chart(pts).encode(y=alt.Y("Model:N", sort=names, title=None, axis=alt.Axis(labelLimit=260)))
        rule = base.mark_rule(color=INK, strokeWidth=2).encode(x=alt.X("Training:Q", scale=alt.Scale(domain=[0.4, 1.0]), title="AUROC"), x2="Test:Q")
        dots = base.mark_circle(size=190, opacity=1, stroke=INK, strokeWidth=1).encode(
            x="AUROC:Q", color=alt.Color("Partition:N", scale=alt.Scale(domain=[TRAIN, TEST], range=[GREY, TEAL])),
            tooltip=["Model", "Partition", alt.Tooltip("AUROC:Q", format=".3f")])
        st.altair_chart(style_chart(rule + dots), width="stretch")

    if fitted:
        with st.expander("Test yourself: which model will rank new patients best?"):
            guess = st.radio("Your guess", fitted, index=None, format_func=lambda m: m11.MODEL_LABEL[m], horizontal=True, key="guess")
            if guess:
                rank_te = sorted(fitted, key=lambda m: -s[m]["test"]).index(guess) + 1
                st.success(f"{m11.MODEL_LABEL[guess]} ranks #{rank_te} of {len(fitted)} on {TEST} patients ({m11.fmt_num(s[guess]['test'])}).")


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

# ---- page ----------------------------------------------------------------------------------
header()
st.sidebar.markdown('<div class="side-logo"><div class="art"></div><b>Judge the<br>model</b></div>', unsafe_allow_html=True)
section = st.sidebar.radio("Section", SECTIONS, key="nav")
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get, key="dataset")
res = load(key)
model = st.sidebar.selectbox("Model", m11.MODELS, key="model",
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
sidebar_help()
recommendations(key)
st.markdown(f'<div class="sec"><h2>{section}</h2><div class="sec-icon i{SECTIONS.index(section)}"></div></div>', unsafe_allow_html=True)
st.caption(f"{m11.DATASET_LABEL[key]} · {res['train']['n']:,} training and {res['test']['n']:,} test patients")
if model in res["models"] or section == SECTIONS[2]:
    RENDER[section](res, model, key)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
