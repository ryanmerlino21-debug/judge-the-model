"""Judge the Model: a Streamlit app for a clinical lead deciding whether a classifier is fit for a job.

Run locally:  bash run_app.sh  (it runs streamlit run streamlit_app.py)
Every number comes from m11.py applied to the files in results/; the app never retrains a model.
"""

import altair as alt
import pandas as pd
import streamlit as st

import m11

TRAIN = "Training"
TEST = "Test"
SECTIONS = ["1. Beyond Accuracy", "2. A Threshold for the Job", "3. Training vs. Test"]

POSITIVE = {
    "readmission": "readmitted to hospital within 30 days of discharge",
    "diabetes": "has a doctor diagnosis of diabetes, or an HbA1c of 6.5% or more",
}
GROUP = {"readmission": "patients", "diabetes": "adults"}
SENS_TARGET = {"diabetes": 0.90}  # the screening team's target; readmission has no fixed target

st.set_page_config(page_title="Judge the Model", page_icon="🩺", layout="wide")

LIME, TEAL, AMBER, CORAL, GREY = "#B6F23A", "#4FD1C5", "#F6C453", "#FF6B5E", "#8A9A90"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Fraunces', Georgia, serif !important; letter-spacing: -0.01em; }
.stApp { background: radial-gradient(1200px 500px at 12% -10%, rgba(182,242,58,0.10), transparent 60%), #0A1712; }
.block-container { padding-top: 2rem; max-width: 1200px; }
[data-testid="stSidebar"] { background: #0E2018; border-right: 1px solid rgba(182,242,58,0.12); }
.hero { background: linear-gradient(135deg, #12342A 0%, #0E2018 55%, #1B3A12 100%); border: 1px solid rgba(182,242,58,0.22);
        border-radius: 24px; padding: 2.2rem 2.4rem; margin-bottom: 1.2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.35); }
.hero-title { font-size: 3rem; font-weight: 700; line-height: 1.05; margin: 0; color: #F4FAF0; }
.hero-title span { color: #B6F23A; }
.hero-sub { color: #B9CDBF; font-size: 1.1rem; margin: 0.7rem 0 1.1rem; max-width: 46rem; }
.chip { display: inline-block; padding: 0.25rem 0.8rem; margin: 0 0.4rem 0.3rem 0; border-radius: 999px; font-size: 0.82rem; font-weight: 500;
        background: rgba(182,242,58,0.12); color: #D8F58F; border: 1px solid rgba(182,242,58,0.3); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; border-color: rgba(182,242,58,0.2) !important; background: rgba(18,38,29,0.7);
        box-shadow: 0 6px 24px rgba(0,0,0,0.25); }
[data-testid="stMetric"] { background: #12261D; border: 1px solid rgba(182,242,58,0.18); border-radius: 18px; padding: 1rem 1.2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25); transition: transform .15s ease, border-color .15s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: rgba(182,242,58,0.55); }
[data-testid="stMetricValue"] { font-family: 'Fraunces', serif; color: #B6F23A; }
[data-testid="stExpander"] { border-radius: 16px; border: 1px solid rgba(182,242,58,0.18); background: rgba(18,38,29,0.55); }
[data-testid="stTable"] table { border-collapse: separate; border-spacing: 0; border-radius: 14px; overflow: hidden; border: 1px solid rgba(182,242,58,0.18); width: 100%; }
[data-testid="stTable"] th { background: #17352A; color: #D8F58F; font-weight: 600; }
[data-testid="stTable"] td, [data-testid="stTable"] th { border-color: rgba(255,255,255,0.06) !important; padding: 0.55rem 0.8rem; }
[data-testid="stTable"] tr:hover td { background: rgba(182,242,58,0.05); }
[data-testid="stAlert"] { border-radius: 16px; }
.stButton > button { border-radius: 999px; background: #B6F23A; color: #0A1712; font-weight: 600; border: none; padding: 0.5rem 1.2rem; transition: transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(182,242,58,0.35); color: #0A1712; }
h2 { border-left: 4px solid #B6F23A; padding-left: 0.7rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style_chart(ch):
    return (ch.configure(background="transparent").configure_view(strokeWidth=0)
            .configure_axis(labelColor="#B9CDBF", titleColor="#B9CDBF", gridColor="#1F3A2E", domainColor="#2B4A3B")
            .configure_legend(labelColor="#EAF2EC", titleColor="#B9CDBF", orient="top"))


@st.cache_data
def load(name: str):
    return m11.load(name)


def pct(v, d: int = 1) -> str:
    return m11.fmt_pct(v, d)


def per100(v) -> str:
    return m11.fmt_num(v, 1)


def matrix(c: dict, who: str, short: bool = False):
    """Course layout: rows = what actually happened, columns = what the model said, positives first.
    Correct cells (TP, TN) are green and mistakes (FN, FP) are red."""
    df = pd.DataFrame(
        [[f"TP = {c['tp']:,}", f"FN = {c['fn']:,}"], [f"FP = {c['fp']:,}", f"TN = {c['tn']:,}"]],
        index=["Actual +", "Actual −"] if short else ["Actual: positive", "Actual: negative"],
        columns=["Pred. +", "Pred. −"] if short else ["Predicted: positive (flagged)", "Predicted: negative (not flagged)"],
    )
    good, bad = "background-color: rgba(182,242,58,0.28)", "background-color: rgba(255,107,94,0.30)"
    return df.style.map(lambda v: good if v[:2] in ("TP", "TN") else bad)


def how_to_read(text: str) -> None:
    with st.expander("How to read this page", expanded=False):
        st.markdown(text)


def glossary() -> None:
    with st.expander("Plain-language glossary (tap to open)"):
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


# ---- Recommendations -----------------------------------------------------------
def recommendations() -> None:
    d = load("decisions")
    recs = (d or {}).get("recommendations", {})
    with st.container(border=True):
        st.subheader("Recommendations")
        if not any(recs.values()):
            st.caption("No recommendations yet: fill in `Recommendations` in DECISIONS.md, run "
                       "`python -m analysis`, then commit and push.")
        else:
            for k in m11.DATASETS:
                st.markdown(f"**{m11.DATASET_LABEL[k]}:** {recs.get(k) or '_not written yet_'}")


# ---- 1. Beyond Accuracy ----------------------------------------------------------
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

    how_to_read("1. Pick a **dataset** and a **model** in the sidebar.\n"
                "2. Read the **positive class**: the outcome we are trying to find.\n"
                "3. Compare the model's accuracy with the 'most common class for everyone' accuracy. If they are close, accuracy is not telling you the model helps.\n"
                "4. Read the confusion matrix as four groups of patients (green = right, red = mistakes).\n"
                "5. Scroll to the table that compares all five models at the same 0.50 threshold.")
    st.markdown(f"**Positive class:** a patient counts as positive if they were **{POSITIVE[key]}**. "
                f"All numbers on this page are **{TEST}** results: {m['n']:,} {who} the model never saw while learning, at threshold 0.50.")

    st.subheader("Why accuracy alone can mislead")
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

    st.subheader(f"{m11.MODEL_LABEL[model]}: confusion matrix at 0.50 ({TEST})")
    st.caption(f"{TEST} · rows = what actually happened, columns = what the model said. Positive means {POSITIVE[key]}.")
    st.table(matrix(c, who))
    st.success(f"**In plain words:** of the {m['positives']:,} {who} who had the outcome, {m11.MODEL_LABEL[model]} flagged {c['tp']:,} and missed {c['fn']:,}. "
               f"Of the {m['negatives']:,} who did not, it left {c['tn']:,} alone and flagged {c['fp']:,} by mistake ({TEST}, threshold 0.50).")
    st.markdown(
        f"- **TP (true positive):** {c['tp']:,} {who} flagged who had the outcome.\n"
        f"- **FN (false negative):** {c['fn']:,} {who} not flagged who had the outcome (missed).\n"
        f"- **FP (false positive):** {c['fp']:,} {who} flagged who did not have the outcome (false alarms).\n"
        f"- **TN (true negative):** {c['tn']:,} {who} not flagged who did not have the outcome.")

    st.subheader(f"Sensitivity, specificity and precision ({TEST}, 0.50)")
    st.table(pd.DataFrame([{"Metric": n, "Value": v, "Of the patients who…": w} for n, v, w in metric_rows(c)]).set_index("Metric"))
    if m["flagged"] == 0:
        st.caption("This model flagged no one at 0.50, so precision is 0 ÷ 0, which is undefined (not 0).")

    st.subheader(f"All five models at 0.50 ({TEST})")
    st.caption(f"{TEST} · same {m['n']:,} {who} for every model. Undefined means a 0 ÷ 0 (for example, precision when no one is flagged).")
    rows, cmats = [], {}
    for name in m11.MODELS:
        if name not in res["models"]:
            rows.append({"Model": m11.MODEL_LABEL[name] + " (not trained yet)"})
            continue
        pp, yy = m11.preds(res, name)
        cc = m11.confusion(pp, yy, 0.5)
        mm = m11.metrics(cc)
        cmats[name] = cc
        rows.append({
            "Model": m11.MODEL_LABEL[name], "Accuracy": pct(mm["accuracy"]), "Sensitivity (recall, TPR)": pct(mm["sensitivity"]),
            "Specificity": pct(mm["specificity"]), "FPR (1 − specificity)": pct(mm["fpr"]),
            "Precision": pct(mm["precision"]), "F1": m11.fmt_num(mm["f1"]),
        })
    st.table(pd.DataFrame(rows).set_index("Model").fillna("–"))
    st.caption("Accuracy ranks differently from F1 and sensitivity. Read the whole row, not one column.")
    cols = st.columns(len(m11.MODELS))
    for col, name in zip(cols, m11.MODELS):
        with col:
            st.markdown(f"**{m11.MODEL_LABEL[name]}**")
            if name in cmats:
                st.caption(f"{TEST} · 0.50 · rows actual, columns predicted")
                st.table(matrix(cmats[name], who, short=True))
            else:
                st.caption("No results yet.")


# ---- 2. A Threshold for the Job ----------------------------------------------------
def delta(new, old, as_pct: bool = True) -> str:
    if new is None or old is None:
        return "undefined"
    d = (new - old) * (100 if as_pct else 1)
    return f"{d:+.1f}" + (" pts" if as_pct else "")


def threshold(res: dict, model: str, key: str) -> None:
    p, y = m11.preds(res, model)
    who = GROUP[key]
    st.markdown(f"All numbers on this page are **{TEST}** results ({len(y):,} {who} the model never saw while learning). "
                "The **threshold** is the probability above which we flag a patient. It is a decision made separately from the model: "
                "moving it changes who is flagged, never the probabilities or the ranking.")
    how_to_read("1. Choose a **dataset** and **model** in the sidebar.\n"
                "2. Move the **threshold slider**. A lower threshold flags more patients: it finds more positives but raises false alarms.\n"
                "3. Compare the **0.50** column with the **slider** column. Flagged per 100 shows the workload.\n"
                "4. The line chart shows every threshold at once. The ranking (AUROC) never changes.")
    if key == "diabetes":
        st.info("**The diabetes job:** finding positives matters most. The screening team wants a sensitivity of at least 0.90.")
    else:
        st.info("**The readmission job:** there is no fixed target. See what each threshold gains and gives up, then choose.")

    skey = f"thr_{key}_{model}"
    if key in SENS_TARGET:
        target = SENS_TARGET[key]
        grid = [i / 100 for i in range(1, 100)]
        ok = [t for t in grid if (m11.metrics(m11.confusion(p, y, t))["sensitivity"] or 0) >= target]
        if ok:
            best = max(ok)
            if st.button(f"Set the threshold to {best:.2f}, the highest in steps of 0.01 that reaches sensitivity {target:.2f}"):
                st.session_state[skey] = best
        else:
            st.caption(f"No threshold from 0.01 to 0.99 reaches a sensitivity of {target:.2f} for this model.")
    st.session_state.setdefault(skey, 0.50)
    thr = st.slider("Threshold (flag a patient when their probability is at least this)", 0.01, 0.99, step=0.01, key=skey)

    c0, c1 = m11.confusion(p, y, 0.5), m11.confusion(p, y, thr)
    m0, m1 = m11.metrics(c0), m11.metrics(c1)

    st.subheader(f"Side by side ({TEST}): 0.50 vs. {thr:.2f}")
    l, r = st.columns(2)
    for col, c, t, label in ((l, c0, 0.5, "0.50 (reference)"), (r, c1, thr, f"{thr:.2f} (slider)")):
        with col:
            st.markdown(f"**Threshold {label}**")
            st.caption(f"{TEST} · {m11.MODEL_LABEL[model]} · rows actual, columns predicted")
            st.table(matrix(c, who))

    REF, NEW = "Threshold 0.50 (reference)", f"Threshold {thr:.2f} (slider)"

    def row(name, k, as_pct=True):
        f = pct if as_pct else per100
        return {"Metric": name, REF: f(m0[k]), NEW: f(m1[k]), "Change": delta(m1[k], m0[k], as_pct)}

    cmp = [row("Flagged per 100 patients", "flagged_per_100", False), row("Sensitivity (recall, TPR)", "sensitivity"),
           row("Specificity", "specificity"), row("FPR (1 − specificity)", "fpr"), row("Precision", "precision"),
           row("Accuracy", "accuracy"), row("F1", "f1")]
    cmp[0]["Change"] = delta(m1["flagged_per_100"], m0["flagged_per_100"], False)
    cmp += [{"Metric": "Positives found (TP)", REF: f"{c0['tp']:,}", NEW: f"{c1['tp']:,}", "Change": f"{c1['tp'] - c0['tp']:+,}"},
            {"Metric": "False alarms (FP)", REF: f"{c0['fp']:,}", NEW: f"{c1['fp']:,}", "Change": f"{c1['fp'] - c0['fp']:+,}"}]
    st.caption(f"{TEST} · Flagged per 100 = of every 100 {who}, how many the model flags (changes in percentage points, pts).")
    st.table(pd.DataFrame(cmp).set_index("Metric"))

    nflag0, nflag1 = m0["flagged_per_100"], m1["flagged_per_100"]
    if m0["n"] and nflag0 is not None and nflag1 is not None:
        extra_fp = 100 * (c1["fp"] - c0["fp"]) / m1["n"]
        extra_tp = 100 * (c1["tp"] - c0["tp"]) / m1["n"]
        st.markdown(f"Compared with 0.50, per 100 {who} screened: **{nflag1 - nflag0:+.1f} flagged**, "
                    f"**{extra_tp:+.1f} more positives found**, **{extra_fp:+.1f} more false alarms** ({TEST}).")
    if key in SENS_TARGET:
        s = m1["sensitivity"]
        met = s is not None and s >= SENS_TARGET[key]
        st.markdown(f"Sensitivity target {SENS_TARGET[key]:.2f}: {'**met**' if met else '**not met**'} at {thr:.2f} (sensitivity {pct(s)}, {TEST}).")
        st.caption("This threshold was picked by looking at these test patients, so on new patients sensitivity will usually be slightly lower (peeking).")

    st.subheader(f"Every threshold at once ({TEST})")
    st.caption(f"{TEST} · {m11.MODEL_LABEL[model]} · each line is computed at thresholds 0.01 to 0.99. Gaps in the precision line are thresholds where no one is flagged (precision undefined).")
    curve = []
    for t in range(1, 100):
        mt = m11.metrics(m11.confusion(p, y, t / 100))
        curve.append({"Threshold": t / 100,
                      "Sensitivity (%)": None if mt["sensitivity"] is None else 100 * mt["sensitivity"],
                      "Precision (%)": None if mt["precision"] is None else 100 * mt["precision"],
                      "Flagged per 100": mt["flagged_per_100"]})
    long = pd.DataFrame(curve).melt("Threshold", var_name="Measure", value_name="Value").dropna()
    line = alt.Chart(long).mark_line().encode(x=alt.X("Threshold:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Value:Q", title="Percent / per 100"),
                                              color=alt.Color("Measure:N", scale=alt.Scale(domain=["Sensitivity (%)", "Precision (%)", "Flagged per 100"], range=[LIME, TEAL, AMBER])), tooltip=["Threshold", "Measure", alt.Tooltip("Value:Q", format=".1f")])
    mark = alt.Chart(pd.DataFrame({"Threshold": [thr]})).mark_rule(strokeDash=[4, 4], color="#FFFFFF").encode(x="Threshold:Q")
    st.altair_chart(style_chart(line + mark), width="stretch")
    st.caption("The dashed line is your slider's threshold.")

    st.subheader("What the slider does not change")
    auc = m11.auroc(p, y)
    st.markdown(f"{TEST} AUROC for {m11.MODEL_LABEL[model]}: **{m11.fmt_num(auc)}**. It is the same at every threshold. "
                "AUROC describes ranking (how often a patient with the outcome gets a higher probability than one without); "
                "it is not accuracy, and it does not choose a threshold.")


# ---- 3. Training vs. Test --------------------------------------------------------
def train_vs_test(res: dict, model: str, key: str) -> None:
    s = m11.auroc_summary(res)
    who = GROUP[key]
    n_tr, n_te = res["train"]["n"], res["test"]["n"]
    st.markdown("A model is judged on patients it never saw. AUROC describes **ranking**: how often a patient with the outcome gets a higher "
                "probability than one without (0.50 is no better than chance, 1.00 is perfect). It is not accuracy.")
    how_to_read("1. Choose a **dataset** in the sidebar.\n"
                "2. In the table, compare each model's **Training** AUROC (patients it learned from) with its **Test** AUROC (patients it never saw).\n"
                "3. A big **gap** means the model memorised its training patients. Trust the **Test** column.\n"
                "4. The chart shows the same thing: a grey dot for Training, a green dot for Test.")
    a, b = st.columns(2)
    a.metric(f"{TRAIN} patients", f"{n_tr:,}", help=f"{res['train']['n_pos']:,} positive. Each model learned from these {who}.")
    b.metric(f"{TEST} patients", f"{n_te:,}", help=f"{res['test']['n_pos']:,} positive. Each model never saw these {who} while learning.")
    st.caption(f"{TRAIN} AUROC rests on {n_tr:,} {who} ({res['train']['n_pos']:,} positive); {TEST} AUROC rests on {n_te:,} {who} ({res['test']['n_pos']:,} positive). "
               "With fewer patients, an AUROC can move by chance.")

    order = sorted(s, key=lambda m: (s[m]["test"] is None, -(s[m]["test"] or 0)))
    rows = [{"Model": m11.MODEL_LABEL[m] + (" ◀ selected" if m == model else ""), f"{TRAIN} AUROC": m11.fmt_num(s[m]["train"]),
             f"{TEST} AUROC": m11.fmt_num(s[m]["test"]), "Gap (Training − Test)": m11.fmt_num(s[m]["gap"])} for m in order]
    st.subheader("Training and test AUROC side by side")
    st.caption(f"{TRAIN} and {TEST} · sorted by {TEST} AUROC, because that is the one to expect on new patients. "
               f"{TRAIN} AUROC flatters every model: it is measured on the patients the model learned from.")
    st.table(pd.DataFrame(rows).set_index("Model"))

    st.subheader("Training vs. test, model by model")
    st.caption(f"{TRAIN} (grey) and {TEST} (green) AUROC for each model. The longer the line between the dots, the bigger the gap.")
    pts = pd.DataFrame([{"Model": m11.MODEL_LABEL[m], "Partition": part, "AUROC": s[m][k], "Training": s[m]["train"], "Test": s[m]["test"]}
                        for m in order if s[m]["gap"] is not None for part, k in ((TRAIN, "train"), (TEST, "test"))])
    if len(pts):
        names = [m11.MODEL_LABEL[m] for m in order]
        base = alt.Chart(pts).encode(y=alt.Y("Model:N", sort=names, title=None, axis=alt.Axis(labelLimit=260)))
        rule = base.mark_rule(color="#5F7A6B", strokeWidth=3).encode(x=alt.X("Training:Q", scale=alt.Scale(domain=[0.4, 1.0]), title="AUROC"), x2="Test:Q")
        dots = base.mark_circle(size=140, opacity=1).encode(
            x="AUROC:Q", color=alt.Color("Partition:N", scale=alt.Scale(domain=[TRAIN, TEST], range=[GREY, LIME])),
            tooltip=["Model", "Partition", alt.Tooltip("AUROC:Q", format=".3f")])
        st.altair_chart(style_chart(rule + dots), width="stretch")

    if model in s and s[model]["gap"] is not None:
        v = s[model]
        st.markdown(f"**{m11.MODEL_LABEL[model]}:** {TRAIN} AUROC {m11.fmt_num(v['train'])}, {TEST} AUROC {m11.fmt_num(v['test'])}, gap {m11.fmt_num(v['gap'])}. "
                    + ("On patients it had already seen it ranked far better than on new patients, a sign it memorised them. "
                       if v["gap"] >= 0.1 else "The gap is small, so it ranked training and new patients about equally well. ")
                    + f"Expect the {TEST} AUROC on new patients, not the {TRAIN} one.")
    st.caption("A test AUROC a little above the training AUROC can happen by chance when the test set is small.")


RENDER = dict(zip(SECTIONS, [beyond_accuracy, threshold, train_vs_test]))

st.markdown(
    f"""<div class="hero"><p class="hero-title">Judge the <span>Model</span></p>
    <p class="hero-sub">Which classifier would you trust for this job, at what threshold, and with what limits?</p>
    <span class="chip">{len(m11.DATASETS)} clinical jobs</span><span class="chip">{len(m11.MODELS)} models, one baseline</span>
    <span class="chip">Every table labelled Training or Test</span></div>""", unsafe_allow_html=True)
recommendations()
glossary()
section = st.sidebar.radio("Section", SECTIONS)
key = st.sidebar.selectbox("Dataset", m11.DATASETS, format_func=m11.DATASET_LABEL.get)
res = load(key)
model = st.sidebar.selectbox("Model", m11.MODELS,
                             format_func=lambda m: m11.MODEL_LABEL[m] + ("" if m in res["models"] else " (not trained yet)"))
st.header(section)
st.caption(f"{m11.DATASET_LABEL[key]}: {res['train']['n']:,} training rows ({res['train']['n_pos']:,} positive), "
           f"{res['test']['n']:,} test rows ({res['test']['n_pos']:,} positive).")
if model in res["models"] or section == SECTIONS[2]:
    RENDER[section](res, model, key)
else:
    st.warning(f"{m11.MODEL_LABEL[model]} has no results yet: run `python -m analysis`, then commit and push `results/`.")
