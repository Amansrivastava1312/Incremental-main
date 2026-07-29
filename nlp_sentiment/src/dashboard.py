
import pandas as pd
import plotly.express as px
import streamlit as st

PRED_FILE = "artifact/predictions.csv"
ASPECT_FILE = "artifact/aspect_sentiment.csv"

st.set_page_config(page_title="Market Sentiment Insight", layout="wide")
st.title("📊 Market Sentiment Insight Module")


@st.cache_data
def load_data():
    preds = pd.read_csv(PRED_FILE)
    try:
        aspects = pd.read_csv(ASPECT_FILE)
    except FileNotFoundError:
        aspects = None
    return preds, aspects


preds, aspects = load_data()
label_col = "true_label" if "true_label" in preds.columns else "bert_pred"

# ---------- Task 1: sentiment distribution ----------
st.header("1. Sentiment distribution")
c1, c2 = st.columns(2)

dist = preds[label_col].value_counts().reset_index()
dist.columns = ["sentiment", "count"]

with c1:
    fig_pie = px.pie(dist, names="sentiment", values="count",
                     title="Overall sentiment share", hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    fig_bar = px.bar(dist, x="sentiment", y="count", color="sentiment",
                     title="Review count by sentiment", text="count")
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------- Task 2 & 3: aspect breakdown ----------
st.header("2. Aspect-wise sentiment")
if aspects is not None:
    sentiment_cols = [c for c in aspects.columns if c not in ("aspect", "total")]
    melted = aspects.melt(id_vars="aspect", value_vars=sentiment_cols,
                          var_name="sentiment", value_name="count")
    fig_aspect = px.bar(melted, x="aspect", y="count", color="sentiment",
                        barmode="group", title="Aspect keyword counts by sentiment")
    st.plotly_chart(fig_aspect, use_container_width=True)
    st.dataframe(aspects, use_container_width=True)
else:
    st.info("Run `python3 src/aspects.py` first to generate aspect_sentiment.csv")

# ---------- Model comparison ----------
if {"logreg_correct", "bert_correct"}.issubset(preds.columns):
    st.header("3. Model comparison")
    acc = pd.DataFrame({
        "model": ["Logistic Regression", "BERT"],
        "accuracy": [preds["logreg_correct"].mean(), preds["bert_correct"].mean()],
    })
    fig_acc = px.bar(acc, x="model", y="accuracy", color="model",
                     text=acc["accuracy"].round(3), title="Accuracy comparison")
    fig_acc.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_acc, use_container_width=True)

# ---------- Browsable reviews ----------
st.header("4. Review explorer")
choice = st.selectbox("Filter by sentiment", ["all"] + sorted(preds[label_col].unique()))
view = preds if choice == "all" else preds[preds[label_col] == choice]
st.dataframe(view, use_container_width=True)
