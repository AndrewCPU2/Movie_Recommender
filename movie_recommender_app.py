# movie_recommender_app.py
import json
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- DEFINE FILE PATHS ----
BASE_DIR         = Path(__file__).parent.resolve()
IMDB_CSV         = BASE_DIR / "imdb_top_1000.csv"
USER_FB_FILE     = BASE_DIR / "user_feedback.json"
CD_FB_FILE       = BASE_DIR / "cooldown_feedback.json"
NOT_WATCHED_FILE = BASE_DIR / "not_watched.json"
RECS_FILE        = BASE_DIR / "recommendations.json"

# ---- I/O HELPERS ----

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Poster_URL"})
    df["Genre"]       = df["Genre"].fillna("")
    df["IMDB_Rating"] = df["IMDB_Rating"].fillna(df["IMDB_Rating"].mean())
    df["Meta_score"]  = df["Meta_score"].fillna(df["Meta_score"].mean())
    return df

def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=4), encoding="utf-8")

def save_recs(df: pd.DataFrame) -> None:
    text = df.to_json(orient="records", indent=2)
    RECS_FILE.write_text(text, encoding="utf-8")

def load_recs() -> pd.DataFrame:
    if RECS_FILE.exists():
        records = json.loads(RECS_FILE.read_text(encoding="utf-8"))
        return pd.DataFrame.from_records(records)
    return pd.DataFrame()

@st.cache_resource
def build_vectorizer_and_sim(df: pd.DataFrame):
    vect = TfidfVectorizer(stop_words="english")
    mat  = vect.fit_transform(df["Genre"])
    sim  = cosine_similarity(mat)
    return vect, sim

# ---- LOAD DATA & STATE ----

imdb_df     = load_csv(IMDB_CSV)
user_fb     = load_json(USER_FB_FILE)
cd_fb       = load_json(CD_FB_FILE)
not_watched = load_json(NOT_WATCHED_FILE)
vect, sim   = build_vectorizer_and_sim(imdb_df)

# load last‐saved recs into session_state (if any)
if "recs" not in st.session_state:
    saved = load_recs()
    if not saved.empty:
        st.session_state.recs     = saved
        st.session_state.feedback = {
            row["Series_Title"]: 0
            for _, row in saved.iterrows()
        }

if "search_count" not in st.session_state:
    st.session_state.search_count = len(user_fb)

# ---- RECOMMENDER FUNCTION ----

def hybrid_recommendation(df, sim, genre, year, director):
    d = df.copy()
    if genre != "Any Genre":
        d = d[d["Genre"].str.contains(genre, case=False, na=False)]
    if year:
        d = d[d["Released_Year"].astype(str) == year]
    if director not in ("", "Any Director"):
        d = d[d["Director"].str.contains(director, case=False, na=False)]

    current = st.session_state.search_count
    d = d[~d["Series_Title"].isin(not_watched.keys())]
    d = d[~d["Series_Title"].apply(
        lambda t: (cd_fb.get(t) or 0) > current
    )]

    if d.empty:
        return pd.DataFrame()

    d["Weighted_Score"] = d["IMDB_Rating"] * 0.7 + (d["Meta_score"] / 10) * 0.3

    def adjust(r):
        fb = user_fb.get(r["Series_Title"])
        base = r["Weighted_Score"]
        if fb:
            base += (fb[0] - 5) * 0.1
        return base

    d["Weighted_Score"] = d.apply(adjust, axis=1)

    idxs    = d.index.tolist()
    avg_sim = sim[idxs].mean(axis=0)[idxs]
    d["Similarity_Score"] = avg_sim

    return (
        d.sort_values(["Weighted_Score", "Similarity_Score"], ascending=False)
         .head(3)[
             ["Poster_URL", "Series_Title", "Released_Year", "IMDB_Rating", "Weighted_Score"]
         ]
    )

# ---- UI ----

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# sidebar: show saved recs & filters
with st.sidebar:
    if st.checkbox("Show saved recs"):
        saved = load_recs()
        if not saved.empty:
            st.write("### Previously saved recommendations")
            st.dataframe(saved)
        else:
            st.warning("No recommendations.json found.")

    st.markdown("---")
    with st.expander("🔍 Settings & Filters", expanded=True):
        all_genres = sorted({
            g.strip().capitalize()
            for row in imdb_df["Genre"]
            for g in row.split(",") if g
        })
        directors = sorted(imdb_df["Director"].dropna().unique())

        genre_sel    = st.selectbox("Genre", ["Any Genre"] + all_genres)
        year_sel     = st.text_input("Year (leave blank for any)", "")
        director_sel = st.selectbox("Director", ["Any Director"] + directors)

        if st.button("Get Recommendations"):
            recs = hybrid_recommendation(
                imdb_df, sim,
                genre_sel,
                year_sel.strip(),
                director_sel
            )
            if recs.empty:
                st.warning("No matches—try different filters.")
            else:
                st.session_state.recs     = recs
                st.session_state.feedback = {
                    t: 0 for t in recs.Series_Title
                }
                save_recs(recs)
                st.session_state.pop("show_prompt", None)

        if st.button("Start Over"):
            for k in ("recs","feedback","show_prompt"):
                st.session_state.pop(k, None)
            st.experimental_rerun()

# main area: feedback prompt
if st.session_state.get("show_prompt"):
    st.markdown(
        """<style>[data-testid="stAppViewContainer"]{filter:brightness(30%);}</style>""",
        unsafe_allow_html=True,
    )
    st.write("## Search again?")
    c1, c2 = st.columns(2)
    if c1.button("🔍 New Search"):
        for k in ("recs","feedback","show_prompt"):
            st.session_state.pop(k, None)
        st.experimental_rerun()
    if c2.button("⏹️ Exit"):
        st.write("Enjoy your movies! 🍿")
    st.stop()

if "recs" in st.session_state:
    recs = st.session_state.recs
    st.subheader("Top 3 Recommendations")
    cols = st.columns(len(recs))
    labels = [
        "0 = Not seen yet","1 = Bad","2 = Poor","3 = Fair","4 = Okay",
        "5 = Average","6 = Good","7 = Very Good","8 = Great",
        "9 = Excellent","10 = Masterpiece",
    ]

    with st.form("feedback_form"):
        for i, (_, row) in enumerate(recs.iterrows()):
            title = row.Series_Title
            col   = cols[i]
            with col:
                st.image(row.Poster_URL, caption=title, width=200)
                choice = st.selectbox(
                    "Your rating:", labels,
                    index=st.session_state.feedback.get(title, 0),
                    key=f"rating_{i}"
                )
                st.session_state.feedback[title] = int(choice.split(" = ")[0])

        if st.form_submit_button("Submit Feedback"):
            cnt = st.session_state.search_count
            for title, score in st.session_state.feedback.items():
                if score == 0:
                    not_watched[title] = cnt
                else:
                    user_fb[title]   = (score, cnt + 1)
                    cd_fb[title]     = cnt + (20 if score >= 7 else 5)

            # persist all feedback JSONs
            save_json(user_fb, USER_FB_FILE)
            save_json(cd_fb,   CD_FB_FILE)
            save_json(not_watched, NOT_WATCHED_FILE)

            st.session_state.search_count = cnt + 1
            st.success("Thanks for your feedback! 🎉")
            st.session_state.show_prompt = True
