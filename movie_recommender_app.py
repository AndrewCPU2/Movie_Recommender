# … all your imports and helper definitions above …

# ---- LOAD DATA & FEEDBACK ----
imdb_df     = load_csv("imdb_top_1000.csv")
user_fb     = load_json("user_feedback.json")
cd_fb       = load_json("cooldown_feedback.json")
not_watched = load_json("not_watched.json")
vect, sim   = build_vectorizer_and_sim(imdb_df)

# load previous recs into session_state…
if "recs" not in st.session_state:
    saved = load_recs()
    if not saved.empty:
        st.session_state.recs     = saved
        st.session_state.feedback = {r["Series_Title"]: 0 for _, r in saved.iterrows()}

if "search_count" not in st.session_state:
    st.session_state.search_count = len(user_fb)

# ---- UI SETUP ----
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# ---- SIDEBAR ----
with st.sidebar:
    # ← **Add your “show saved recs” checkbox here:**
    if st.checkbox("Show saved recs"):
        if os.path.exists("recommendations.json"):
            saved = load_recs("recommendations.json")
            st.write("### Previously saved recommendations")
            st.dataframe(saved)
        else:
            st.warning("No saved recommendations.json found.")

    st.markdown("---")  # optional separator

    with st.expander("🔍 Settings & Filters", expanded=True):
        genre_sel    = st.selectbox("Genre", ["Any Genre"] + all_genres)
        year_sel     = st.text_input("Year (leave blank for any)", value="")
        director_sel = st.selectbox("Director", ["Any Director"] + directors)

        if st.button("Get Recommendations"):
            # … your recommendation logic …

        if st.button("Start Over"):
            # … your reset logic …



            
import os
import json
import streamlit as st
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---- I/O HELPERS ----

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Poster_URL"})
    df['Genre']       = df['Genre'].fillna("")
    df['IMDB_Rating'] = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean())
    df['Meta_score']  = df['Meta_score'].fillna(df['Meta_score'].mean())
    return df

def load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_recs(df: pd.DataFrame, path: str = "recommendations.json") -> None:
    records = df.to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

def load_recs(path: str = "recommendations.json") -> pd.DataFrame:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if records:
            return pd.DataFrame.from_records(records)
    return pd.DataFrame()

@st.cache_resource
def build_vectorizer_and_sim(df: pd.DataFrame) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    vect = TfidfVectorizer(stop_words="english")
    mat  = vect.fit_transform(df["Genre"])
    sim  = cosine_similarity(mat)
    return vect, sim

# ---- LOAD DATA & FEEDBACK ----

imdb_df     = load_csv("imdb_top_1000.csv")
user_fb     = load_json("user_feedback.json")
cd_fb       = load_json("cooldown_feedback.json")
not_watched = load_json("not_watched.json")
vect, sim   = build_vectorizer_and_sim(imdb_df)

# load last‐saved recs into session_state (if any)
if "recs" not in st.session_state:
    saved = load_recs()
    if not saved.empty:
        st.session_state.recs     = saved
        # initialize feedback trackers for loaded recs
        st.session_state.feedback = {
            row["Series_Title"]: 0 for _, row in saved.iterrows()
        }

# initialize search count
if "search_count" not in st.session_state:
    st.session_state.search_count = len(user_fb)

# ---- RECOMMENDER FUNCTION ----

def hybrid_recommendation(
    df: pd.DataFrame,
    sim: pd.DataFrame,
    genre: str,
    year: str,
    director: str
) -> pd.DataFrame:
    d = df.copy()
    if genre != "Any Genre":
        d = d[d["Genre"].str.contains(genre, case=False, na=False)]
    if year:
        d = d[d["Released_Year"].astype(str) == year]
    if director not in ("", "Any Director"):
        d = d[d["Director"].str.contains(director, case=False, na=False)]
    # exclude skipped and cooldown
    current = st.session_state.search_count
    d = d[~d["Series_Title"].isin(not_watched.keys())]
    def in_cd(t): 
        exp = cd_fb.get(t)
        return exp is not None and exp > current
    d = d[~d["Series_Title"].apply(in_cd)]
    if d.empty:
        return pd.DataFrame()
    # base weighted score
    d = d.assign(
        Weighted_Score = d["IMDB_Rating"] * 0.7 + (d["Meta_score"]/10) * 0.3
    )
    # adjust by past feedback
    def adjust(r):
        title = r["Series_Title"]
        fb = user_fb.get(title)
        base = r["Weighted_Score"]
        if fb:
            rating, _ = fb
            base += (rating - 5)*0.1
        return base
    d["Weighted_Score"] = d.apply(adjust, axis=1)
    # similarity
    idxs   = d.index.tolist()
    avg_sim = sim[idxs].mean(axis=0)[idxs]
    d["Similarity_Score"] = avg_sim
    # top 3
    return (
        d.sort_values(["Weighted_Score","Similarity_Score"], ascending=False)
         .head(3)[["Poster_URL","Series_Title","Released_Year","IMDB_Rating","Weighted_Score"]]
    )

# ---- UI ----

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# build filters
all_genres = sorted({g.strip().capitalize() 
    for row in imdb_df["Genre"] for g in row.split(",") if g})
directors = sorted(imdb_df.get("Director", pd.Series()).dropna().unique())

# — Sidebar
with st.sidebar.expander("🔍 Settings & Filters", expanded=True):
    genre_sel    = st.selectbox("Genre", ["Any Genre"] + all_genres)
    year_sel     = st.text_input("Year (leave blank for any)", value="")
    director_sel = st.selectbox("Director", ["Any Director"] + directors)

    if st.button("Get Recommendations"):
        recs = hybrid_recommendation(
            imdb_df, sim, genre_sel, year_sel.strip(), director_sel
        )
        if recs.empty:
            st.warning("No matches—try different filters.")
        else:
            st.session_state.recs     = recs
            st.session_state.feedback = { t: 0 for t in recs.Series_Title }
            save_recs(recs)                     # ← persist to disk
            st.session_state.pop("show_prompt", None)

    if st.button("Start Over"):
        for k in ("recs","feedback","show_prompt"):
            st.session_state.pop(k, None)
        st.experimental_rerun()

# — Main  
if st.session_state.get("show_prompt"):
    st.markdown(
        """<style>[data-testid="stAppViewContainer"]{filter:brightness(30%);}</style>""",
        unsafe_allow_html=True,
    )
    st.write("## Would you like to search again?")
    c1, c2 = st.columns(2)
    if c1.button("🔍 New Search"):
        for k in ("recs","feedback","show_prompt"): st.session_state.pop(k, None)
        st.experimental_rerun()
    if c2.button("⏹️ Exit"):
        st.write("Enjoy your movies! 🍿")
    st.stop()

if "recs" in st.session_state:
    recs = st.session_state.recs
    st.subheader("Top 3 Recommendations")
    cols = st.columns(len(recs))
    rating_labels = [
        "0 = Not seen yet","1 = Bad","2 = Poor","3 = Fair","4 = Okay",
        "5 = Average","6 = Good","7 = Very Good","8 = Great",
        "9 = Excellent","10 = Masterpiece",
    ]
    with st.form("feedback_form"):
        for i, (_, row) in enumerate(recs.iterrows()):
            title = row.Series_Title
            col = cols[i]
            with col:
                st.image(row.Poster_URL, caption=title, width=200)
                choice = st.selectbox(
                    "Your rating:", rating_labels,
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
                    user_fb[title] = (score, cnt+1)
                    cd_fb[title]   = cnt + (20 if score >= 7 else 5)
            # write feedback JSONs
            for obj, fn in [(user_fb,"user_feedback.json"),
                            (cd_fb,"cooldown_feedback.json"),
                            (not_watched,"not_watched.json")]:
                with open(fn, "w", encoding="utf-8") as f:
                    json.dump(obj, f, indent=4)
            st.session_state.search_count = cnt+1
            st.success("Thanks for your feedback! 🎉")
            st.session_state.show_prompt = True
