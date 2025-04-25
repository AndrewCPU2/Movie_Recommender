import streamlit as st
import pandas as pd
import json
from typing import Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ---- DATA LOADING & CACHING ----

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Poster_URL"})
    df['Genre'] = df['Genre'].fillna("")
    df['IMDB_Rating'] = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean())
    df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mean())
    return df

def load_json(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_resource
def build_vectorizer_and_sim(df: pd.DataFrame) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    vect = TfidfVectorizer(stop_words="english")
    mat = vect.fit_transform(df["Genre"])
    sim = cosine_similarity(mat)
    return vect, sim

# ---- RECOMMENDER LOGIC ----

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
    if d.empty:
        return pd.DataFrame()

    # base weighted score
    d = d.assign(
        Weighted_Score = d["IMDB_Rating"] * 0.7 + (d["Meta_score"] / 10) * 0.3
    )

    # adjust by user feedback
    def adjust_score(row):
        title = row["Series_Title"]
        base = row["Weighted_Score"]
        fb = user_fb.get(title)
        if fb:
            rating, _ = fb
            base += (rating - 5) * 0.1  # feedback multiplier
        return base

    d["Weighted_Score"] = d.apply(adjust_score, axis=1)

    # optionally filter out hated movies
    hated = [t for t, (r, _) in user_fb.items() if r <= 2]
    if hated:
        d = d[~d["Series_Title"].isin(hated)]
        if d.empty:
            return pd.DataFrame()

    # similarity score
    idxs = d.index.tolist()
    avg_sim = sim[idxs].mean(axis=0)[idxs]
    d["Similarity_Score"] = avg_sim

    return (
        d.sort_values(["Weighted_Score", "Similarity_Score"], ascending=False)
         .head(3)[[
             "Poster_URL",
             "Series_Title",
             "Released_Year",
             "IMDB_Rating",
             "Weighted_Score"
         ]]
    )

# ---- UI SETUP ----

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# load data & models
imdb_df     = load_csv("imdb_top_1000.csv")
user_fb     = load_json("user_feedback.json")
cd_fb       = load_json("cooldown_feedback.json")
not_watched = load_json("not_watched.json")
vect, sim   = build_vectorizer_and_sim(imdb_df)

# initialize search_count if missing\if not in session_state
if "search_count" not in st.session_state:
    st.session_state.search_count = len(user_fb)

# build filter options
all_genres = sorted({
    g.strip().capitalize()
    for row in imdb_df["Genre"] for g in row.split(",") if g
})
directors = sorted(imdb_df.get("Director", pd.Series()).dropna().unique())

# ---- SIDEBAR ----

with st.sidebar.expander("🔍 Settings & Filters", expanded=True):
    genre_sel    = st.selectbox("Genre", ["Any Genre"] + all_genres)
    year_sel     = st.text_input("Year (leave blank for any)", value="")
    director_sel = st.selectbox("Director", ["Any Director"] + directors)

    if st.button("Get Recommendations"):
        recs = hybrid_recommendation(
            imdb_df, sim,
            genre_sel,
            year_sel.strip(),
            director_sel
        )
        st.session_state.recs = recs
        if not recs.empty:
            st.session_state.feedback = {t: 0 for t in recs.Series_Title}
            st.session_state.pop("show_prompt", None)

    if st.button("Start Over"):
        for key in ("recs", "feedback", "show_prompt"):
            st.session_state.pop(key, None)
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.stop()

# ---- MAIN AREA ----

# show “search again?” prompt
if st.session_state.get("show_prompt"):
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            filter: brightness(30%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.write("## Would you like to search again?")
    c1, c2 = st.columns(2)
    if c1.button("🔍 New Search"):
        for key in ("recs", "feedback", "show_prompt"):
            st.session_state.pop(key, None)
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            st.stop()
    if c2.button("⏹️ Exit"):
        st.write("Enjoy your movies! 🍿")
    st.stop()

# recommendations + feedback form
if "recs" in st.session_state:
    recs = st.session_state.recs

    if recs.empty:
        st.warning("No movies match those filters. Please try again.")
    else:
        st.subheader("Top 3 Recommendations")
        cols = st.columns(len(recs))

        with st.form("feedback_form"):
            for i, (_, row) in enumerate(recs.iterrows()):
                title = row.Series_Title
                col = cols[i]
                with col:
                    st.image(
                        row.Poster_URL,
                        caption=title,
                        use_container_width=False,
                        width=200
                    )
                    st.session_state.feedback[title] = st.slider(
                        "Your rating", 0, 10,
                        st.session_state.feedback.get(title, 0),
                        key=f"slider_{i}"
                    )

            submitted = st.form_submit_button("Submit Feedback")
            if submitted:
                cnt = st.session_state.search_count
                for title, score in st.session_state.feedback.items():
                    if score == 0:
                        not_watched[title] = cnt
                    else:
                        user_fb[title] = (score, cnt + 1)
                        cd_fb[title]   = cnt + (20 if score >= 7 else 5)

                # write out the JSONs freshly to disk
                with open("user_feedback.json",     "w", encoding="utf-8") as f:
                    json.dump(user_fb,     f, indent=4)
                with open("cooldown_feedback.json","w", encoding="utf-8") as f:
                    json.dump(cd_fb,       f, indent=4)
                with open("not_watched.json",      "w", encoding="utf-8") as f:
                    json.dump(not_watched, f, indent=4)

                # increment search count for next turn
                st.session_state.search_count = cnt + 1

                st.success("Thanks for your feedback! 🎉")
                st.session_state.show_prompt = True
