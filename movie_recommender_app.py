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

@st.cache_data
def load_json(path: str) -> Dict:
    try:
        with open(path, "r") as f:
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
    if year != "Any Year":
        d = d[d["Released_Year"].astype(str) == year]
    if director not in ("", "Any Director"):
        d = d[d["Director"].str.contains(director, case=False, na=False)]
    if d.empty:
        return pd.DataFrame()

    d = d.assign(
        Weighted_Score=d["IMDB_Rating"] * 0.7 + (d["Meta_score"] / 10) * 0.3
    )
    idxs = d.index.tolist()
    avg_sim = sim[idxs].mean(axis=0)[idxs]
    d["Similarity_Score"] = avg_sim

    return (
        d.sort_values("Weighted_Score", ascending=False)
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

# filter options
all_genres = sorted({g.strip().capitalize()
                     for row in imdb_df["Genre"]
                     for g in row.split(",") if g})
years = sorted(
    pd.to_numeric(imdb_df["Released_Year"], errors="coerce")
      .dropna().astype(int).astype(str)
      .unique().tolist()
)
directors = sorted(imdb_df.get("Director", pd.Series()).dropna().unique())

# ---- SIDEBAR ----

with st.sidebar.expander("🔍 Settings & Filters", expanded=True):
    genre_sel    = st.selectbox("Genre", ["Any Genre"] + all_genres)
    year_sel     = st.selectbox("Year", ["Any Year"] + years)
    director_sel = st.selectbox("Director", ["Any Director"] + directors)

    if st.button("Get Recommendations"):
        recs = hybrid_recommendation(imdb_df, sim, genre_sel, year_sel, director_sel)
        st.session_state.recs = recs
        if not recs.empty:
            st.session_state.feedback = {t: 0 for t in recs.Series_Title}

# ---- MAIN AREA ----

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
                    # smaller poster: width=200
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
                cnt = st.session_state.get("search_count", len(user_fb))
                for title, score in st.session_state.feedback.items():
                    if score == 0:
                        not_watched[title] = cnt
                    else:
                        user_fb[title] = (score, cnt + 1)
                        cd_fb[title]   = cnt + (20 if score >= 7 else 5)

                # persist feedback
                with open("user_feedback.json","w")     as f: json.dump(user_fb,     f, indent=4)
                with open("cooldown_feedback.json","w") as f: json.dump(cd_fb,       f, indent=4)
                with open("not_watched.json","w")       as f: json.dump(not_watched, f, indent=4)

                st.success("Thanks for your feedback! 🎉")

                # --- modal fallback snippet ---
                def _render_search_again_buttons():
                    c1, c2 = st.columns(2)
                    if c1.button("Yes, new search"):
                        for k in ("recs", "feedback"):
                            st.session_state.pop(k, None)
                        st.experimental_rerun()
                    if c2.button("No, exit"):
                        st.write("Enjoy your movies! 🍿")

                if hasattr(st, "modal"):
                    with st.modal("search_again_modal"):
                        st.write("**Would you like to search again?**")
                        _render_search_again_buttons()
                else:
                    st.markdown(
                        """
                        <style>
                        [data-testid="stAppViewContainer"] {
                            filter: brightness(40%);
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("**Would you like to search again?**")
                    _render_search_again_buttons()
