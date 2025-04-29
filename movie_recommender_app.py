# movie_recommender_app.py

import json
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- PATHS ----
BASE_DIR         = Path(__file__).parent.resolve()
IMDB_CSV         = BASE_DIR / "imdb_top_1000.csv"
USER_FB_FILE     = BASE_DIR / "user_feedback.json"
CD_FB_FILE       = BASE_DIR / "cooldown_feedback.json"
NOT_WATCHED_FILE = BASE_DIR / "not_watched.json"
RECS_FILE        = BASE_DIR / "recommendations.json"

# ---- HELPERS ----
def do_rerun():
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.stop()

@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Poster_URL"})
    df["Genre"]       = df["Genre"].fillna("")
    df["IMDB_Rating"] = df["IMDB_Rating"].fillna(df["IMDB_Rating"].mean())
    df["Meta_score"]  = df["Meta_score"].fillna(df["Meta_score"].mean())
    return df

def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=4), encoding="utf-8")

def save_recs(df: pd.DataFrame):
    # overwrite recommendations.json
    RECS_FILE.write_text(df.to_json(orient="records", indent=2),
                         encoding="utf-8")

def load_recs() -> pd.DataFrame:
    if RECS_FILE.exists():
        data = json.loads(RECS_FILE.read_text(encoding="utf-8"))
        return pd.DataFrame.from_records(data)
    return pd.DataFrame()

@st.cache_resource
def build_vect_sim(df: pd.DataFrame):
    vect = TfidfVectorizer(stop_words="english")
    mat  = vect.fit_transform(df["Genre"])
    sim  = cosine_similarity(mat)
    return vect, sim

# ---- LOAD ALL DATA & STATE ----
imdb_df     = load_csv(IMDB_CSV)
user_fb     = load_json(USER_FB_FILE)
cd_fb       = load_json(CD_FB_FILE)
not_watched = load_json(NOT_WATCHED_FILE)
vect, sim   = build_vect_sim(imdb_df)

# restore last recs if any
if "recs" not in st.session_state:
    prev = load_recs()
    if not prev.empty:
        st.session_state.recs     = prev
        st.session_state.feedback = {
            r["Series_Title"]: 0 for _, r in prev.iterrows()
        }

if "search_count" not in st.session_state:
    st.session_state.search_count = len(user_fb)

# ---- RECOMMENDER ----
def hybrid_recommendation(df, sim, genre, year, director):
    d = df.copy()
    if genre != "Any Genre":
        d = d[d["Genre"].str.contains(genre, case=False, na=False)]
    if year:
        d = d[d["Released_Year"].astype(str) == year]
    if director not in ("", "Any Director"):
        d = d[d["Director"].str.contains(director, case=False, na=False)]

    current = st.session_state.search_count
    d = d[~d["Series_Title"].isin(not_watched)]
    d = d[~d["Series_Title"].apply(lambda t: (cd_fb.get(t) or 0) > current)]

    if d.empty:
        return pd.DataFrame()

    d["Weighted_Score"] = d["IMDB_Rating"]*0.7 + (d["Meta_score"]/10)*0.3

    def adjust(r):
        fb = user_fb.get(r["Series_Title"])
        base = r["Weighted_Score"]
        if fb:
            base += (fb[0] - 5)*0.1
        return base

    d["Weighted_Score"] = d.apply(adjust, axis=1)

    idxs    = d.index.tolist()
    avg_sim = sim[idxs].mean(axis=0)[idxs]
    d["Similarity_Score"] = avg_sim

    return (
        d.sort_values(["Weighted_Score","Similarity_Score"], ascending=False)
         .head(3)[["Poster_URL","Series_Title","Released_Year","IMDB_Rating","Weighted_Score"]]
    )

# ---- UI SETUP ----
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender")

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("---")

    with st.expander("🔍 Settings & Filters", expanded=True):
        # build options
        genres    = sorted({g.strip().capitalize()
                            for row in imdb_df["Genre"]
                            for g in row.split(",") if g})
        directors = sorted(imdb_df["Director"].dropna().unique())

        # restore or default
        curr_g = st.session_state.get("genre_sel", "Any Genre")
        opts_g = ["Any Genre"] + genres
        genre_sel = st.selectbox(
            "Genre", opts_g,
            index=opts_g.index(curr_g),
            key="genre_sel"
        )

        curr_y = st.session_state.get("year_sel", "")
        year_sel = st.text_input(
            "Year (leave blank)", value=curr_y, key="year_sel"
        )

        curr_d = st.session_state.get("director_sel", "Any Director")
        opts_d = ["Any Director"] + directors
        director_sel = st.selectbox(
            "Director", opts_d,
            index=opts_d.index(curr_d),
            key="director_sel"
        )

        if st.button("Get Recommendations"):
            recs = hybrid_recommendation(
                imdb_df, sim, genre_sel, year_sel.strip(), director_sel
            )
            if recs.empty:
                st.warning("No matches—try different filters.")
            else:
                st.session_state.recs     = recs
                st.session_state.feedback = {t: 0 for t in recs.Series_Title}
                save_recs(recs)
                st.sidebar.success(f"✅ Saved {len(recs)} recs")
                st.session_state.pop("show_prompt", None)

        if st.button("Start Over"):
            # clear recs & feedback
            for k in ("recs","feedback","show_prompt"):
                st.session_state.pop(k, None)

            # reset filters in state
            st.session_state["genre_sel"]    = "Any Genre"
            st.session_state["year_sel"]     = ""
            st.session_state["director_sel"] = "Any Director"

            # clear saved recs file
            save_recs(pd.DataFrame())

            do_rerun()

# ---- MAIN AREA ----
if st.session_state.get("show_prompt"):
    st.markdown(
        """<style>[data-testid="stAppViewContainer"]{filter:brightness(30%);}</style>""",
        unsafe_allow_html=True
    )
    st.write("## Would you like to search again?")
    c1, c2 = st.columns(2)

    if c1.button("🔍 New Search"):
        # clear state + filters
        for k in ("recs","feedback","show_prompt"):
            st.session_state.pop(k, None)
        st.session_state["genre_sel"]    = "Any Genre"
        st.session_state["year_sel"]     = ""
        st.session_state["director_sel"] = "Any Director"
        save_recs(pd.DataFrame())
        do_rerun()

    if c2.button("⏹️ Exit"):
        st.write("Enjoy your movies! 🍿")
    st.stop()

if "recs" in st.session_state:
    recs = st.session_state.recs
    st.subheader("Top 3 Recommendations")
    cols = st.columns(len(recs))
    labels = [f"{i} = {t}" for i, t in enumerate([
        "Not seen yet","Bad","Poor","Fair","Okay",
        "Average","Good","Very Good","Great",
        "Excellent","Masterpiece"
    ])]

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
                    user_fb[title] = (score, cnt+1)
                    cd_fb[title]   = cnt + (20 if score >= 7 else 5)

            save_json(user_fb,     USER_FB_FILE)
            save_json(cd_fb,       CD_FB_FILE)
            save_json(not_watched, NOT_WATCHED_FILE)

            st.session_state.search_count += 1
            st.success("Thanks for your feedback! 🎉")
            st.session_state.show_prompt = True
