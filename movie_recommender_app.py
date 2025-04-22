import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------
# Caching data loading and vector computations
# ----------------------------------------
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df['Genre'] = df['Genre'].fillna("")
    df['IMDB_Rating'] = df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean())
    df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mean())
    return df

@st.cache_data
def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def build_similarity(df):
    vect = TfidfVectorizer(stop_words='english')
    mat = vect.fit_transform(df['Genre'])
    sim = cosine_similarity(mat)
    return sim

imdb_df = load_csv("imdb_top_1000.csv")
cosine_sim = build_similarity(imdb_df)

# ----------------------------------------
# Theme CSS
# ----------------------------------------
def set_theme(theme: str):
    bg, fg = ("#333333", "#ffffff") if theme=="Dark" else ("#ffffff", "#000000")
    sidebar_bg, sidebar_fg = ("#444444","#ffffff") if theme=="Dark" else ("#f0f0f0","#000000")
    st.markdown(f"""
        <style>
          [data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            color: {sidebar_fg};
          }}
          [data-testid="stAppViewContainer"] {{
            background-color: {bg};
            color: {fg};
          }}
        </style>
    """, unsafe_allow_html=True)

# ----------------------------------------
# Feedback storage
# ----------------------------------------
user_feedback     = load_json("user_feedback.json")
cooldown_feedback = load_json("cooldown_feedback.json")
not_watched       = load_json("not_watched.json")

def save_all():
    with open("user_feedback.json","w")     as f: json.dump(user_feedback, f,     indent=4)
    with open("cooldown_feedback.json","w") as f: json.dump(cooldown_feedback, f, indent=4)
    with open("not_watched.json","w")       as f: json.dump(not_watched, f,       indent=4)

# ----------------------------------------
# Prepare filter options
# ----------------------------------------
# Genres
all_genres = sorted({g.strip().capitalize()
                     for row in imdb_df['Genre'] for g in row.split(",") if g.strip()})
genres_list = ["Any Genre"] + all_genres

# Years
years_list = ["Any Year"] + sorted(imdb_df['Released_Year'].dropna().astype(int).astype(str).unique().tolist())

# Directors
directors = sorted(imdb_df['Director'].dropna().unique())
directors_list = ["Any Director"] + directors if directors else []

# ----------------------------------------
# Hybrid recommendation logic
# ----------------------------------------
def hybrid_recommendation(genre, year, director, df, sim):
    df_filtered = df.copy()
    if genre!="Any Genre":
        mask = df['Genre'].str.lower().str.contains(genre.lower(), na=False)
        df_filtered = df_filtered[mask]
    if year!="Any Year":
        df_filtered = df_filtered[df_filtered['Released_Year'].astype(str)==year]
    if director!="Any Director":
        df_filtered = df_filtered[df_filtered['Director'].str.contains(director, case=False, na=False)]

    if df_filtered.empty:
        st.error("No movies match those filters. Try something else.")
        return pd.DataFrame()

    # Weighted score
    df_filtered = df_filtered.assign(
        Normalized_Meta = df_filtered['Meta_score'] / 10,
        Weighted_Score   = df_filtered['IMDB_Rating']*0.7 + (df_filtered['Meta_score']/10)*0.3
    )
    # Similarity (average across chosen subset)
    idxs = df_filtered.index.tolist()
    avg_sim = sim[idxs].mean(axis=0)[idxs]
    df_filtered['Similarity_Score'] = avg_sim

    recs = df_filtered.sort_values("Weighted_Score", ascending=False).head(3)
    return recs[["Series_Title","Released_Year","IMDB_Rating","Weighted_Score"]]

# ----------------------------------------
# Sidebar UI
# ----------------------------------------
st.sidebar.title("Settings & Filters")
theme = st.sidebar.radio("Theme", ["Light","Dark"])
set_theme(theme)

selected_genre    = st.sidebar.selectbox("Genre", genres_list)
selected_year     = st.sidebar.selectbox("Year", years_list)
selected_director = (st.sidebar.selectbox("Director", directors_list)
                     if directors_list else "Any Director")

if "search_count" not in st.session_state:
    st.session_state.search_count = len(user_feedback)

if st.sidebar.button("Get Recommendations"):
    recs = hybrid_recommendation(selected_genre, selected_year, selected_director, imdb_df, cosine_sim)
    st.session_state.recommendations = recs
    if not recs.empty:
        # init feedback scores
        st.session_state.feedback = {t: 0 for t in recs.Series_Title}

# ----------------------------------------
# Main panel
# ----------------------------------------
st.title("Movie Recommender")
st.write("Personalized picks based on what you like!")

if st.session_state.get("recommendations") is not None:
    recs = st.session_state.recommendations
    if not recs.empty:
        st.subheader("Top 3 Recommendations")
        st.dataframe(recs, use_container_width=True)

        st.subheader("Rate Them")
        cols = st.columns(len(recs))
        for i,(_,row) in enumerate(recs.iterrows()):
            title = row.Series_Title
            with cols[i]:
                st.write(title)
                st.session_state.feedback[title] = st.slider(
                    "Your rating",
                    min_value=0, max_value=10,
                    value=st.session_state.feedback[title],
                    key=f"feed_{i}"
                )

        if st.button("Submit Feedback"):
            cnt = st.session_state.search_count
            for title,score in st.session_state.feedback.items():
                if score==0:
                    not_watched[title] = cnt
                else:
                    user_feedback[title] = (score, cnt+1)
                    cooldown_feedback[title] = cnt + (20 if score>=7 else 5)
            save_all()
            st.success("Thanks for your feedback!")
            st.session_state.search_count += 1
            st.session_state.feedback.clear()
    else:
        st.info("No recommendations to show.")
