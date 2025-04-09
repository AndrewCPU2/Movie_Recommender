import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to set custom CSS for a dark/light theme toggle
def set_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            .main {background-color: #333333; color: #ffffff;}
            .sidebar .sidebar-content {background-color: #444444; color: #ffffff;}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .main {background-color: #ffffff; color: #000000;}
            .sidebar .sidebar-content {background-color: #f0f0f0; color: #000000;}
            </style>
            """,
            unsafe_allow_html=True
        )

# Load dataset
file_path = "imdb_top_1000.csv"
imdb_df = pd.read_csv(file_path)

# Load existing feedback data
def load_json(file):
    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_json(data, file):
    with open(file, "w") as f:
        json.dump(data, f, indent=4)

user_feedback = load_json("user_feedback.json")
cooldown_movies = load_json("cooldown_feedback.json")
not_watched_movies = load_json("not_watched.json")

# Fill missing values
imdb_df['Genre'] = imdb_df['Genre'].fillna("")
imdb_df['IMDB_Rating'] = imdb_df['IMDB_Rating'].fillna(imdb_df['IMDB_Rating'].mean())

# Compute Genre Similarity for Content-Based Filtering
vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = vectorizer.fit_transform(imdb_df['Genre'])
cosine_sim = cosine_similarity(genre_matrix)

# Extract unique genres from the CSV file
all_genres = set()
for genre_str in imdb_df['Genre']:
    for genre in genre_str.split(","):
        genre = genre.strip()
        if genre:
            all_genres.add(genre.capitalize())
genres_list = sorted(list(all_genres))

# Extract unique release years from the CSV file
years_list = sorted(imdb_df['Released_Year'].dropna().unique())

# Additional filter: Director (if column exists)
if 'Director' in imdb_df.columns:
    directors_list = sorted(imdb_df['Director'].dropna().unique())
else:
    directors_list = []

# Hybrid Recommendation Function
def hybrid_recommendation(selected_genre, selected_year, selected_director, df):
    # Convert the selected genre to lowercase for matching
    genre = selected_genre.strip().lower()
    df['Genre'] = df['Genre'].str.strip().str.lower()

    # Filter by genre and year
    filtered_df = df[df['Genre'].str.contains(genre, case=False, na=False)]
    filtered_df = filtered_df[filtered_df['Released_Year'].astype(str).str.contains(str(selected_year))]
    
    # Further filter by director if provided and available
    if selected_director and 'Director' in df.columns and selected_director != "":
        filtered_df = filtered_df[filtered_df['Director'].str.contains(selected_director, case=False, na=False)]
    
    if filtered_df.empty:
        st.write(f"No movies found with genre '{selected_genre}', year '{selected_year}'", end="")
        if selected_director:
            st.write(f", and director '{selected_director}'", end="")
        st.write(". Showing top alternative picks.")
        filtered_df = df.head(3)  # Provide top alternative recommendations

    # Compute weighted scores
    filtered_df['Meta_score'] = filtered_df['Meta_score'].fillna(filtered_df['Meta_score'].mean())
    filtered_df['Normalized_Meta'] = filtered_df['Meta_score'] / 10
    filtered_df['Weighted_Score'] = (filtered_df['IMDB_Rating'] * 0.7) + (filtered_df['Normalized_Meta'] * 0.3)

    # Compute similarity scores
    movie_indices = filtered_df.index.tolist()
    avg_sim_scores = cosine_sim[movie_indices].mean(axis=0)
    filtered_df['Similarity_Score'] = avg_sim_scores[movie_indices]

    # Sort by Hybrid Score
    filtered_df = filtered_df.sort_values(by='Weighted_Score', ascending=False)

    # Get top 3 recommendations
    top_movies = filtered_df.head(3)
    return top_movies[['Series_Title', 'Released_Year', 'IMDB_Rating', 'Weighted_Score']]

# Sidebar for theme selection and filters
st.sidebar.title("Filters & Settings")

# Theme toggle
theme_choice = st.sidebar.radio("Choose Theme", ("Light", "Dark"))
set_theme(theme_choice)

# Custom logo (make sure "logo.png" is present in your project folder)
try:
    st.sidebar.image("logo.png", use_column_width=True)
except Exception as e:
    st.sidebar.write("Logo not found.")

# Filter options
selected_genre = st.sidebar.selectbox("Select Genre:", genres_list)
selected_year = st.sidebar.selectbox("Select Year:", years_list)

selected_director = ""
if directors_list:
    selected_director = st.sidebar.selectbox("Select Director (Optional):", [""] + directors_list)

st.sidebar.markdown("---")
st.sidebar.write("Adjust the filters to get better movie recommendations!")

# Main Page Layout
st.title("Movie Recommender")
st.write("Welcome! Get personalized movie recommendations based on your preferences.")

# Button to get recommendations with a spinner for progress indication
if st.sidebar.button("Get Recommendations"):
    with st.spinner("Fetching recommendations..."):
        st.session_state.recommendations = hybrid_recommendation(selected_genre, selected_year, selected_director, imdb_df)
        # Initialize feedback scores in session state
        st.session_state.feedback_scores = {
            title: 0 for title in st.session_state.recommendations['Series_Title'].tolist()
        }

# Display recommendations and feedback sliders if available
if "recommendations" in st.session_state and st.session_state.recommendations is not None:
    st.subheader("Top Recommendations:")
    st.write(st.session_state.recommendations)

    st.write("### Rate the recommendations:")
    # Use columns to display sliders side by side
    cols = st.columns(len(st.session_state.recommendations))
    movie_list = st.session_state.recommendations['Series_Title'].tolist()
    for idx, movie in enumerate(movie_list):
        with cols[idx]:
            st.write(movie)
            # Provide a unique key for each slider widget
            st.session_state.feedback_scores[movie] = st.slider(
                "Rating (1-10, 0 if not watched)",
                0, 10,
                st.session_state.feedback_scores[movie],
                key=f"slider_{movie}_{idx}"
            )

    # Submit Feedback button with progress indication
    if st.button("Submit Feedback"):
        with st.spinner("Saving feedback..."):
            search_count = st.session_state.search_count if "search_count" in st.session_state else len(user_feedback)
            for movie_title, rating in st.session_state.feedback_scores.items():
                if rating == 0:
                    not_watched_movies[movie_title] = search_count
                else:
                    # Store user feedback with incremented search count
                    user_feedback[movie_title] = (rating, search_count + 1)

                    # Adjust cooldown rules
                    if rating >= 7:
                        cooldown_movies[movie_title] = search_count + 20  # Longer cooldown for liked movies
                    elif rating < 7:
                        cooldown_movies[movie_title] = max(cooldown_movies.get(movie_title, 0), search_count + 5)  # Shorter cooldown

            # Save feedback to JSON files
            save_json(user_feedback, "user_feedback.json")
            save_json(cooldown_movies, "cooldown_feedback.json")
            save_json(not_watched_movies, "not_watched.json")
            st.success("Feedback saved! Thank you for your input.")

            # Clear feedback sliders after submission and update search count
            st.session_state.feedback_scores = {}
            st.session_state.search_count = search_count + 1
            # Clear feedback sliders after submission and update search count
            st.session_state.feedback_scores = {}
            st.session_state.search_count = search_count + 1 
