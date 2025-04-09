import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

user_feedback = load_json("user_feedback.json")import streamlit as st
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Hybrid Recommendation Function
def hybrid_recommendation(selected_genre, selected_year, df):
    # Convert the selected genre to lowercase for matching
    genre = selected_genre.strip().lower()
    df['Genre'] = df['Genre'].str.strip().str.lower()

    # Filter by genre and year
    filtered_df = df[df['Genre'].str.contains(genre, case=False, na=False)]
    filtered_df = filtered_df[filtered_df['Released_Year'].astype(str).str.contains(str(selected_year))]
    
    if filtered_df.empty:
        st.write(f"No movies found with genre '{selected_genre}' and year '{selected_year}'. Showing top alternative picks.")
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

# Streamlit App Layout
st.title("Movie Recommender")

# Initialize session state for recommendations & feedback scores
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "feedback_scores" not in st.session_state:
    st.session_state.feedback_scores = {}

if "search_count" not in st.session_state:
    st.session_state.search_count = len(user_feedback)

# Genre and Year inputs using selectbox dropdowns
selected_genre = st.selectbox("Select Genre:", genres_list)
selected_year = st.selectbox("Select Year:", years_list)

if st.button("Get Recommendations"):
    if selected_genre and selected_year:
        st.session_state.recommendations = hybrid_recommendation(selected_genre, selected_year, imdb_df)
        # Initialize feedback scores in session state
        st.session_state.feedback_scores = {
            title: 0 for title in st.session_state.recommendations['Series_Title'].tolist()
        }
    else:
        st.write("Please select both a genre and a year to get recommendations.")

# Display recommendations (if available)
if st.session_state.recommendations is not None:
    st.write("### Top Recommendations:")
    st.write(st.session_state.recommendations)

    # Collect feedback with sliders
    for movie in st.session_state.recommendations['Series_Title'].tolist():
        st.session_state.feedback_scores[movie] = st.slider(
            f"Rate '{movie}' (1-10, or 0 if not watched)",
            0, 10, st.session_state.feedback_scores[movie]
        )

    # Submit feedback button
    if st.button("Submit Feedback"):
        search_count = st.session_state.search_count
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
        st.success("Feedback saved! Future recommendations will improve.")

        # Clear feedback sliders after submission
        st.session_state.feedback_scores = {}

        # Increment search count to ensure unique cooldown periods
        st.session_state.search_count += 1 
