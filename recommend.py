
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import pickle
from difflib import get_close_matches

# ----------------------------
# 1️⃣ Load preprocessed data
# ----------------------------
movies_cleaned = pd.read_csv("movies_cleaned.csv")
sig_matrix = load_npz("sig_matrix.npz")
with open("indices.pkl", "rb") as f:
    indices = pickle.load(f)

# ----------------------------
# 2️⃣ Combine text features for richer plot understanding
# ----------------------------
def combine_features(df):
    # Use overview + keywords + genres
    df['combined'] = df['overview'].fillna('') + ' ' + df['keywords'].fillna('') + ' ' + df['genres'].fillna('')
    return df

movies_cleaned = combine_features(movies_cleaned)

# ----------------------------
# 3️⃣ Recommendation function
# ----------------------------
def recommend_movies(title, similarity_matrix=sig_matrix):
    # Make the search case-insensitive
    title_lower = title.lower()
    movie_titles = movies_cleaned['original_title'].str.lower().tolist()

    # Find closest match
    closest = get_close_matches(title_lower, movie_titles, n=1, cutoff=0.6)
    if not closest:
        return f"Movie '{title}' not found. Try a different title!"

    # Get the index of the matched movie
    matched_idx = movie_titles.index(closest[0])

    # Compute similarity scores
    similarity_scores = list(enumerate(similarity_matrix[matched_idx].toarray().flatten()))
    top_indices = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:11]

    # Return recommended movie titles
    recommended_titles = [movies_cleaned['original_title'].iloc[i] for i, _ in top_indices]
    return recommended_titles

# ----------------------------
# 4️⃣ Streamlit UI
# ----------------------------
st.title("Movie Recommendation Engine")
st.subheader("Type a movie and get 10 similar movies based on plot, keywords, and genres!")
st.markdown("🎵 Let’s find your next movie adventure!")

movie_input = st.text_input("Enter a movie title:")
if movie_input:
    recommendations = recommend_movies(movie_input)
    if isinstance(recommendations, list):
        st.write("🎥 Recommended movies:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.write(recommendations)
