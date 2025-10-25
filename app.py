import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "dd990daaca9c4d69e96943df8de6acd6"  
MOVIES_PATH = "movies_2010_hybrid.csv"
RATINGS_PATH = "ratings.csv"
RATINGS_2010_PATH = "ratings_2010_users.csv"

st.set_page_config(page_title="🎬 AI Movie Recommender", layout="wide")

#Loading data
@st.cache_data
def load_data():
    movies = pd.read_csv(MOVIES_PATH)
    ratings_2010 = pd.read_csv(RATINGS_2010_PATH)
    return movies, ratings_2010

movies, ratings_2010 = load_data()

#Content similarity matrix
@st.cache_resource
def build_similarity_matrix(movies_df):
    movies_df["Combined"] = (
        movies_df["Genres"].fillna("") + " " +
        movies_df["Cast"].fillna("") + " " +
        movies_df["Director"].fillna("")
    )
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["Combined"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies_df.index, index=movies_df["Title"]).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = build_similarity_matrix(movies)

#TMbD poster loader
@st.cache_data
def loading_poster(title):
    """Get poster from TMDb API by movie title"""
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    try:
        res = requests.get(url).json()
        poster_path = res["results"][0]["poster_path"]
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return "https://via.placeholder.com/300x450?text=No+Image"

# Hybrid recommendation function
def recommend_movies(favorites, top_n=10,
                     w_user_rating=0.2, w_tmdb=0.2):

    sim_scores = np.zeros(len(movies))
    for title in favorites:
        if title not in indices:
            continue
        idx = indices[title]
        sim_scores += cosine_sim[idx]
    sim_scores = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min())

    movies["User_Rating_Norm"] = movies["User_Avg_Rating"] / movies["User_Avg_Rating"].max()
    movies["TMDb_Rating_Norm"] = movies["TMDb_Rating"] / movies["TMDb_Rating"].max()

    movies["HybridScore"] = (
        sim_scores
        + w_user_rating * movies["User_Rating_Norm"]
        + w_tmdb * movies["TMDb_Rating_Norm"]
    )

    recs = movies.sort_values("HybridScore", ascending=False)
    recs = recs[~recs["Title"].isin(favorites)].head(top_n)

    return recs[["Title", "Genres", "Director", "TMDb_Rating", "User_Avg_Rating"]]

# Streamlit interface
st.title("🎥 AI-Powered Hybrid Movie Recommender")
st.markdown("Select movies you like and get smart recommendations based on similar genres, actors, directors, and user preferences.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
user_rating_weight = st.sidebar.slider("Weight: User Rating", 0.0, 0.5, 0.2, 0.05)
tmdb_rating_weight = st.sidebar.slider("Weight: TMDb Rating", 0.0, 0.5, 0.2, 0.05)
num_recommendations = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Movie search
st.subheader("🔍 Search for Movies You Like")
query = st.text_input("Type a movie name to search (e.g., Inception, Shutter Island):")

if query:
    matches = movies[movies["Title"].str.contains(query, case=False, na=False)]
    if len(matches) > 0:
        cols = st.columns(5)
        for i, (_, row) in enumerate(matches.head(5).iterrows()):
            with cols[i % 5]:
                poster = loading_poster(row["Title"])
                st.image(poster, width=150)
                if st.button(f"❤️ {row['Title']}", key=f"fav_{i}"):
                    st.session_state.setdefault("favorites", []).append(row["Title"])
    else:
        st.info("No matches found.")

# Favorites display
st.subheader("⭐ Your Favorite Movies")

if "favorites" not in st.session_state:
    st.session_state["favorites"] = []

if st.session_state["favorites"]:
    fav_cols = st.columns(len(st.session_state["favorites"]))
    for i, title in enumerate(st.session_state["favorites"]):
        with fav_cols[i]:
            st.image(loading_poster(title), width=150)
            st.caption(title)
            if st.button(f"❌ Remove {title}", key=f"remove_{i}"):
                st.session_state["favorites"].remove(title)
                recommendations = recommend_movies(
                st.session_state["favorites"],
                top_n=num_recommendations,
                w_user_rating=user_rating_weight,
                w_tmdb=tmdb_rating_weight
            )
else:
    st.write("No favorites selected yet.")

# Recommendation section
if st.button("🎬 Recommend Movies"):
    if not st.session_state["favorites"]:
        st.warning("Please select at least one favorite movie first!")
    else:
        with st.spinner("Finding the best matches... 🎥"):
            recommendations = recommend_movies(
                st.session_state["favorites"],
                top_n=num_recommendations,
                w_user_rating=user_rating_weight,
                w_tmdb=tmdb_rating_weight
            )

        st.success("✅ Recommendations Ready!")
        cols = st.columns(5)
        for i, (_, row) in enumerate(recommendations.iterrows()):
            with cols[i % 5]:
                st.image(loading_poster(row["Title"]), width=150)
                st.markdown(f"**{row['Title']}**")
                st.caption(f"{row['Genres']} — 🎬 {row['Director']}")
                st.write(f"⭐ TMDb: {row['TMDb_Rating']}, 👥 ML: {row['User_Avg_Rating']:.2f}")
