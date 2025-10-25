import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


print("📂 Loading datasets...")

movies = pd.read_csv("movies_2010_hybrid.csv")
ratings = pd.read_csv("ml-25m/ratings.csv")
ratings_2010 = pd.read_csv("ratings_2010_users.csv")

print(f"✅ Loaded {len(movies)} movies and {len(ratings_2010)} 2010 ratings.")

# Build movie content matrix
movies["Combined"] = (
    movies["Genres"].fillna("") + " " +
    movies["Cast"].fillna("") + " " +
    movies["Director"].fillna("")
)

print("🔍 Computing TF-IDF content similarity...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["Combined"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies["Title"]).drop_duplicates()
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


print("Filtering to active users (≥10 ratings)...")
active_users = ratings_2010.groupby('userId').size()
active_users = active_users[active_users >= 10].index
ratings_active = ratings_2010[ratings_2010['userId'].isin(active_users)]

print(f"✅ {len(active_users)} active users remain after filtering.")

# Build user–movie rating matrix
user_movie_matrix = ratings_active.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Convert to sparse matrix
user_movie_sparse = csr_matrix(user_movie_matrix.values)

print("👥 Computing sparse cosine similarity...")
user_sim_sparse = cosine_similarity(user_movie_sparse, dense_output=False)

# Store mapping from user index to user ID
user_ids = user_movie_matrix.index.tolist()
print(f"✅ Sparse user similarity computed for {len(user_ids)} users.")

# Get top-k similar users for a given user
def get_similar_users_sparse(user_id, top_k=20):
    if user_id not in user_ids:
        return pd.Series([], dtype=float)
    idx = user_ids.index(user_id)
    row = user_sim_sparse.getrow(idx).toarray().ravel()
    top_idx = row.argsort()[-(top_k+1):-1][::-1]
    return pd.Series(row[top_idx], index=[user_ids[i] for i in top_idx])

#Aggregate similar user influence

def get_similar_user_influence(user_id, similar_users):
    if similar_users.empty:
        return pd.DataFrame(columns=['movieId', 'SimilarUserScore'])

    sim_users = similar_users.index
    sim_weights = similar_users.values

    sim_ratings = ratings_active[ratings_active['userId'].isin(sim_users)]
    movie_weighted_scores = sim_ratings.groupby('movieId')['rating'].apply(
        lambda x: np.average(x, weights=sim_weights[:len(x)]) if len(x) > 0 else 0
    ).reset_index(name='SimilarUserScore')

    return movie_weighted_scores

# Hybrid Recommendation Function

def hybrid_recommend_for_user(user_id, favorites, top_n=10,
                              w_user_rating=0.2, w_tmdb=0.2, w_similar_user=0.3):
    """
    Hybrid recommender combining:
    - TF-IDF content similarity
    - MovieLens + TMDb ratings
    - Similar-user influence (collaborative filtering)
    """

    # Step 1. Content similarity
    sim_scores = np.zeros(len(movies))
    for title in favorites:
        if title not in indices:
            print(f"⚠️ '{title}' not found.")
            continue
        idx = indices[title]
        sim_scores += cosine_sim[idx]
    sim_scores = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min())

    # Step 2. Normalize base ratings
    movies["User_Rating_Norm"] = movies["User_Avg_Rating"] / movies["User_Avg_Rating"].max()
    movies["TMDb_Rating_Norm"] = movies["TMDb_Rating"] / movies["TMDb_Rating"].max()

    # Step 3. Similar user influence
    similar_users = get_similar_users_sparse(user_id, top_k=20)
    similar_user_scores = get_similar_user_influence(user_id, similar_users)
    movies_scored = movies.merge(similar_user_scores, on='movieId', how='left')
    movies_scored["SimilarUserScore"].fillna(0, inplace=True)

    # Step 4. Final hybrid score
    movies_scored["HybridScore"] = (
        sim_scores
        + w_user_rating * movies_scored["User_Rating_Norm"]
        + w_tmdb * movies_scored["TMDb_Rating_Norm"]
        + w_similar_user * (movies_scored["SimilarUserScore"] / 5.0)
    )

    # Step 5. Rank top movies
    recommendations = movies_scored.sort_values("HybridScore", ascending=False)
    recommendations = recommendations[~recommendations["Title"].isin(favorites)]

    return recommendations.head(top_n)[
        ["Title", "Genres", "Director", "TMDb_Rating", "User_Avg_Rating", "HybridScore"]
    ]


if __name__ == "__main__":
    user_id = int(user_ids[0])  # pick any valid active user
    user_favorites = ["Inception", "The Social Network", "Shutter Island"]

    print(f"\n🎬 Generating hybrid recommendations for user {user_id}...")
    recs = hybrid_recommend_for_user(user_id, user_favorites, top_n=10)

    print("\n✅ Top Hybrid Recommendations:\n")
    print(recs.to_string(index=False))
