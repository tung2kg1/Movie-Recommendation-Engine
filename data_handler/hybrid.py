import pandas as pd

MOVIES_PATH = "data/movies_2010_full_with_id.csv"
RATINGS_PATH = "data/ratings.csv"
OUTPUT_PATH = "data/movies_2010_hybrid.csv"

# Load data
movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH)

# Compute average MovieLens rating per movie
avg_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
avg_ratings.rename(columns={"rating": "User_Avg_Rating"}, inplace=True)

# Merge with your movie metadata
movies_hybrid = movies.merge(avg_ratings, on="movieId", how="left")

# Normalize user rating (0–1 scale)
movies_hybrid["User_Avg_Rating"].fillna(movies_hybrid["User_Avg_Rating"].mean(), inplace=True)
movies_hybrid["Rating_Norm"] = movies_hybrid["User_Avg_Rating"] / movies_hybrid["User_Avg_Rating"].max()

# Save new dataset
movies_hybrid.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"Added average user ratings to {len(movies_hybrid)} movies")
print(f"Saved to '{OUTPUT_PATH}'")
