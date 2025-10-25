import pandas as pd
import re

# Path to MovieLens 25M data
MOVIES_PATH = "data/movies.csv"
OUTPUT_PATH = "data/movies_2010.csv"

# Load the movies.csv file
movies = pd.read_csv(MOVIES_PATH)

# Extract release year from title 
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

# Filter for movies released in 2010
movies_2010 = movies[movies["year"] == 2010]

# Save to CSV
movies_2010.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"✅ Found {len(movies_2010)} movies released in 2010.")
print(f"💾 Saved to '{OUTPUT_PATH}'")
