import pandas as pd

# Input paths
MOVIES_2010_PATH = "data/movies_2010_full.csv"
LINKS_PATH = "data/links.csv"
OUTPUT_PATH = "data/movies_2010_full_with_id.csv"

# Load datasets
movies_2010 = pd.read_csv(MOVIES_2010_PATH)
links = pd.read_csv(LINKS_PATH)

# Ensure tmdbId type matches
movies_2010['tmdbId'] = movies_2010['tmdbId'].astype(int)
links = links.dropna(subset=['tmdbId'])
links['tmdbId'] = links['tmdbId'].astype(int)

# Merge to bring back movieId
merged = movies_2010.merge(links[['movieId', 'tmdbId']], on='tmdbId', how='left')

# Save result
merged.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

print(f"Added 'movieId' to 2010 movie dataset.")
print(f"Saved to '{OUTPUT_PATH}'")
print(merged.head())
