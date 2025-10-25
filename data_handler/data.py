import pandas as pd
import requests
import time

API_KEY = "dd990daaca9c4d69e96943df8de6acd6"
MOVIES_PATH = "data/movies.csv"
LINKS_PATH = "data/links.csv"
OUTPUT_PATH = "data/movies_2010.csv"

# Load MovieLens data
movies = pd.read_csv(MOVIES_PATH)
links = pd.read_csv(LINKS_PATH)

# Merge movies with tmdbId
df = movies.merge(links[['movieId', 'tmdbId']], on='movieId', how='left')
df = df.dropna(subset=['tmdbId'])
df['tmdbId'] = df['tmdbId'].astype(int)

print(f"Loaded {len(df)} movies with TMDb IDs")

#TMDb API base URL
BASE_URL = "https://api.themoviedb.org/3/movie/"

def get_release_year(tmdb_id):
    try:
        url = f"{BASE_URL}{tmdb_id}"
        params = {"api_key": API_KEY, "language": "en-US"}
        res = requests.get(url, params=params)
        if res.status_code != 200:
            return None
        data = res.json()
        release_date = data.get("release_date", "")
        if release_date:
            return int(release_date[:4])
    except Exception:
        return None
    return None

#Fetch movie years
selected_movies = []
for i, row in df.iterrows():
    tmdb_id = row['tmdbId']
    title = row['title']

    year = get_release_year(tmdb_id)
    if year == 2010:
        selected_movies.append(row)
        print(f"{title} ({year})")
    else:
        print(f"{title} ({year})")

    time.sleep(0.25)  # avoid API rate limit

# Save filtered movies
movies_2010 = pd.DataFrame(selected_movies)
movies_2010.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

print(f"\nSaved {len(movies_2010)} movies released in 2010 to '{OUTPUT_PATH}'")
