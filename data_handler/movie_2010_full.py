import pandas as pd
import requests
import re
import time

BASE_URL = "https://api.themoviedb.org/3/movie/"
API_KEY = "dd990daaca9c4d69e96943df8de6acd6"
MOVIES_PATH = "data/movies.csv"
LINKS_PATH = "data/links.csv"
OUTPUT_PATH = "data/movies_2010_full.csv"

# Load MovieLens data
movies = pd.read_csv(MOVIES_PATH)
links = pd.read_csv(LINKS_PATH)

# Extract release year from title
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)

# Filter for movies released in 2010
movies_2010 = movies[movies["year"] == 2010]
movies_2010 = movies_2010.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
movies_2010 = movies_2010.dropna(subset=["tmdbId"])
movies_2010["tmdbId"] = movies_2010["tmdbId"].astype(int)

print(f"🎬 Found {len(movies_2010)} MovieLens movies from 2010 with TMDb IDs")


def fetch_tmdb_metadata(tmdb_id):
    try:
        url = f"{BASE_URL}{tmdb_id}"
        params = {"api_key": API_KEY, "language": "en-US", "append_to_response": "credits"}
        res = requests.get(url, params=params)
        if res.status_code != 200:
            print(f"⚠️ {tmdb_id} returned {res.status_code}")
            return None

        data = res.json()
        return {
            "tmdbId": tmdb_id,
            "Title": data.get("title", ""),
            "Release_Date": data.get("release_date", ""),
            "TMDb_Rating": data.get("vote_average", 0),
            "Genres": ", ".join([g["name"] for g in data.get("genres", [])]),
            "Cast": ", ".join([c["name"] for c in data.get("credits", {}).get("cast", [])[:5]]),
            "Director": ", ".join([c["name"] for c in data.get("credits", {}).get("crew", []) if c["job"] == "Director"]),
            "Overview": data.get("overview", "")
        }
    except Exception as e:
        print(f"Error fetching {tmdb_id}: {e}")
        return None

metadata = []
for i, row in movies_2010.iterrows():
    print(f"🔹 ({i+1}/{len(movies_2010)}) {row['title']}")
    info = fetch_tmdb_metadata(row["tmdbId"])
    if info:
        info["Original_Title"] = row["title"]
        metadata.append(info)
        print(f"Added: {info['Title']}")
    time.sleep(0.25)  # prevent TMDb rate limit

#Convert to DataFrame 
df = pd.DataFrame(metadata)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\nSaved {len(df)} movies to '{OUTPUT_PATH}'")
