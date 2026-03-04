import urllib.request
import zipfile
from pathlib import Path
from collections import defaultdict

class MovieLens:
    def __init__(self):
        self.movielens_url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        self.movielens_zip = "data/movielens/ml-100k.zip"
        self.movielens_dir = Path("data/movielens/")
    
    def download_movielens_data(self):
        urllib.request.urlretrieve(self.movielens_url, self.movielens_zip)
    
    def extract_movielens_data(self):
        with zipfile.ZipFile(self.movielens_zip, "r") as zip_ref:
            zip_ref.extractall(self.movielens_dir)
    
    def read_movielens_data(self):
        user_movies = defaultdict(set)
        
        with open(self.movielens_dir / "ml-100k" / "u.data", "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    user_id = int(parts[0])
                    movie_id = int(parts[1])
                    rating = int(parts[2])
                    user_movies[user_id].add(movie_id)
        return user_movies