import os
import zipfile
import requests
import pandas as pd
from tqdm import tqdm

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"
ZIP_PATH = os.path.join(RAW_DIR, "ml-1m.zip")
EXTRACTED_DIR = os.path.join(RAW_DIR, "ml-1m")


def download_dataset():
    if not os.path.exists(ZIP_PATH):
        print("Descargando MovieLens 1M dataset...")
        response = requests.get(DATA_URL, stream=True)
        with open(ZIP_PATH, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)
    else:
        print("Dataset ya descargado.")


def extract_zip():
    """Descomprime el archivo zip en la carpeta raw"""
    if not os.path.exists(EXTRACTED_DIR):
        print("Extrayendo archivo ZIP...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
    else:
        print("Archivo ya extraído.")

def load_users():
    path = os.path.join(EXTRACTED_DIR, "users.dat")
    return pd.read_csv(path, sep="::", engine="python", encoding="latin-1",
                       names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])


def load_movies():
    path = os.path.join(EXTRACTED_DIR, "movies.dat")
    return pd.read_csv(path, sep="::", engine="python", encoding="latin-1",
                       names=["MovieID", "Title", "Genres"])


def load_ratings():
    path = os.path.join(EXTRACTED_DIR, "ratings.dat")
    return pd.read_csv(path, sep="::", engine="python", encoding="latin-1",
                       names=["UserID", "MovieID", "Rating", "Timestamp"])



def save_clean_csv(df, name):
    output_path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df.to_csv(output_path, index=False)
    print(f"Guardado {name}.csv en {output_path}")


def main():
    download_dataset()
    extract_zip()

    print("Cargando archivos...")
    users = load_users()
    movies = load_movies()
    ratings = load_ratings()

    print("Guardando CSVs procesados...")
    save_clean_csv(users, "users")
    save_clean_csv(movies, "movies")
    save_clean_csv(ratings, "ratings")


if __name__ == "__main__":
    main()
