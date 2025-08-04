import pandas as pd
from sklearn.preprocessing import LabelEncoder

def build_features():
    # Cargar datos
    ratings = pd.read_csv("data/processed/ratings.csv")
    users = pd.read_csv("data/processed/users_enriched.csv")
    movies = pd.read_csv("data/processed/movies.csv")
    clusters = pd.read_csv("data/processed/users_clustered.csv")
    clusters = clusters[["UserID", "GeoCluster"]]

    # Merge
    df = ratings.merge(users, on="UserID").merge(movies, on="MovieID")
    df = df.merge(clusters, on="UserID")

    # Agrupar edades
    age_map = {
        1: "<18", 18: "18-24", 25: "25-34", 35: "35-44",
        45: "45-49", 50: "50-55", 56: "56+"
    }
    df["AgeGroup"] = df["Age"].map(age_map)

    # Ocupación legible
    occupation_map = {
        0: "other", 1: "educator", 2: "artist", 3: "admin",
        4: "student", 5: "customer service", 6: "doctor", 7: "manager",
        8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
        12: "programmer", 13: "retired", 14: "sales", 15: "scientist",
        16: "self-employed", 17: "engineer", 18: "tradesman",
        19: "unemployed", 20: "writer"
    }
    df["OccupationLabel"] = df["Occupation"].map(occupation_map)

    # Género binario
    df["GenderBinary"] = df["Gender"].map({"F": 0, "M": 1})

    # Estadísticas por usuario
    user_stats = df.groupby("UserID")["Rating"].agg(["count", "mean"]).reset_index()
    user_stats.columns = ["UserID", "NumRatingsUser", "AvgRatingUser"]
    df = df.merge(user_stats, on="UserID", how="left")

    # Estadísticas por película
    movie_stats = df.groupby("MovieID")["Rating"].agg(["count", "mean"]).reset_index()
    movie_stats.columns = ["MovieID", "NumRatingsMovie", "AvgRatingMovie"]
    df = df.merge(movie_stats, on="MovieID", how="left")

    # Género principal
    df["Genres_List"] = df["Genres"].str.split("|")
    df["MainGenre"] = df["Genres_List"].apply(lambda x: x[0] if isinstance(x, list) else None)

    # GeoCluster (si existe)
    if "GeoCluster" not in df.columns:
        df["GeoCluster"] = -1

    # ---- Encoding ----
    le_age = LabelEncoder()
    le_occ = LabelEncoder()
    le_genre = LabelEncoder()

    df["AgeEncoded"] = le_age.fit_transform(df["AgeGroup"])
    df["OccupationEncoded"] = le_occ.fit_transform(df["OccupationLabel"])
    df["MainGenreEncoded"] = le_genre.fit_transform(df["MainGenre"])

    # Selección final de columnas
    final = df[[
        "UserID", "MovieID", "Rating", "GenderBinary", "AgeEncoded", "OccupationEncoded",
        "NumRatingsUser", "AvgRatingUser", "NumRatingsMovie", "AvgRatingMovie",
        "MainGenreEncoded", "GeoCluster"
    ]]

    final.to_csv("data/processed/features_dataset.csv", index=False)
    print("✅ Features guardadas en data/processed/features_dataset.csv")

if __name__ == "__main__":
    build_features()
