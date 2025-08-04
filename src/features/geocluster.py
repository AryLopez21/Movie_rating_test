import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_elbow(coords, max_k=10):
    """Grafica la inercia para distintos valores de k"""
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=7)
        kmeans.fit(coords)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.title("Método del Codo para KMeans")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()

def cluster_users_by_location(users_path, n_clusters=5):
    df = pd.read_csv(users_path)

    geo_df = df.dropna(subset=["Latitude", "Longitude"]).copy()
    coords = geo_df[["Latitude", "Longitude"]]

    kmeans = KMeans(n_clusters=n_clusters, random_state=7)
    geo_df["GeoCluster"] = kmeans.fit_predict(coords)

    # Visualización rápida
    plt.scatter(coords["Longitude"], coords["Latitude"], c=geo_df["GeoCluster"], cmap="viridis", alpha=0.5)
    plt.title("Clusters geográficos de usuarios")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.show()

    return geo_df
