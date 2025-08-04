import os
import time
import requests
import pandas as pd
from tqdm import tqdm

ZIP_API_URL = "http://api.zippopotam.us/us/"


def get_zip_info(zip_code):
    try:
        response = requests.get(ZIP_API_URL + str(zip_code), timeout=5)
        if response.status_code == 200:
            data = response.json()
            place = data["places"][0]
            return {
                "Zip-code": zip_code,
                "City": place["place name"],
                "State": place["state"],
                "Latitude": float(place["latitude"]),
                "Longitude": float(place["longitude"]),
            }
    except:
        pass
    return {
        "Zip-code": zip_code,
        "City": None,
        "State": None,
        "Latitude": None,
        "Longitude": None,
    }


def enrich_zip_codes(users_df, delay=0.3):
    zip_codes = users_df["Zip-code"].unique()
    enriched = []

    print(f"Consultando {len(zip_codes)} códigos postales únicos...")

    for zip_code in tqdm(zip_codes):
        info = get_zip_info(zip_code)
        enriched.append(info)
        time.sleep(delay)

    return pd.DataFrame(enriched)


def main():
    users_path = "data/processed/users.csv"
    output_path = "data/processed/users_enriched.csv"

    users_df = pd.read_csv(users_path)
    enriched_zip_df = enrich_zip_codes(users_df)
    enriched_users = users_df.merge(enriched_zip_df, on="Zip-code", how="left")
    enriched_users.to_csv(output_path, index=False)

    print(f"✅ Enriquecimiento guardado en {output_path}")


if __name__ == "__main__":
    main()
