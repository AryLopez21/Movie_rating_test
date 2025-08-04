import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

RANDOM_STATE = 7

import os
os.makedirs("models", exist_ok=True)

def train_model():
    # Cargar datos
    df = pd.read_csv("data/processed/features_dataset.csv")

    # Definir variables
    target = "Rating"
    drop_cols = ["UserID", "MovieID"]  # IDs no aportan información
    features = [col for col in df.columns if col not in drop_cols + [target]]

    X = df[features]
    y = df[target]

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Modelo
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        objective="reg:squarederror"
    )

    # Entrenar
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluación
    print("📊 Métricas en test:")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    # Validación cruzada
    print("\n🔁 Validación cruzada (5-fold RMSE):")
    rmse_cv = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error"))
    print(f"RMSE mean: {rmse_cv.mean():.4f} | std: {rmse_cv.std():.4f}")

    # Importancia de variables
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\n🔥 Top 10 variables más importantes:")
    print(importance_df.head(10))

    # Guardar modelo
    joblib.dump(model, "models/xgb_model.pkl")
    print("\n✅ Modelo guardado en models/xgb_model.pkl")


if __name__ == "__main__":
    train_model()
