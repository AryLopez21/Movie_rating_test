

# Recomendador de Películas - Prueba técnica Banorte (Científico de Datos)

Este proyecto fue desarrollado como parte de una prueba técnica para la vacante de Científico de Datos en el área de Analítica Modelaje en Banorte. Utiliza el dataset MovieLens 1M y está enfocado en predecir la calificación que un usuario podría darle a una película, usando variables creadas, enriquecidas y procesadas profesionalmente.

---

## 🔎 Exploración de Datos (EDA)

## Analisis exploratorio – Hallazgos clave

### 📦 Datos generales

- Dataset utilizado: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- Contiene:
  - Aproximadamente 1 millón de calificaciones
  - 6,000+ usuarios
  - 3,900+ películas
- Cada calificación incluye: `UserID`, `MovieID`, `Rating` (1-5), `Timestamp`
- Usuarios incluyen: `Gender`, `Age`, `Occupation`, `Zip-code`
- Las películas tienen: `Title` y `Genres` (separados por "|")

---

### 👤 Usuarios

- Mayoría de usuarios en los rangos de edad **18-24** y **25-34**.
- Aproximadamente **70% de los usuarios son hombres**.
- Ocupaciones más comunes: estudiante, ingeniero, otro, educador.
- No se detectaron datos faltantes.

---

### 🎬 Películas

- Las películas tienen uno o más géneros (`Action|Comedy|Drama`, etc.).
- Géneros más frecuentes:
  - `Drama`, `Comedy`, `Action`, `Thriller`, `Romance`
- Existe una **distribución sesgada** en número de calificaciones por película (long-tail).
- Algunas películas tienen miles de ratings; otras, muy pocos.
- No hay información de fecha de estreno en el dataset base.

###  Ocupacion

- para el analisis se agregó la descripcion de a que número le corresponde la ocupacion.
---

### ⭐ Calificaciones

- La distribución está **sesgada hacia arriba**:
  - Predominan ratings de **3, 4 y 5 estrellas**.
  - Calificaciones de 1 o 2 estrellas son poco frecuentes.
- Rating promedio general: ~**3.5 estrellas**

---

### 🕒 Análisis temporal

- La mayoría de las calificaciones ocurrieron entre **enero 2000 y junio 2000**.
- El rating promedio se mantiene estable en el tiempo (entre 3.4 y 3.7).
- Se pueden construir features temporales adicionales como:
  - Frecuencia de actividad del usuario

---

### 🧠 Posibles sesgos y patrones detectados

- **Género del usuario**:
  - Las mujeres tienden a calificar ligeramente más alto en promedio.
- **Edad**:
  - Usuarios más jóvenes (`<25`) y mayores (`50+`) califican un poco más alto.
- **Ocupación**:
  - Algunas ocupaciones como des-empleados o granjeros tienden a dar ratings ligeramente más bajos.
- **Género de película vs género de usuario**:
  - Mujeres califican más alto películas de **Romance** y **Drama**
  - Hombres califican más alto géneros como **Action**, **Sci-Fi**, y **War**

---

### 💡 Ideas para Feature Engineering

- Agrupación de edad (`AgeGroup`)
- Popularidad de película (cantidad total de ratings)
- Promedio histórico de calificación por usuario
- Interacciones usuario × género de película
- Variables temporales (año, mes, tiempo desde última calificación)
- Género principal de película (a partir del campo `Genres`)
---

## 🔄 Enriquecimiento de datos

### 📍 API externa: Zippopotam.us

Se consultó la API [Zippopotam.us](https://www.zippopotam.us/) usando el código postal de cada usuario para obtener:
- Ciudad
- Estado
- Coordenadas (latitud, longitud)

Con esto, se construyó una nueva variable geográfica mediante clustering.

### 🔍 Clustering geográfico

- Se aplicó `KMeans` sobre latitud y longitud para generar la variable `GeoCluster`.
- El número de clusters fue elegido mediante el método del codo, justificado visualmente en el notebook `02_clustering_exploratorio.ipynb`.

---

## 🔧 Ingeniería de variables

Archivo: `build_features.py`

Se generaron variables pensadas para aportar información real al modelo, incluyendo:

- **Promedios históricos**:
  - `AvgRatingUser`: promedio de calificaciones por usuario
  - `AvgRatingMovie`: promedio de calificaciones de la película
- **Volumen de interacción**:
  - `NumRatingsUser`, `NumRatingsMovie`
- **Demográficas**:
  - `GenderBinary`, `AgeEncoded`, `OccupationEncoded`
- **Contenido**:
  - `MainGenreEncoded`
- **Geografía**:
  - `GeoCluster`
- **Temporales**:
  - `RatingYear`, `EstimatedAgeAtRating`

---

## 📊 Modelado con XGBoost

Archivo: `train_model.py`

Se entrenó un modelo con `XGBRegressor`, con los siguientes hiperparámetros:

- `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`
- `random_state=7` para reproducibilidad

### Variables utilizadas:

```text
['GenderBinary', 'AgeEncoded', 'OccupationEncoded',
 'NumRatingsUser', 'AvgRatingUser', 'NumRatingsMovie', 'AvgRatingMovie',
 'MainGenreEncoded', 'GeoCluster']
```

---

## 📊 Métricas del modelo

* **RMSE**: `0.9015`
* **MAE**: `0.7143`
* **R²**: `0.3453`
* **CV RMSE (5 folds)**: `0.9118 ± 0.0053`

> El modelo predice con un error promedio de \~0.9 estrellas, y explica el 34.5% de la variabilidad. Es razonable para datos de comportamiento subjetivo como ratings.

---

## 🔍 Análisis visual del modelo

En el notebook `04_modelo_resultados.ipynb` se incluyeron:

### 1. Real vs Predicho

Visualización de predicciones contra calificaciones reales:

* Los puntos deberían estar cerca de la diagonal
* Se detecta una ligera tendencia a subestimar ratings altos

### 2. Distribución de errores

* Centrado en 0
* Sin sesgos extremos

### 3. Importancia de variables

Las más importantes:

* `AvgRatingMovie` (67%)
* `AvgRatingUser` (24%)
* Las demás aportan señal marginal pero complementaria

---

## 📆 Estructura del proyecto

```
.
├── data/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── requirements.txt
└── README.md
```

Todos los scripts se ejecutan con:

```bash
python -m src.<modulo>
```

---

## 💼 Conclusión

Este proyecto demuestra:

* Capacidad técnica en ciencia de datos aplicada
* Dominio de flujo profesional: EDA, features, modelado, validación
* Creatividad para enriquecer con APIs y construir variables personalizadas
* Documentación clara y ejecución reproducible

---   
```
```


## 🚀 Cómo reproducir este proyecto en otra máquina

1. Clona este repositorio:

```bash
git clone https://github.com/tu_usuario/nombre_repositorio.git
cd nombre_repositorio
```

2. Crea un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
```

3. Instala dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecuta los scripts necesarios en orden:

```bash
python -m src.data.load_movielens
python -m src.data.enrich_zipcode
python -m src.features.geocluster
python -m src.features.build_features
python -m src.models.train_model
```

5. Explora los notebooks en `notebooks/` para visualizar el proceso y resultados.

---

## 💼 Conclusión


* El modelo puede ser útil como base para un sistema de recomendación más completo, combinando contenido (géneros), historial de usuario y ubicación.
* El modelo nos ayuda a predecir de forma aceptable la calificación que un usuario podría dar a una película, utilizando únicamente variables estructuradas.
* Durante el desarrollo se concluyó que XGBoost era una excelente opción por su capacidad para manejar variables numéricas y categóricas sin necesidad de codificación one-hot.
* Sería interesante explorar otros enfoques como modelos de ranking, deep learning o sistemas híbridos para comparar desempeño.
* El modelo puede seguir mejorando si se enriquece con otras fuentes de información como sinopsis, reseñas o comportamiento en tiempo real.

---
