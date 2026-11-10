
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict


app = FastAPI(
    title="Iris Clustering API",
    description="API для кластеризации цветков ириса",
    version="1.0.0"
)

# Путь к экспортированной модели
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"

# Загружаем Pipeline: StandardScaler + KMeans
model = joblib.load(MODEL_PATH)


# Структура входных данных
class IrisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: IrisInput):
    features = pd.DataFrame([{
        "sepal length (cm)": data.sepal_length,
        "sepal width (cm)": data.sepal_width,
        "petal length (cm)": data.petal_length,
        "petal width (cm)": data.petal_width
    }])

    cluster = model.predict(features)[0]

    return {
        "cluster": int(cluster)
    }
