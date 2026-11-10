
import argparse
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


# Аргументы командной строки
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, required=True)
args = parser.parse_args()

if not 2 <= args.k <= 7:
    parser.error("--k должен быть от 2 до 7")

# Настройка MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris_kmeans_clean")

# Загрузка данных
iris = load_iris(as_frame=True)
X = iris.data

# Разбиение: 60% / 20% / 20%
X_train, X_temp = train_test_split(
    X, test_size=0.4, random_state=42
)

X_val, X_test = train_test_split(
    X_temp, test_size=0.5, random_state=42
)

# Pipeline предотвращает утечку данных:
# StandardScaler обучается только на X_train
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(
        n_clusters=args.k,
        random_state=42,
        n_init=10
    ))
])

with mlflow.start_run(run_name=f"kmeans_k{args.k}") as run:

    # Обучение
    pipeline.fit(X_train)

    # Валидация
    val_labels = pipeline.predict(X_val)
    X_val_scaled = pipeline.named_steps[
        "scaler"
    ].transform(X_val)

    n_labels = len(set(val_labels))

    if 2 <= n_labels < len(X_val):
        score = silhouette_score(
            X_val_scaled, val_labels
        )
    else:
        score = float("nan")

    # Параметры
    mlflow.log_params({
        "algorithm": "KMeans",
        "n_clusters": args.k,
        "random_state": 42,
        "n_init": 10,
        "train_size": 0.6,
        "val_size": 0.2,
        "test_size": 0.2
    })

    # Метрика
    if pd.notna(score):
        mlflow.log_metric(
            "validation_silhouette", score
        )

    # Теги
    mlflow.set_tags({
        "dataset": "Iris",
        "task": "clustering",
        "split": "train_validation_test"
    })

    # Сохранение модели
    input_example = X_train.head(3)

    signature = mlflow.models.infer_signature(
        X_train,
        pipeline.predict(X_train)
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="iris_pipeline",
        input_example=input_example,
        signature=signature
    )

    # Артефакты
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        predictions = X_val.copy()
        predictions["cluster"] = val_labels

        predictions.to_csv(
            temp_path / "validation_clusters.csv",
            index=False
        )

        with open(
            temp_path / "experiment_info.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump({
                "algorithm": "KMeans",
                "n_clusters": args.k,
                "validation_silhouette": (
                    float(score)
                    if pd.notna(score)
                    else None
                )
            }, file, indent=4)

        mlflow.log_artifacts(
            str(temp_path),
            artifact_path="evaluation"
        )

    print(f"\nКоличество кластеров: {args.k}")
    print(f"Validation Silhouette: {score:.4f}")
    print(f"run_id: {run.info.run_id}")
    print(f"model_uri: {model_info.model_uri}")

print("\nЭксперимент успешно завершён.")
