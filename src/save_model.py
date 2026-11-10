import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Подключение к MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris_kmeans")

# Загрузка данных
iris = load_iris(as_frame=True)
X = iris.data

# Создание Pipeline
best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=2, random_state=42, n_init=10))
])

# Обучение модели
best_pipeline.fit(X)

# Сохранение модели в MLflow
with mlflow.start_run(run_name="best_kmeans_pipeline"):
    mlflow.log_param("n_clusters", 2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_init", 10)

    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        name="iris_pipeline"
    )

print("Лучшая модель успешно сохранена в MLflow.")
