from dotenv import load_dotenv
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import boto3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


load_dotenv()  # charge automatiquement les variables depuis .env
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

def prepare_minio():
    """Vérifie si le bucket mlflow existe sinon le crée"""

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]

    if "mlflow" not in buckets:
        s3.create_bucket(Bucket="mlflow")
        print("Bucket 'mlflow' créé")


def train_and_register():
    mlflow.set_experiment("iris_experiment")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    # model = LogisticRegression(max_iter=200)
    model = RandomForestClassifier()

    with mlflow.start_run():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        model_name = "iris_model"
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

    client = MlflowClient()
    latest_version = client.get_latest_versions(
        model_name,
        stages=["None"]
    )[0].version

    # client.set_registered_model_alias(
    #     model_name,
    #     "production",
    #     latest_version
    # )

    print("Modèle enregistré. Version:", latest_version)


if __name__ == "__main__":
    prepare_minio()
    train_and_register()