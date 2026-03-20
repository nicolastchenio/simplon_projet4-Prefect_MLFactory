from dotenv import load_dotenv
import os
import argparse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import boto3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def prepare_minio():
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


def train_and_register(model_type, production=False):
    mlflow.set_experiment("iris_experiment")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    if model_type == "logistic":
        model = LogisticRegression(max_iter=200)
    elif model_type == "randomforest":
        model = RandomForestClassifier()
    else:
        raise ValueError("Model type must be 'logistic' or 'randomforest'")

    with mlflow.start_run():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", accuracy)

        model_name = "iris_model"

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

    if production:
        client = MlflowClient()
        latest_version = client.get_latest_versions(
            model_name,
            stages=["None"]
        )[0].version

        client.set_registered_model_alias(
            model_name,
            "production",
            latest_version
        )

        print("Modèle mis en production. Version:", latest_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["logistic", "randomforest"],
        help="Type de modèle à entraîner"
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="Mettre le modèle en production"
    )

    args = parser.parse_args()

    prepare_minio()
    train_and_register(args.model, args.production)