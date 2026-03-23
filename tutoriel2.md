1. Préparation de l'environnement
Vous devez ajouter Prefect à votre projet. Comme vous utilisez uv, lancez :
```
uv add prefect
```

2. Adaptation de l'Infrastructure (Serveur 2)
Le "Serveur 2" est votre stack Docker actuelle. Gardez MinIO et MLflow actifs via votre docker-compose.yml.
* Assurez-vous qu'ils tournent bien : docker-compose up -d minio mlflow.
* L'API et le Front ne sont pas strictement nécessaires pour l'étape Prefect, mais vous pouvez les laisser pour valider la prédiction finale.

3. Transformation de src/train/train.py en Prefect Flow
Vous devez modifier votre script d'entraînement pour qu'il soit reconnu par Prefect. Voici la structure à adopter :

* Décorateurs : Utilisez @task pour les étapes (préparation MinIO, entraînement) et @flow pour orchestrer le tout.
* Tracking Distant : Gardez mlflow.set_tracking_uri("http://localhost:5000") (ou l'IP de votre serveur 2 si c'est une autre machine).
* Automatisation : Utilisez la méthode .serve() à la fin du script.

4. Lancement de l'Orchestration (Serveur 1)
   1. Démarrer Prefect : Dans un nouveau terminal, lancez le serveur Prefect pour accéder à l'interface :
      prefect server start
      (Accessible sur http://localhost:4200)
   2. Déployer le Flow : Lancez votre script train.py modifié. Il restera en attente et enverra des instructions au serveur Prefect toutes les 5 minutes grâce au paramètre cron.

# train\train.py #

modification du fichier train.py

```
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import boto3

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from prefect import task, flow

# ---------------------------------------------------------
# Configuration de l'infrastructure (Serveur 2 - ML Factory)
# ---------------------------------------------------------
# Note : Si MLflow/MinIO sont sur une autre machine, remplacez 'localhost' par l'IP du serveur.
MLFLOW_TRACKING_URI = "http://localhost:5000"
MINIO_ENDPOINT_URL = "http://localhost:9000"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

@task(name="Préparation MinIO", description="Vérifie ou crée le bucket 'mlflow' dans MinIO")
def prepare_minio():
    """Vérifie si le bucket mlflow existe sinon le crée"""
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )
    try:
        buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
        if "mlflow" not in buckets:
            s3.create_bucket(Bucket="mlflow")
            print("✅ Bucket 'mlflow' créé avec succès.")
        else:
            print("ℹ️ Bucket 'mlflow' déjà existant.")
    except Exception as e:
        print(f"❌ Erreur lors de la connexion à MinIO : {e}")
        raise

@task(name="Entraînement et Enregistrement", description="Entraîne le modèle Iris et l'enregistre dans MLflow")
def train_and_register():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("iris_experiment_prefect")

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=200)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)

        model_name = "iris_model_prefect"

        # Log du modèle dans MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        print(f"📊 Accuracy du modèle : {accuracy:.4f}")

    # Gestion du Model Registry (Alias Production)
    client = MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

    client.set_registered_model_alias(
        model_name,
        "Production",
        latest_version
    )
    print(f"🚀 Modèle enregistré. Version: {latest_version} avec alias 'Production'")

@flow(name="Iris ML Pipeline", description="Pipeline d'entraînement automatisé pour le projet ML Factory")
def iris_training_flow():
    """Orchestration des tâches d'entraînement"""
    prepare_minio()
    train_and_register()

if __name__ == "__main__":
    # Déploiement via Prefect .serve()
    # Le script restera actif et lancera le flow toutes les 5 minutes
    print("🛰️ Lancement du déploiement Prefect (Cron: toutes les 5 min)...")
    iris_training_flow.serve(
        name="iris-training-deployment",
        cron="*/5 * * * *",
        description="Déploiement d'entraînement Iris automatique toutes les 5 minutes."
    )
```

# a faire #

Le fichier src/train/train.py est maintenant prêt pour l'orchestration Prefect.

  Voici les étapes à suivre pour valider votre installation (Étape 3 du projet) :

  1. Lancez l'infrastructure (Serveur 2)
  Assurez-vous que vos conteneurs MLflow et MinIO sont bien démarrés :
  ```
  docker-compose up -d minio mlflow
  ```
  - MinIO accessible sur http://localhost:9000 avec login/password du .env
  - MLflow accessible sur http://localhost:5000

  2. Démarrez le serveur Prefect (Serveur 1)
  Ouvrez un nouveau terminal et lancez le serveur Prefect pour accéder au tableau de bord :
  ```
  uv run prefect server start
  ```
  L'interface sera accessible sur http://localhost:4200.

  1. Lancez le script d'entraînement
  Dans votre terminal principal, exécutez le script. Il va s'enregistrer auprès du serveur Prefect et rester en attente :
  ```
  uv run python src/train/train.py
  ```

  1. Validation
   2. Sur Prefect (Port 4200) : Allez dans l'onglet "Deployments". Vous devriez voir Iris ML Pipeline/iris-training-deployment. Cliquez dessus pour voir les prochaines exécutions prévues (toutes les 5 minutes).
   3. Sur MLflow (Port 5000) : Après la première exécution (immédiate ou après 5 min), vérifiez l'expérience iris_experiment_prefect. Vous devriez y voir vos métriques d'accuracy et votre modèle enregistré sous iris_model_prefect avec l'alias Production.

  Note : Pour que l'API et le Front fonctionnent avec ce nouveau modèle, n'oubliez pas de mettre à jour la variable d'environnement MODEL_NAME dans votre
  fichier .env pour qu'elle pointe vers iris_model_prefect au lieu de iris_model.