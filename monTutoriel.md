# Phase 1 #
1) Creation de l'arborescence projet Phase 1
```
simplon_projet3_MLFactory
├── data/
│   └── iris_test.csv        # dataset pour tester le modèle
├── src/
│   ├── train/
│   │   └── train.py         # script d’entraînement LogisticRegression
│   ├── api/
│   │   ├── main.py          # FastAPI pour servir le modèle
│   └── front/
│       ├── app.py           # Streamlit interface
├── docker-compose.yml       # infrastructure (MLflow + MinIO + API + front)
├── pyproject.toml           # dépendances Python
└── .env                     # variables d’environnement (MinIO / MLflow)

```

2) creation du docker-compose.yml

```
services:
  # --------------------
  # Stockage objet S3
  # --------------------
  minio:
    image: minio/minio
    container_name: minio
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    command: server /data --console-address ":9001"
    ports:
      - "${MINIO_PORT}:9000"
      - "${MINIO_CONSOLE_PORT}:9001"
    volumes:
      - minio-data:/data

  # --------------------
  # Registre MLflow
  # --------------------
  mlflow:
    image: ghcr.io/mlflow/mlflow
    container_name: mlflow_server
    environment:
      # MLFLOW_S3_ENDPOINT_URL: ${MLFLOW_S3_ENDPOINT_URL}
      MLFLOW_S3_ENDPOINT_URL: "http://minio:${MINIO_PORT:-9000}"
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
      # MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}
    ports:
      - "${MLFLOW_PORT}:5000"
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root s3://mlflow/
      --host 0.0.0.0

    depends_on:
      - minio

  # --------------------
  # API FastAPI
  # --------------------
  api:
    build: ./src/api
    container_name: api
    environment:
      # MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}
      MLFLOW_TRACKING_URI: "http://mlflow:${MLFLOW_PORT:-5000}"
    ports:
      - "${API_PORT}:8000"
    depends_on:
      - mlflow

  # --------------------
  # Front Streamlit
  # --------------------
  front:
    build: ./src/front
    container_name: front
    ports:
      - "${FRONT_PORT}:8501"
    depends_on:
      - api

# --------------------
# Volumes pour persistance
# --------------------
volumes:
  minio-data:
```

✅ Explications

- MinIO :
Stockage persistant dans minio-data
Accessible sur http://localhost:9000 avec login/password du .env

- MLflow :
Pointé sur MinIO (MLFLOW_S3_ENDPOINT_URL) pour stocker les modèles
Accessible sur http://localhost:5000
Dépend de MinIO

- API FastAPI :
Construite depuis src/api/Dockerfile
Communique avec MLflow via MLFLOW_TRACKING_URI

- Streamlit :
Construite depuis src/front/Dockerfile
Communique avec l’API pour récupérer les prédictions

- Volumes :
minio-data pour garder les modèles après redémarrage


3) creation du .env
```
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# ------------------------
# MLflow (Model Registry)
# ------------------------
MLFLOW_TRACKING_URI=http://localhost:5000
# MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
# MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}

# ------------------------
# Ports utilisés
# ------------------------
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
MLFLOW_PORT=5000
API_PORT=8000
FRONT_PORT=8501
```

4) Tu peux maintenant tester l’infrastructure :

ATTENTIONS : pour tester commenter la partie API FastAPI et Front Streamlit sinon cela ne va pas fonctionner

```
docker-compose up -d --build
```
Après cela :
Vérifie MinIO sur http://localhost:9000
Vérifie MLflow sur http://localhost:5000

5) creer les dokerfile :

Pour src/api/Dockerfile :

```
FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn mlflow pandas scikit-learn

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Pour src/front/Dockerfile :
```
FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install streamlit pandas requests

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Décommenter la partie API FastAPI et Front Streamlit sinon cela ne va pas fonctionner
Relancer

```
docker-compose up -d --build
```
voir si les Service	URL fonctionne :
- MinIO API	http://localhost:9000
- MinIO Console	http://localhost:9001
- MLflow	http://localhost:5000
- FastAPI	http://localhost:8000
- Streamlit	http://localhost:8501

6) creer je jeu de donnee "iris_test.csv"
```
uv add scikit-learn pandas mlflow
uv sync
.venv\Scripts\activate
```
puis dans le terminal :
```
python -c "from sklearn.datasets import load_iris;import pandas as pd;iris=load_iris();df=pd.DataFrame(iris.data,columns=iris.feature_names);df['target']=iris.target;df.to_csv('data/iris_test.csv',index=False)"
```

7) train.py
```
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import boto3

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# configuration MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"


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

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("iris_experiment")

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

    client.set_registered_model_alias(
        model_name,
        "Production",
        latest_version
    )

    print("Modèle enregistré. Version:", latest_version)


if __name__ == "__main__":
    prepare_minio()
    train_and_register()
```

Ce que fait ce script
- configure les variables pour accéder à MinIO
- vérifie que le bucket mlflow existe
- charge le dataset Iris
- entraîne une LogisticRegression
- enregistre le modèle dans MLflow
- crée une version dans le Model Registry
- attribue automatiquement l’alias Production

Comment lancer l’entraînement

Depuis la racine du projet :
```
uv run python src/train/train.py
```
Vérification

Ouvrir :
http://localhost:5000

Puis vérifier :
- Experiment : iris_experiment
- Model Registry : iris_model
- Version 1
- Alias : Production

note personnel :
Cas 1 — script lancé sur ta machine (le plus probable)

Si tu lances :
```
uv run python src/train/train.py
```
le script tourne sur ton PC, pas dans Docker.

=> mlflow.set_tracking_uri("http://localhost:5000")


Cas 2 — script lancé dans un conteneur Docker

Si le script s’exécute dans un conteneur du compose, alors Docker crée un réseau interne où les services se voient par leur nom.

Dans ce cas :

mlflow

est résolu automatiquement.

=>  l’URI correcte est : mlflow.set_tracking_uri("http://mlflow:5000")
=> Console MinIO : http://localhost:9001

dans le terminal  => python src/train/train.py


- entraîne une LogisticRegression
- enregistre le modèle dans MLflow
- crée le modèle dans le Model Registry

- attribue l’alias Production


8) main.py
   
```
# main.py
from dotenv import load_dotenv
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----------------------------
# Chargement des variables d'environnement
# ----------------------------
load_dotenv()

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "iris_model")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "Production")

# ----------------------------
# Initialisation MLflow
# ----------------------------
client = MlflowClient(tracking_uri=MLFLOW_URI)

# ----------------------------
# Cache du modèle pour éviter de recharger à chaque requête
# ----------------------------
state = {
    "model": None,
    "version": None
}

def load_production_model():
    """Vérifie la version en production et recharge si nécessaire."""
    try:
        # Récupère la version de l'alias 'Production'
        alias_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        prod_version = alias_info.version

        # Recharge le modèle si ce n'est pas en cache ou si la version a changé
        if state["model"] is None or prod_version != state["version"]:
            print(f"Chargement de la version {prod_version} depuis MLflow...")
            model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            state["model"] = mlflow.pyfunc.load_model(model_uri)
            state["version"] = prod_version

        return state["model"]

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Erreur MLflow: {str(e)}")

# ----------------------------
# Définition de l'API FastAPI
# ----------------------------
app = FastAPI(title="Iris Prediction API")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"message": "API Iris MLflow prête."}

@app.post("/predict")
def predict(input: IrisInput):
    model = load_production_model()
    input_df = pd.DataFrame([[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]], columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0]), "model_version": state["version"]}
```
Pour Tester :
- docker-compose ps
- uvicorn src.api.main:app --reload --port 8000
- curl http://127.0.0.1:8000/
- 
- curl -X POST "http://127.0.0.1:8000/predict" ^
     -H "Content-Type: application/json" ^
     -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"

9) app.py => streamlit
```
# src/front/app.py
import streamlit as st
import requests

# ----------------------------
# Configuration de l'API
# ----------------------------
API_URL = "http://api:8000"  # si tu lances via Docker Compose
# Pour tester localement depuis Windows, tu peux remplacer par "http://127.0.0.1:8000"

st.set_page_config(page_title="Iris Prediction", layout="centered")

st.title("Prédiction des fleurs Iris avec MLflow")

st.markdown(
    """
    Entrez les mesures d'une fleur Iris pour obtenir la prédiction du modèle Logistic Regression.
    """
)

# ----------------------------
# Saisie des caractéristiques
# ----------------------------
sepal_length = st.number_input("Longueur du sépale (cm)", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Largeur du sépale (cm)", 0.0, 10.0, 3.5)
petal_length = st.number_input("Longueur du pétale (cm)", 0.0, 10.0, 1.4)
petal_width = st.number_input("Largeur du pétale (cm)", 0.0, 10.0, 0.2)

if st.button("Prédire"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        response.raise_for_status()
        data = response.json()

        st.success(f"Prédiction: **{data['prediction']}** (version modèle: {data['model_version']})")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API: {e}")
```

Pour lancer streamlit
```
streamlit run src/front/app.py
```