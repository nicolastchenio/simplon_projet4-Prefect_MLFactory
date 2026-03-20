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
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "production")

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

    probabilities = model._model_impl.predict_proba(input_df)[0].tolist()
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0]), "model_version": state["version"], "probabilities": probabilities}