# src/front/app.py
import os
import streamlit as st
import requests
import pandas as pd

# ----------------------------
# Configuration de l'API
# ----------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")  # si tu lances via Docker Compose
# API_URL = "http://api:8000"  # si tu lances via Docker Compose
# Pour tester localement depuis Windows, tu peux remplacer par "http://127.0.0.1:8000"

st.set_page_config(page_title="Iris Prediction", layout="centered")

st.title("Prédiction des fleurs Iris avec MLflow")

st.markdown(
    """
    Entrez les mesures d'une fleur Iris ou sélectionnez une ligne du dataset de test.
    """
)

# ----------------------------
# Chemin vers le dataset
# ----------------------------
DATA_PATH = "/app/data/iris_test.csv"

# ----------------------------
# Choix du mode d'entrée
# ----------------------------
mode = st.radio(
    "Mode de test",
    ["Saisie manuelle", "Utiliser une ligne du dataset iris_test.csv"]
)

# ----------------------------
# Mode dataset
# ----------------------------
if mode == "Utiliser une ligne du dataset iris_test.csv":

    try:
        df = pd.read_csv(DATA_PATH)

        st.subheader("Dataset de test")
        st.dataframe(df)

        row_index = st.selectbox("Choisir une ligne du dataset", df.index)

        selected_row = df.loc[row_index]

        st.write("Valeurs sélectionnées :", selected_row)

        sepal_length = float(selected_row["sepal length (cm)"])
        sepal_width = float(selected_row["sepal width (cm)"])
        petal_length = float(selected_row["petal length (cm)"])
        petal_width = float(selected_row["petal width (cm)"])

    except Exception as e:
        st.error(f"Impossible de charger iris_test.csv : {e}")
        st.stop()

# ----------------------------
# Mode manuel
# ----------------------------
else:

    sepal_length = st.number_input("Longueur du sépale (cm)", 0.0, 10.0, 5.1)
    sepal_width = st.number_input("Largeur du sépale (cm)", 0.0, 10.0, 3.5)
    petal_length = st.number_input("Longueur du pétale (cm)", 0.0, 10.0, 1.4)
    petal_width = st.number_input("Largeur du pétale (cm)", 0.0, 10.0, 0.2)

# ----------------------------
# Prédiction
# ----------------------------
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

        prediction = data["prediction"]
        version = data["model_version"]
        probabilities = data["probabilities"]

        class_names = ["Setosa", "Versicolor", "Virginica"]

        st.success(
            f"Prédiction : **{class_names[prediction]}** (version modèle : {version})"
        )

        # ----------------------------
        # Affichage des probabilités
        # ----------------------------
        st.subheader("Probabilités")

        st.write(f"Setosa : {probabilities[0]:.3f}")
        st.write(f"Versicolor : {probabilities[1]:.3f}")
        st.write(f"Virginica : {probabilities[2]:.3f}")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API: {e}")