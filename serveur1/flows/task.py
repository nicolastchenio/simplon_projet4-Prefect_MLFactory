import time
from prefect import flow, task as prefect_task


@prefect_task(name="Calcul de précision", retries=2)
def logic_calcul(a: int, b: int):
    """Une tâche Prefect avec gestion d'erreurs et retries."""
    print(f"Exécution d'un calcul complexe pour {a} et {b}...")
    time.sleep(2)
    return a + b

@flow(name="Flow de Calcul Automatique")
def mon_flow_calcul(a: int = 10, b: int = 20):
    """Le flow qui sera appelé par le Worker Prefect."""
    print("Démarrage du flow de calcul...")
   
    resultat = logic_calcul(a, b)
    
    print(f"Flow terminé. Résultat sauvegardé : {resultat}")
    return {"statut": "success", "valeur": resultat}