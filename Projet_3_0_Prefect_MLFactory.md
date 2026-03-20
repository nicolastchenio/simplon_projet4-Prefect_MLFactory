# Architecture ML Distante (Orchestration & Stockage)

### Objectif
Mettre en place une communication entre deux serveurs distincts pour automatiser un cycle d'entraînement ML.

### Architecture
* **Serveur 1 (Orchestration) :** Instance Prefect Server active.
* **Serveur 2 (ML Factory) :** Stack Docker avec MLflow (Tracking) et MinIO (Artifacts).

---

### Étape 1 : Infrastructure ML Factory (Serveur 2)
Reprenez votre stack ML Flow + Minio sans l'entraînement (Projet_2_1_MLFactory)

### Étape 2 : Développement du Flow (Serveur 1)
Reprenez le code de la première démo de prefect puis créez un script `train.py` utilisant `.serve()` pour automatiser la tâche.

**Contraintes techniques :**
1.  **Tâche :** Faites un entraînement, collecter l'`accuracy` n'importe quel modèle et dataset.
2.  **Export :** Configurer `mlflow.set_tracking_uri` pour pointer vers le Serveur 2.
3.  **Planification :** Utiliser `.serve(cron="*/5 * * * *")` pour un déclenchement toutes les 5 minutes.

### Étape 3 : Validation
1.  Vérifier l'apparition du déploiement sur l'UI Prefect (Port 4200).
2.  Vérifier la réception des métriques sur l'UI MLflow (Port 5000) toutes les 5 minutes.

---

### Conclusion
Si le système fonctionne avec `.serve()`, la logique réseau est validée. Le passage à un **véritable déploiement (Worker)** ne demandera aucune modification du code métier, seulement un changement de méthode d'exécution pour s'affranchir du terminal local.
