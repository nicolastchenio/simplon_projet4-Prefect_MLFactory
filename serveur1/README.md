# Guide de Déploiement : API et Orchestration Prefect

Ce projet permet de déployer une API FastAPI orchestrée par Prefect 3 au sein d'une infrastructure Docker.

## 1. Préparation de l'environnement local
L'utilisation de `uv` permet une gestion rapide et isolée des dépendances Python.

Copiez le contenu de `env.example` dans un fichier `.env`

```bash
# Installation des dépendances et synchronisation du projet
uv sync

# Activation de l'environnement virtuel (Windows)
.venv/Scripts/Activate.ps1
```

## 2. Construction et Publication de l'Image
Cette étape prépare l'image Docker contenant tout le code source dans le dossier /app.

```bash
# Construction de l'image Docker
docker build -t devermyst/api-flow:latest .

# Publication sur le registre (Docker Hub)
docker push devermyst/api-flow:latest
```

## 3. Lancement de l'Infrastructure
Démarrage de la base de données PostgreSQL et du serveur Prefect (UI).

```bash
docker-compose up
```
ou
```bash
docker-compose up -d
```

## 4. Déploiement du Flow
Ces commandes : 
* crée le "pool" 
* enregistrent la configuration du job (définie dans le fichier prefect.yaml) sur le serveur Prefect distant
* effectue le déploiement 

```bash
prefect work-pool create "docker-pool" --type docker
prefect config set PREFECT_API_URL="http://localhost:4200/api"
prefect deploy --all
```

## 6. Quick Run et Résultats
On enregistre le déploiement sur le serveur.

```bash
prefect worker start --pool 'docker-pool'
```




