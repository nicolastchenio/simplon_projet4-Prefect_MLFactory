1. Lancer l’infrastructure

Le projet utilise Docker Compose pour démarrer les services nécessaires :

MLflow server

MinIO (stockage des artefacts)

Lancer l’infrastructure :
```
docker-compose up -d --build
```

L’option :

-d lance les conteneurs en arrière-plan

--build reconstruit les images si nécessaire

2. Les scripts sont exécutés avec uv run.

Entrainer un modèle RandomForest
```
uv run python train.py --model randomforest
```
Entrainer un modèle Logistic Regression
```
uv run python train.py --model logistic
```
Entrainer et mettre en production
```
uv run python train.py --model randomforest --production
```

Dans ce cas :
- le modèle est enregistré dans MLflow
- l’alias production est assigné à la dernière version