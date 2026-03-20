FROM prefecthq/prefect:3-python3.11

# On force la création du dossier
RUN mkdir -p /app
WORKDIR /app

# On installe les dépendances
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# On copie tout le contenu du dossier actuel dans /app
COPY . /app

# On s'assure que Python regarde dans /app
ENV PYTHONPATH=/app