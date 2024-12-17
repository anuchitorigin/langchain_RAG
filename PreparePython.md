## Python Project Preparation

1. To create virtual environment (only once), use this command:
    ```bash
    python3 -m venv venv
    ```
2. To activate virtual environment, use this command:
    ```bash
    source venv/bin/activate
    ```
3. To install all dependencies in the current virtual environment, use this command:
    ```bash
    pip install -r requirements.txt
    ```
    OR
    ```bash
    pip install <your-dependencies>
    ```
4. To create requirements.txt file, use this command:
    ```bash
    pip freeze > requirements.txt
    ```
5. To deactivate virtual environment, use this command:
    ```bash
    deactivate
    ```
6. To run the project with ASGI server, go to main file location (e.g. /app/main.py) and use this command:
    ```bash
    uvicorn main:app --reload
    ```
7. To run Celery module, go to root directory (e.g. /app) and use this command:
    ```bash
    celery -A app.celerymain worker -c 1 -l DEBUG
    ```
    OR
    ```bash
    celery --app=app.celerymain worker --concurrency=4 --loglevel=INFO
    ```
8. To run Flower module, go to root directory (e.g. /app) and use this command:
    ```bash
    celery --broker=redis://localhost:6379/0 flower --port=5555
    ```
9. If Pylance reports an error like "reportMissingImports", you can add this line to your `settings.json` file:
    ```json
    "python.defaultInterpreterPath": "./<your-project-folder>/venv/bin/python3"
    ```
    OR
    ```json
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python3"
    ```
    OR
    ```json
    "python.analysis.extraPaths": ["./app"]
    ```
