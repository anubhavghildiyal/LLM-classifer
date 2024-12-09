release: flask db upgrade
web: gunicorn llm_app.app:create_app\(\) -b 0.0.0.0:$PORT -w 3
