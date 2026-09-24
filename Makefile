install:
	python -m pip install -r requirements-dev.txt
test:
	pytest -q
smoke:
	python evals/smoke.py
up:
	docker compose up --build
down:
	docker compose down
