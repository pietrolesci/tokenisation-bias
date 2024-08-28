sources = src/ scripts/

format:
	ruff format $(sources)

lint:
	ruff check $(sources) --fix

activate:
	source .venv/bin/activate