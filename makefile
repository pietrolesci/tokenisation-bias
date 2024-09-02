sources = src/ scripts/ notebooks/

format:
	uv run ruff format $(sources)

lint:
	uv run ruff check $(sources) --fix

activate:
	source .venv/bin/activate