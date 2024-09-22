sources = src/ scripts/

format:
	uv run ruff format $(sources)

lint:
	uv run ruff check $(sources) --fix --unsafe-fixes

activate:
	source .venv/bin/activate