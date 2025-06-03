sources = commands/ primer/ cli.py notebooks/

format:
	uv run ruff format $(sources)

lint:
	uv run ruff check $(sources) --fix --unsafe-fixes

update-submodules:
	git submodule update --init --recursive --remote

fix:
	uv run ruff format $(sources)
	uv run ruff check $(sources) --fix --unsafe-fixes