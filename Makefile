PYTHON ?= python

.PHONY: lint test e2e all quickstart

lint:
	ruff check .
	black --check .
	mypy packages

test:
	pytest -q

e2e:
	bash scripts/quickstart.sh

all: lint test e2e
