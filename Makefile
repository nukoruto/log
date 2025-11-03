PYTHON ?= python

.PHONY: lint test e2e all quickstart

lint:
	ruff check .
	black --check .
	mypy packages

test:
	pytest -q

e2e:
	PYTHONPATH=packages/ds_contract/src bash scripts/quickstart.sh

all: lint test e2e
