PYTHON ?= python3
.PHONY: install test demo serve
install:
	$(PYTHON) -m pip install -e .
test:
	$(PYTHON) -m pytest -q
demo:
	PYTHONPATH=src $(PYTHON) -m roundstable demo
serve:
	PYTHONPATH=src $(PYTHON) -m roundstable serve
