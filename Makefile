format:
	pre-commit run --all-files

test:
	PYTHONPATH=$(CURDIR) pytest tests