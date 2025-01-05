format:
	pre-commit run --all-files

test:
	PYTHONPATH=$(CURDIR) pytest tests

abinet:
	PYTHONPATH=$(CURDIR) python3 lightning_ocr/models/abinet.py
