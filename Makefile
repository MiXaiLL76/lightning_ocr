format:
	pre-commit run --all-files

test:
	PYTHONPATH=$(CURDIR) pytest tests

abinet:
	PYTHONPATH=$(CURDIR) python3 lightning_ocr/models/abinet.py

trocr:
	PYTHONPATH=$(CURDIR) python3 lightning_ocr/models/trocr.py

mgpstr:
	PYTHONPATH=$(CURDIR) python3 lightning_ocr/models/mgp-str.py
