wheel:
	pipx run build --wheel .

sdist:
	pipx run build --sdist .

clean:
	rm -rf build dist *.egg-info lightning_ocr/*.egg-info
	pip3 uninstall lightning_ocr -y

install:
	pip3 install -e .

format:
	pre-commit run --all-files

test:
	PYTHONPATH=$(CURDIR) pytest tests/

abinet:
	PYTHONPATH=$(CURDIR) python3 lightning_ocr/models/abinet.py

trocr:
	PYTHONPATH=$(CURDIR) python3 lightning_ocr/models/trocr.py

mgpstr:
	PYTHONPATH=$(CURDIR) python3 lightning_ocr/models/mgp_str.py
