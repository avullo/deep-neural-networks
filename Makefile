init:
	pip install -r requirements.txt


test:
	python -m unittest discover


.PHONY: init test
