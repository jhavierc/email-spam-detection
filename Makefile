PY ?= .venv/bin/python

.PHONY: install data train evaluate notebook test run all

install:        ## create the virtualenv and install pinned dependencies
	python3.12 -m venv .venv && $(PY) -m pip install -r requirements.txt

data:           ## stratified splits + dummy e-mail pool for the web app
	$(PY) -c "import sys; sys.path.insert(0, 'src'); from spam_detection.data_split import make_splits; make_splits()"
	$(PY) scripts/build_email_db.py

train:          ## fine-tune Goodmotion/spam-mail-classifier on the SMS corpus
	$(PY) scripts/train_model.py

evaluate:       ## compare zero-shot vs fine-tuned vs TF-IDF baseline on the test split
	$(PY) scripts/evaluate_model.py

notebook:       ## re-execute the analysis notebook in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/spam_detection.ipynb

test:
	$(PY) -m pytest -q

run:            ## start the web app at http://127.0.0.1:8000
	$(PY) -m app

all: data train evaluate notebook
