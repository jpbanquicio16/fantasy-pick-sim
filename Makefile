PY := .venv/bin/python
DRAFTAI := .venv/bin/draftai

.PHONY: install data refresh train project lockin app test test-fast

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:        ## fetch + normalise + validate (cached; past seasons never refetched)
	$(DRAFTAI) ingest

refresh:     ## daily: Sleeper players, injuries, crosswalk
	$(DRAFTAI) ingest --sleeper-only

train:       ## rolling-origin evaluation, test season, final models, report
	$(DRAFTAI) train

project:     ## the draft board for the target season
	$(DRAFTAI) project

lockin:      ## Lock-In headline experiment on the held-out season
	$(DRAFTAI) lockin-experiment

all: data train project lockin

app:
	$(DRAFTAI) app

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow" tests/unit tests/property
