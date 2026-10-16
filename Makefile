PY := .venv/bin/python
DRAFTAI := .venv/bin/draftai

.PHONY: install data refresh train project lockin opponents backtest mock app test test-fast

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

opponents:   ## fit the opponent model on your real Sleeper drafts (username in configs/league.yaml)
	$(DRAFTAI) opponents

backtest:    ## paired draft backtest on past seasons + replay of your real drafts (~15 min)
	$(DRAFTAI) backtest

mock:        ## offline mock in the terminal; the other teams draft themselves
	$(DRAFTAI) mock

all: data train project lockin opponents backtest

app:
	$(DRAFTAI) app

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow" tests/unit tests/property
