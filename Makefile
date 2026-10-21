PY := .venv/bin/python
DRAFTAI := .venv/bin/draftai

.PHONY: install data refresh train rookies rates project lockin opponents ltr ltr-report backtest mock app api web test test-fast

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:        ## fetch + normalise + validate (cached; past seasons never refetched)
	$(DRAFTAI) ingest

refresh:     ## daily: Sleeper players, injuries, crosswalk
	$(DRAFTAI) ingest --sleeper-only

train:       ## rolling-origin evaluation, test season, final models, report
	$(DRAFTAI) train

rookies:     ## refit and evaluate only the rookie model (V5)
	$(DRAFTAI) rookies

rates:       ## compare the per-minute rate priors on the validation seasons (V5)
	$(DRAFTAI) rates-experiment

project:     ## the draft board for the target season
	$(DRAFTAI) project

lockin:      ## Lock-In headline experiment on the held-out season
	$(DRAFTAI) lockin-experiment

opponents:   ## fit the opponent model on your real Sleeper drafts (username in configs/league.yaml)
	$(DRAFTAI) opponents

ltr:         ## learning-to-rank (V5): collect labelled queries (~30 min, cached), train, judge offline
	$(DRAFTAI) ltr --no-backtest

backtest:    ## paired draft backtest on past seasons + replay of your real drafts (~20 min)
	$(DRAFTAI) backtest

ltr-report:  ## the learning-to-rank report, judged by the last backtest
	$(DRAFTAI) ltr --use-backtest-rows

mock:        ## offline mock in the terminal; the other teams draft themselves
	$(DRAFTAI) mock

all: data train project lockin opponents ltr backtest ltr-report

app:         ## Streamlit app
	$(DRAFTAI) app

web:         ## build the React front end into web/dist (served by `draftai api`)
	cd web && npm ci && npm run build

api:         ## FastAPI + the React front end on http://127.0.0.1:8000
	$(DRAFTAI) api

test:
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -m "not slow" tests/unit tests/property
