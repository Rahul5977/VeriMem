.PHONY: setup test lint fmt cache-stats clean-cache smoke data-check

PY := .venv/bin/python
PIP := .venv/bin/pip

setup:            ## create the local (CPU) env
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cp -n .env.example .env || true

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

smoke:            ## GPU box only: P2.8 LoRA smoke test on the committed fixture
	$(PY) -m training.sft --data training/fixtures/sample_trajectories.jsonl --name smoke --smoke

data-check:       ## Mac-safe: build the SFT dataset and print stats, no GPU
	$(PY) -m training.sft --data training/fixtures/sample_trajectories.jsonl \
		--name data-check --mode retrieval --dry-run

cache-stats:
	@find cache -name '*.json' | awk -F/ '{print $$2}' | sort | uniq -c

clean-cache:      ## destroys cached API spend; ask before running
	@echo "This deletes cached LLM/search/fetch results. Ctrl-C to abort."; read _
	rm -rf cache/*/
