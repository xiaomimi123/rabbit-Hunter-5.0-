.PHONY: build test fetch backtest features shell clean

DC := docker compose -f docker/docker-compose.yml

build:
	$(DC) build

test:
	$(DC) run --rm rabbit pytest

fetch:
	$(DC) run --rm rabbit rabbit fetch

backtest:
	$(DC) run --rm rabbit rabbit backtest

features:
	$(DC) run --rm rabbit rabbit features build

shell:
	$(DC) run --rm rabbit bash

clean:
	rm -rf data/features snapshots reports
