.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help install install-torch data train cnn evaluate predict serve test lint demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the NumPy-only runtime dependency
	$(PY) -m pip install -r requirements.txt

install-torch: ## Install the optional PyTorch (CPU) backend
	$(PY) -m pip install -r requirements-torch.txt

data: ## Download the MNIST IDX files into data/
	$(PY) -m mnist.data

train: ## Train the NumPy MLP (configurable via ARGS="...")
	$(PY) -m mnist.train --out models/mlp.npz --metrics-out reports/mlp_history.json $(ARGS)

cnn: ## Train the PyTorch CNN (requires `make install-torch`)
	$(PY) -m mnist.torch_cnn --out models/cnn.pt --metrics-out reports/cnn_history.json $(ARGS)

evaluate: ## Evaluate models/mlp.npz on the test set
	$(PY) -m mnist.evaluate --model models/mlp.npz --show-confusion

predict: ## Predict N random test images and save a PNG grid
	$(PY) -m mnist.predict --model models/mlp.npz --sample 16 --save-grid reports/predictions.png

serve: ## Run the web UI at http://127.0.0.1:8000
	$(PY) -m mnist serve --port 8000 $(ARGS)

test: ## Run the unit test suite
	$(PY) -m unittest discover -s tests -t . -v

lint: ## Lint with ruff (pip install ruff)
	ruff check mnist tests

demo: ## Fast end-to-end demo on a subset of the data
	$(PY) examples/demo.py

clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ models reports/*.png
