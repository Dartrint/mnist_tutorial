"""mnist — a small, dependency-light MNIST digit-classification toolkit.

The package implements the full pipeline in plain NumPy:

* :mod:`mnist.data`     — download + parse the IDX files (cached on disk)
* :mod:`mnist.nn`       — layers, losses and optimizers built from scratch
* :mod:`mnist.model`    — a configurable multi-layer perceptron (MLP)
* :mod:`mnist.train`    — mini-batch training loop + CLI
* :mod:`mnist.evaluate` — accuracy, confusion matrix and per-class metrics
* :mod:`mnist.predict`  — run inference on images and render a prediction grid
"""

__version__ = "0.1.0"

__all__ = ["data", "nn", "model", "train", "evaluate", "predict", "utils"]
