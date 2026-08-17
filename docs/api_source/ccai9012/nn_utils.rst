nn_utils
========

This module gathers the repeated data preparation, training, and evaluation steps used in the course's introductory neural-network examples. It keeps the training loop available for study while reducing notebook boilerplate.

What this module helps you do
-----------------------------

* split and standardize tabular features before creating PyTorch data loaders
* select an available CPU, CUDA, or Apple Silicon device
* train a supplied model while recording loss history
* evaluate regression and classification predictions with familiar metrics

When to use it
--------------

* You are training a small supervised model on tabular or already-vectorized data.
* Several notebooks need the same split, loader, and evaluation conventions.
* You want to compare model behavior rather than rewrite infrastructure in each exercise.

Typical workflow
----------------

#. Separate features and targets, then call ``prepare_dataloaders``.
#. Define a PyTorch model and choose a device with ``get_best_device``.
#. Train the model with ``train_model``.
#. Use the matching regression or classification evaluator to interpret performance.

Functions
---------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.nn_utils.prepare_dataloaders
   ~ccai9012.nn_utils.get_best_device
   ~ccai9012.nn_utils.train_model
   ~ccai9012.nn_utils.mean_absolute_percentage_error
   ~ccai9012.nn_utils.evaluate_regression_model
   ~ccai9012.nn_utils.evaluate_classification_model
