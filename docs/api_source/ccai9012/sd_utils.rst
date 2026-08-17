sd_utils
========

This module offers a small interface for text-to-image generation with Stable Diffusion. It keeps credential retrieval, local pipeline setup, hosted inference, and image display behind one client so students can focus on prompt choices and outputs.

What this module helps you do
-----------------------------

* retrieve a Hugging Face credential from the course token mechanism
* initialize either a local diffusion pipeline or hosted inference client
* generate multiple images from one prompt for comparison
* display and return outputs for notebook-based analysis

When to use it
--------------

* You are studying how prompt wording changes generated visual content.
* You need a consistent interface for local and hosted diffusion workflows.
* You understand that model download, inference time, and hosted usage may require external resources.

Typical workflow
----------------

#. Choose a model and local or hosted execution mode.
#. Initialize ``SDClient`` with the required credential and cache settings.
#. Generate a small set of images from a controlled prompt.
#. Compare outputs and record prompt, model, and generation settings.

Related Starter Kits
--------------------

* `Multimodal AI <../../starter_kits/m3_mm.html>`_

Classes
-------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.sd_utils.SDClient

Functions
---------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.sd_utils.get_hf_api_key

Methods
-------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.sd_utils.SDClient.generate_images
