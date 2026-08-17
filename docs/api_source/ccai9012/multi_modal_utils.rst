multi_modal_utils
=================

This module introduces models that connect images and language. ``CLIPClassifier`` compares an image with candidate text labels, while ``VisionQAProcessor`` asks open-ended questions about image content and can extract keywords from the response.

What this module helps you do
-----------------------------

* perform zero-shot image classification with student-defined text categories
* batch-process an image folder and save comparable confidence scores
* generate captions or answers to questions about images
* extract recurring visual attributes for downstream analysis

When to use it
--------------

* Your categories are semantic descriptions rather than labels from a trained classifier.
* Your research question requires both visual evidence and natural-language interpretation.
* You need an exploratory multimodal baseline and will manually validate model responses.

Typical workflow
----------------

#. Choose CLIP classification or visual question answering based on the research question.
#. Initialize the corresponding class and point it to an image collection.
#. Run a single-image example before starting batch processing.
#. Review confidence scores or generated text before aggregating the results.

Related Starter Kits
--------------------

* `Multimodal AI <../../starter_kits/m3_mm.html>`_

Classes
-------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.multi_modal_utils.CLIPClassifier
   ~ccai9012.multi_modal_utils.VisionQAProcessor

Methods
-------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.multi_modal_utils.CLIPClassifier.classify_image
   ~ccai9012.multi_modal_utils.CLIPClassifier.batch_classify
   ~ccai9012.multi_modal_utils.CLIPClassifier.show_result
   ~ccai9012.multi_modal_utils.VisionQAProcessor.extract_keywords
   ~ccai9012.multi_modal_utils.VisionQAProcessor.generate_caption_for_image
   ~ccai9012.multi_modal_utils.VisionQAProcessor.batch_image_qa
