CCAI9012 API Reference
======================

The CCAI9012 toolkit provides reusable helpers for course exercises in language
models, computer vision, generative AI, data preparation, and visualization. It
is written for students who want to understand the workflow while avoiding
repetitive setup code.

Choose a module by task
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Input or goal
     - Utility module
     - Typical output
   * - Documents and prompts
     - :doc:`ccai9012/llm_utils`
     - Responses, retrieval chains, structured text
   * - Paired or generated images
     - :doc:`ccai9012/gan_utils` and :doc:`ccai9012/sd_utils`
     - Trained generators or generated images
   * - Images and detections
     - :doc:`ccai9012/multi_modal_utils`, :doc:`ccai9012/svi_utils`, and :doc:`ccai9012/yolo_utils`
     - Embeddings, street-view images, and detection summaries
   * - Data, models, and results
     - :doc:`ccai9012/nn_utils` and :doc:`ccai9012/viz_utils`
     - Prepared data, evaluation metrics, plots, and maps
   * - Local credentials
     - :doc:`ccai9012/token_utils`
     - Tokens supplied to another utility module

Typical workflow
----------------

#. Load or prepare the data required by a teaching notebook.
#. Select the utility module that matches the task.
#. Call a public function or class using the documented parameters.
#. Inspect, visualize, or save the returned output.

The module pages explain when each group of helpers is useful. Individual object
pages provide the full signature, parameter definitions, return values, and
source links.

.. toctree::
   :maxdepth: 2
   :caption: API reference

   ccai9012/index
