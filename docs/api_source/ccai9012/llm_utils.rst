llm_utils
=========

This module provides the reusable building blocks behind the course's language-model examples. It covers direct prompting, repeated generation, review analysis, and retrieval-augmented question answering over PDF documents.

What this module helps you do
-----------------------------

* initialize the supported chat model without placing credentials in a notebook
* send prompts and collect one or several model responses
* turn PDFs into chunks and a searchable vector retriever
* parse structured Markdown tables and analyze review datasets

When to use it
--------------

* You are building a text-generation or structured-output exercise.
* A question should be answered from a supplied PDF rather than from the model's general knowledge.
* You want course-ready helpers while keeping prompt design and result evaluation visible in the notebook.

Typical workflow
----------------

#. Load the credential and initialize the model.
#. Prepare a prompt or build a PDF retriever, depending on the task.
#. Invoke the model or QA chain and inspect the raw response.
#. Parse, compare, or save the result for later evaluation.

Related Starter Kits
--------------------

* `Large Language Models <../../starter_kits/m2_llm.html>`_

Functions
---------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.llm_utils.get_deepseek_api_key
   ~ccai9012.llm_utils.initialize_llm
   ~ccai9012.llm_utils.ask_llm
   ~ccai9012.llm_utils.generate_multiple_outputs
   ~ccai9012.llm_utils.analyze_airbnb_reviews
   ~ccai9012.llm_utils.load_business_locations
   ~ccai9012.llm_utils.load_reviews_by_city
   ~ccai9012.llm_utils.analyze_reviews
   ~ccai9012.llm_utils.build_pdf_retriever
   ~ccai9012.llm_utils.run_qa_chain
   ~ccai9012.llm_utils.parse_markdown_table
