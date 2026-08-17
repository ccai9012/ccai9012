token_utils
===========

This module centralizes how course utilities obtain API credentials. It checks environment variables and the local token file so notebooks can request a named service token without embedding secrets in teaching material.

What this module helps you do
-----------------------------

* look up a service credential through one consistent function
* keep notebook examples free of literal API keys
* optionally prompt for a missing token during an interactive session
* provide clearer errors when a required credential is unavailable

When to use it
--------------

* Another utility needs a DeepSeek, Hugging Face, or Google credential.
* A notebook should work across student machines without machine-specific paths.
* You need credential handling only; this module does not make the external API request itself.

Typical workflow
----------------

#. Store the credential in the supported local configuration or environment variable.
#. Request it by service name with ``get_token``.
#. Pass the returned value directly to the client that needs it.
#. Never print, save, or commit the returned credential.

Functions
---------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.token_utils.get_token
