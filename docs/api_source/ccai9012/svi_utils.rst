svi_utils
=========

This module supports urban computer-vision workflows that begin with Google Street View imagery. It can sample locations, check image availability, download views, and connect those images to segmentation and visualization steps.

What this module helps you do
-----------------------------

* generate a coordinate grid for systematic street-view sampling
* check whether imagery is available before requesting a download
* download individual or batched street-view images with metadata
* run segmentation and visually compare source images with predicted classes

When to use it
--------------

* Your study links urban locations to visible streetscape characteristics.
* You need a repeatable sampling method rather than manually selecting screenshots.
* You have confirmed API terms, quota, credential, and privacy requirements before online collection.

Typical workflow
----------------

#. Define the study extent and generate candidate coordinates.
#. Check availability and download a small validation sample.
#. Run segmentation on the saved images.
#. Inspect source/prediction pairs before summarizing class coverage.

Related Starter Kits
--------------------

* `Computer Vision <../../starter_kits/m4_cv.html>`_

Classes
-------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.svi_utils.GoogleSVIDownloader

Functions
---------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.svi_utils.segment_and_save_images
   ~ccai9012.svi_utils.visualize_segmentation_pair
   ~ccai9012.svi_utils.batch_segment_and_visualize

Constants
---------

.. autodata:: ccai9012.svi_utils.CITYSCAPES_CLASSES

.. autodata:: ccai9012.svi_utils.CITYSCAPES_COLORS

Methods
-------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.svi_utils.GoogleSVIDownloader.is_svi_available
   ~ccai9012.svi_utils.GoogleSVIDownloader.download_svi
   ~ccai9012.svi_utils.GoogleSVIDownloader.generate_grid_coords
   ~ccai9012.svi_utils.GoogleSVIDownloader.download_grid_svis
