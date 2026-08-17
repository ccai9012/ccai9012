yolo_utils
==========

This module supports an object-detection workflow based on YOLO predictions over images or video. It helps students move from frame-level detections to saved records and an annotated video that can be inspected or summarized.

What this module helps you do
-----------------------------

* run object detection and tracking over a video source
* record detected classes, confidence values, and frame information
* apply lightweight smoothing to reduce unstable frame-to-frame results
* render an annotated output video for qualitative review

When to use it
--------------

* Your project asks what objects appear in a video and how detections change over time.
* You need both a tabular record for analysis and a visual output for validation.
* You will inspect false positives and missed detections before using counts as evidence.

Typical workflow
----------------

#. Select a YOLO model, video source, and classes relevant to the question.
#. Run ``detect_and_track`` and save the frame-level results.
#. Review confidence thresholds and temporal consistency.
#. Create an annotated video with ``visualize_video`` and compare it with the table.

Related Starter Kits
--------------------

* `Computer Vision <../../starter_kits/m4_cv.html>`_

Functions
---------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.yolo_utils.detect_and_track
   ~ccai9012.yolo_utils.visualize_video
