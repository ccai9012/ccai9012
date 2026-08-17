viz_utils
=========

This module contains visualization helpers used across the course, from neural-network diagrams and training curves to review analysis and urban maps. The functions turn intermediate model or data results into figures that can be interpreted and communicated.

What this module helps you do
-----------------------------

* draw simplified model structures and learning curves
* visualize distributions, bias comparisons, keywords, and co-occurrence patterns
* map sampled points, reviews, points of interest, and heat surfaces
* reduce repeated plotting setup while retaining interpretable inputs

When to use it
--------------

* A notebook has computed results but still needs an explanatory figure.
* You want consistent visual conventions across a starter kit or comparison.
* You will choose a plot based on the analytical question rather than merely on available columns.

Typical workflow
----------------

#. Identify the comparison, spatial pattern, or model behavior the figure should communicate.
#. Prepare the function's expected DataFrame, array, model, or coordinates.
#. Generate the plot and inspect labels, scale, and missing values.
#. Add interpretation in the notebook instead of treating the visualization as the conclusion.

Related Starter Kits
--------------------

* `Bias Detection and Interpretability <../../starter_kits/m5_bias.html>`_

Functions
---------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.viz_utils.draw_simple_mlp
   ~ccai9012.viz_utils.plot_loss_curve
   ~ccai9012.viz_utils.plot_pca_words
   ~ccai9012.viz_utils.plot_bar_bias
   ~ccai9012.viz_utils.occupation_comparison
   ~ccai9012.viz_utils.plot_heatmap
   ~ccai9012.viz_utils.plot_points
   ~ccai9012.viz_utils.plot_review_map
   ~ccai9012.viz_utils.plot_review_heatmap
   ~ccai9012.viz_utils.plot_poi_sampled
   ~ccai9012.viz_utils.sample_street_points_map
   ~ccai9012.viz_utils.plot_wordclouds
   ~ccai9012.viz_utils.plot_wordclouds_by_aspect_opinion
   ~ccai9012.viz_utils.plot_star_distribution
   ~ccai9012.viz_utils.viz_keywords_freq
   ~ccai9012.viz_utils.plot_cooccurrence_heatmap
