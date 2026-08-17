gan_utils
=========

This module supports a complete paired image-to-image translation exercise using a compact Pix2Pix-style GAN. It connects the steps that students often see separately: organizing aligned images, loading them as tensors, defining the generator and discriminator, training, and inference.

What this module helps you do
-----------------------------

* turn region-based Source/Target folders into reproducible train and test splits
* apply synchronized augmentation so an input image stays aligned with its target
* train a teaching-scale U-Net generator and PatchGAN discriminator
* load checkpoints and generate output images for a test folder

When to use it
--------------

* You have paired images showing the same place or object in two domains, such as map-to-aerial, facade-to-segmentation, or sketch-to-image.
* You want to study the mechanics of adversarial and reconstruction losses before adopting a larger GAN framework.
* You need a transparent baseline for a course project; it is not intended as a production-scale GAN library.

Typical workflow
----------------

#. Prepare aligned image pairs with ``prepare_gan_dataset``.
#. Create batches with ``create_paired_data_loader``.
#. Initialize ``UNetGenerator`` and ``PatchDiscriminator``.
#. Train with ``train_GAN``, then use ``load_model`` and ``inference_gan`` for prediction.

Related Starter Kits
--------------------

* `Generative AI: GANs <../../starter_kits/m1_gan.html>`_

Classes
-------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.gan_utils.PairedImageDataset
   ~ccai9012.gan_utils.UNetGenerator
   ~ccai9012.gan_utils.PatchDiscriminator

Functions
---------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.gan_utils.setup_directories
   ~ccai9012.gan_utils.collect_image_pairs
   ~ccai9012.gan_utils.split_pairs
   ~ccai9012.gan_utils.process_and_save_image
   ~ccai9012.gan_utils.copy_pair
   ~ccai9012.gan_utils.prepare_gan_dataset
   ~ccai9012.gan_utils.augment_pair
   ~ccai9012.gan_utils.create_paired_data_loader
   ~ccai9012.gan_utils.train_GAN
   ~ccai9012.gan_utils.tensor2img
   ~ccai9012.gan_utils.inference_gan
   ~ccai9012.gan_utils.load_model

Methods
-------

.. autosummary::
   :toctree: ../generated

   ~ccai9012.gan_utils.UNetGenerator.forward
   ~ccai9012.gan_utils.PatchDiscriminator.forward
