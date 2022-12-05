# Vibration condition monitoring with Deep Learning
Source code for training models for vibration-based monitoring. The models considered so far are Deep Learning-based and unsupervised, listed below:
- Variational Autoencoder (VAE):
  - VAE_Normal: a VAE is trained only with normal samples and is used as a normality model. The reconstruction error is then used as an anomaly indicator.
  - VAE_Damage: a VAE is trained with samples from the damaged condition. The model is then used as a dimensionality reduction tool and the latent variables are used as a condition indicator.

- Bidirectional GAN (BiGAN): under tests

The dataset was previously processed by extracting the signal energy content of the Wavelet Packet decomposition for multiple directions of vibrations measurements.
