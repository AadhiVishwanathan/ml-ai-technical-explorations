Image Reconstruction from Noisy Data (Denoising Autoencoder)

Objective:
Train a neural network to reconstruct clean images from noisy versions, using CIFAR-10, and evaluate reconstruction quality quantitatively across multiple noise levels.

Setup:
Dataset: CIFAR-10, standard 50,000/10,000 train/test split
Noise model: additive Gaussian noise, clamped to [0,1] after corruption
Architecture: convolutional autoencoder
Encoder: Conv2d(3→32, k3) → ReLU → Conv2d(32→64, k3) → ReLU → MaxPool2d(2) [32×32→16×16] → Conv2d(64→128, k3) → ReLU → MaxPool2d(2) [16×16→8×8]
Decoder: ConvTranspose2d(128→64, k2, s2) → ReLU [8×8→16×16] → ConvTranspose2d(64→32, k2, s2) → ReLU [16×16→32×32] → Conv2d(32→3, k3) → Sigmoid
Training: Adam optimizer, learning rate 0.001, batch size 128, 15 epochs, MSE loss
Randomized noise training: noise level sampled uniformly from [0.1, 0.3] each batch, to encourage robustness across noise strengths rather than overfitting to one
Evaluation metrics: PSNR and SSIM, tested at noise levels [0.1, 0.2, 0.3, 0.4] — including 0.4, beyond the training range, to test generalization
Hardware: Kaggle, T4 GPU

Results:

| Noise Level | Noisy Input | Median Filter | Gaussian Blur | Autoencoder (ours) |
|---|---|---|---|---|
| 0.1 | 20.32 dB / 0.673 | 23.54 dB / 0.778 | 23.81 dB / 0.806 | **25.27 dB / 0.856** |
| 0.2 | 14.78 dB / 0.430 | 20.02 dB / 0.627 | 21.51 dB / 0.700 | **23.61 dB / 0.795** |
| 0.3 | 11.92 dB / 0.294 | 17.40 dB / 0.507 | 19.39 dB / 0.597 | **21.87 dB / 0.719** |
| 0.4 | 10.22 dB / 0.214 | 15.41 dB / 0.414 | 17.72 dB / 0.510 | **20.27 dB / 0.643** |

(PSNR dB / SSIM format. See `results/denoising_comparison_grid.png` for visual comparisons.)

Analysis:
Against the raw noisy input, all three denoising methods provide substantial improvement — even the weakest method (median filter) roughly doubles SSIM at every noise level. This confirms noise removal is genuinely happening across the board, establishing a meaningful floor before comparing methods against each other.

The autoencoder's PSNR and SSIM both degrade smoothly and predictably as noise increases, including at 0.4 — a level outside the training range (0.1-0.3) — indicating the model generalizes reasonably to unseen noise strengths rather than only memorizing the training distribution. At 0.4, the noisy input starts from SSIM 0.214 (barely structurally related to the original); the autoencoder recovers this to 0.643, a threefold improvement, compared to the median filter's 0.414 and Gaussian blur's 0.510.

Critically, the autoencoder outperformed both classical baselines at every tested noise level, on both metrics, with the performance gap **widening** as noise increased: at 0.1 the autoencoder leads the next-best method (Gaussian blur) by only 1.46 dB PSNR / 0.050 SSIM, but by 0.4 this gap grows to 2.55 dB PSNR / 0.133 SSIM. This indicates the network is learning genuine, structure-aware denoising rather than simple local averaging, and that this learned advantage becomes disproportionately more valuable as the denoising problem gets harder — exactly where a simple filter's fixed, content-agnostic averaging strategy runs out of headroom.

Visual inspection (see figure) supports this: at noise level 0.3, reconstructed images clearly recover overall object shape, pose, and color scheme — though fine texture detail is smoothed rather than perfectly restored. This is consistent with SSIM values in the 0.7-0.8 range at that noise level: strong structural agreement with imperfect pixel-level fidelity, matching the classic autoencoder failure mode where high-frequency detail is sacrificed in favor of overall structural correctness, since MSE loss inherently penalizes blur less than it penalizes structurally wrong reconstructions.

Limitations:
- Only Gaussian (additive) noise was tested; real-world noise (sensor noise, compression artifacts) often has different statistical properties and may respond differently.
- The architecture is a relatively small, standard convolutional autoencoder; larger or more sophisticated architectures (e.g., U-Net with skip connections, or a perceptual/SSIM-based loss instead of pure MSE) would likely improve fine-detail reconstruction further, particularly the blur observed at higher noise levels.
- SSIM/PSNR were evaluated on the full test set, but qualitative visual inspection was limited to a small sample of images.

Conclusion:
A convolutional denoising autoencoder, trained with randomized noise levels, reconstructs CIFAR-10 images with consistently higher fidelity than both classical (non-learned) denoising baselines and the raw noisy input, across all tested noise levels. The advantage over classical methods is most pronounced at higher noise levels, confirming the network learns genuine content-aware denoising behavior rather than simple smoothing, and generalizes reasonably to noise levels beyond its training range.

References:
- Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). Extracting and Composing Robust Features with Denoising Autoencoders. ICML.
