Adversarial Robustness Study on CIFAR-10

Objective:
Apply a deep learning model to CIFAR-10 and modify the input data such that the model's performance degrades statistically, in order to study adversarial vulnerability in image classifiers.

Setup:
Dataset: CIFAR-10, standard 50,000/10,000 train/test split, evaluated on the test set
Normalization: per-channel mean (0.4914, 0.4822, 0.4465), std (0.2470, 0.2435, 0.2616)
Augmentation (training only): random crop (32, padding 4), random horizontal flip
Model: ResNet18 adapted for CIFAR-sized inputs (3×3 stride-1 first conv, no initial maxpool). The standard torchvision ResNet18 is designed for 224×224 ImageNet images; its 7×7 stride-2 first conv plus immediate maxpool discards most spatial information on 32×32 inputs if used unmodified, which was verified to be the cause of an earlier, substantially lower baseline (~73-79%) before this fix.
Training: Adam optimizer, learning rate 0.001, batch size 128, 20 epochs (undefended and FGSM-defended models)
Hardware: Google Colab, T4 GPU
Library: PyTorch / torchvision

Threat Model:
White-box attack: the attacker has full access to model gradients. This is the standard setting for FGSM and PGD.

Attack Methods
FGSM (Goodfellow et al., 2015): single-step perturbation in the direction of the loss gradient, epsilon = 0.03, applied in normalized pixel space. Epsilon (0.03) closely matches the common literature convention of ε = 8/255 ≈ 0.0314 used in CIFAR-10 adversarial robustness benchmarks, differing by under 5%, so results should be broadly comparable to published work using that standard.
PGD (Madry et al., 2018): 10-step iterative version of FGSM with random start within the epsilon ball, step size (alpha) = 0.007. Both attacks clamp perturbed images back to the valid pixel range after each step, following standard practice.

Defense Methods:
Two adversarial training variants were compared:
FGSM-AT: FGSM-perturbed images mixed into training batches alongside clean images, 20 epochs.
PGD-AT: PGD-perturbed images (7 iterations during training, for computational tractability) mixed into training batches, 12 epochs (fewer than FGSM-AT due to the substantially higher per-step cost of iterative attack generation).

Results

Model comparison:

| Model | Baseline | FGSM | PGD |
|---|---|---|---|
| Undefended | 89.68% | 26.80% | 8.57% |
| FGSM-AT (20 epochs) | 89.37% | 73.97% | 72.54% |
| PGD-AT (12 epochs) | 85.65% | 69.22% | 68.35% |

Multi-seed variance (undefended model, 2 seeds):

| Metric | Mean | Std |
|--- |---|---|
| Baseline | 89.77% | 0.42 |
| FGSM | 24.87% | 0.08 |
| PGD | 8.17% | 0.82 |

Per-class accuracy under PGD attack:

| Model | Weakest classes | Strongest classes |
|---|---|---|
| Undefended | dog (1.7%), deer (2.0%), cat (~3%) | horse (18.0%), truck (14.2%), ship (11.7%) |
| FGSM-AT | cat (54.4%), dog (57.5%) | truck (87.9%), car (83.5%), frog (81.1%) |

Below: a cat image, correctly classified originally, misclassified as "bird" after an imperceptible FGSM perturbation. The saliency map shows the model's attention spread diffusely across the image rather than concentrated on the cat itself.

Analysis
The properly-adapted, well-trained classifier (89.68% clean accuracy) loses most of its accuracy under imperceptible perturbations: FGSM reduces it to 26.80%, and PGD reduces it further to 8.57%. This result is stable across random seeds — a repeated run with different initialization seeds produced a mean baseline of 89.77% (± 0.42) and mean PGD accuracy of 8.17% (± 0.82), confirming the original run was not an outlier.

The saliency map for one example image showed attention spread diffusely across the image rather than concentrated on class-relevant regions; based on this small number of qualitative examples, this may partly explain the model's sensitivity to small perturbations, though a broader sample would be needed to generalize this claim.

Per-class breakdown reveals the vulnerability is not uniform. Under the undefended model, "dog," "deer," and "cat" are almost completely broken by PGD (1.7–3% accuracy), while "horse" and "truck" retain some resistance (14–18%) — likely reflecting how visually distinct these classes are from their nearest confusable neighbors in the undefended model's decision space. After FGSM-AT, this pattern persists in relative terms: "cat" and "dog" remain the weakest classes (54–58%) while "truck," "car," and "frog" remain strongest (81–88%), suggesting some classes are intrinsically harder to make robust, independent of the defense applied.

Adversarial training substantially improved robustness. FGSM-AT retained 73.97% accuracy under FGSM attack and 72.54% under PGD attack (versus 26.80%/8.57% undefended), with clean accuracy nearly unchanged (89.37% vs. 89.68%, a difference of well under one point).

PGD-AT, despite being the theoretically stronger defense (Madry et al., 2018), underperformed FGSM-AT in this experiment — 85.65% clean / 69.22% FGSM-robust / 68.35% PGD-robust, compared to FGSM-AT's 89.37%/73.97%/72.54%. This is likely explained by an experimental confound rather than a genuine finding that FGSM-AT is superior: PGD-AT was trained for only 12 epochs (versus 20 for FGSM-AT) due to its substantially higher computational cost (each training step requires 7 forward/backward passes for adversarial example generation, versus 1 for FGSM), meaning the PGD-AT model is likely under-trained relative to its counterpart rather than genuinely less effective. A fair comparison would require matching epoch counts, which was not feasible within the available compute budget for this task.

To rule out gradient masking (Athalye et al., 2018) in the FGSM-AT model — a known failure mode where adversarially-trained models appear robust only because gradient-based attacks fail to find real adversarial directions, not because the model is genuinely robust — two sanity checks were performed: (1) increasing PGD from 10 to 40 steps, which produced negligible change (72.54%→72.25%), and (2) a transfer attack using adversarial examples generated on the undefended model, which yielded 87.38% accuracy on the defended model — higher than the direct white-box PGD result. Both results are inconsistent with gradient masking and support that the observed robustness gain is genuine.

This robustness level exceeds typical reported results for FGSM-only adversarial training in the literature; possible contributing factors include the specific epsilon/alpha configuration or dataset-specific effects, which would be worth further investigation.

Limitations
- PGD-AT and FGSM-AT were trained for different epoch counts (12 vs. 20) due to compute constraints, making their direct comparison indicative rather than definitive; a fair comparison would require matched training budgets.
- Multi-seed variance was evaluated for the undefended model only (2 seeds), not for the defended models, due to time constraints.
- Saliency analysis was qualitative and based on a small number of examples, not a systematic study.

Conclusion
Standard image classifiers, even when properly architected and well-trained, are highly vulnerable to small, imperceptible, gradient-based perturbations, and this vulnerability is not uniform across classes. Adversarial training substantially improves robustness; in this experiment, FGSM-based training was more effective than PGD-based training, though this likely reflects an unequal training budget rather than a genuine superiority of the weaker attack. Robustness gains were verified as genuine rather than an artifact of gradient masking, and results were confirmed stable across random seeds. This work directly relates to my broader research interest in adversarial robustness for security-critical ML systems such as fraud detection and network intrusion detection, where similar vulnerabilities could be exploited by an adversary to evade detection.

References
- Athalye, A., Carlini, N., & Wagner, D. (2018). Obfuscated Gradients Give a False Sense of Security.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples.
- Madry, A., et al. (2018). Towards Deep Learning Models Resistant to Adversarial Attacks.
