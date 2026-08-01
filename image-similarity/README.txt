Learning Image Similarity via Triplet Loss with Semi-Hard Negative Mining

Objective:
Train a neural network to embed images such that visually/semantically similar images (same class) map close together in embedding space, and dissimilar images map far apart — then evaluate this using nearest-neighbor retrieval, compared against a raw-pixel-distance baseline.

Setup:
Dataset: CIFAR-10, standard 50,000/10,000 train/test split
Architecture: CNN embedding network — Conv2d(3→32,k3)→ReLU→MaxPool(2) → Conv2d(32→64,k3)→ReLU→MaxPool(2) → Conv2d(64→128,k3)→ReLU→MaxPool(2) → Linear(2048→64), with L2 normalization on the output embedding
Loss: triplet loss, margin = 0.3, distance metric = Euclidean
Training: Adam optimizer, learning rate 0.001, batch size 128 (primary experiment), 15 epochs
Hardware: Kaggle, T4 GPU

Method: Semi-Hard Negative Mining
A known failure mode of triplet loss is that with randomly-selected negatives, the model quickly satisfies the triplet condition for most "easy" triplets, causing the loss to collapse toward zero and gradients to vanish, stalling learning. To address this, negatives were mined per-batch to satisfy the semi-hard condition:

D(anchor, positive) < D(anchor, negative) < D(anchor, positive) + margin

If no semi-hard negative existed for a given anchor in a batch, the hardest available negative was used as a fallback.

Ablation 1: Does Semi-Hard Mining Actually Help on CIFAR-10?
A second model was trained identically except using fully random negative selection instead of semi-hard mining, to test the mining claim directly rather than merely assert it.

| Model | Final Training Loss | KNN Accuracy (k=5) |
|---|---|---|
| Random negative selection (batch=128) | 0.0829 | 67.44% |
| Semi-hard negative mining (batch=128) | 0.1173 | 67.20% |

The random-negative model's training loss was notably lower, consistent with the expected failure mode (most random negatives are "easy," driving loss toward zero without necessarily reflecting stronger learning). However, downstream KNN accuracy was nearly identical, with random mining marginally outperforming semi-hard mining — the opposite of the theoretically expected outcome.

Ablation 2: Is the Mining Pool (Batch Size) the Bottleneck?
One hypothesis for Ablation 1's null result was that the mini-batch mining pool (128 images) was too small to surface genuinely informative semi-hard negatives. To test this directly, the semi-hard mining model was retrained with batch size 512.

| Model | KNN Accuracy (k=5) |
|---|---|
| Semi-hard mining, batch=128 | 67.20% |
| Semi-hard mining, batch=512 | 64.24% |
| Random negative, batch=128 | 67.44% |

This did not confirm the hypothesis. Semi-hard mining at batch=512 performed worse than both batch=128 semi-hard mining and random selection — the opposite of what a "larger pool helps mining" explanation would predict. A plausible explanation is that a larger candidate pool increases the likelihood of selecting negatives very close to the margin boundary, which can destabilize training when the margin and learning rate are not retuned for the new batch size — a known sensitivity in metric learning. This result rules out simple mining-pool scarcity as the explanation, and instead points toward CIFAR-10's coarse class separability as the more likely primary factor behind mining showing no advantage on this dataset.

Results Summary:

| Method | KNN Accuracy (k=5) |
|---|---|
| Raw pixel distance (baseline) | 28.72% |
| Semi-hard mining, batch=512 | 64.24% |
| Semi-hard mining, batch=128 | 67.20% |
| Random negative, batch=128 | 67.44% |

(See `results/retrieval_query_plane.png` and `results/tsne_embedding_visualization.png`)

Analysis:
All trained embeddings (64.24–67.44%) substantially outperformed the raw-pixel baseline (28.72%), clear evidence that triplet-loss training learns meaningful, class-discriminative structure regardless of negative-selection strategy or batch size — the choice of mining strategy had a much smaller effect than the choice to use a learned embedding at all.

Qualitative inspection of retrieval results (see figure) shows that for a query image of a "plane," 3 of 5 nearest neighbors were correctly retrieved as "plane," while the 2 incorrect retrievals ("bird" and "ship") shared visually plausible compositional similarities with the query. Three factors likely explain the ~65-67% accuracy ceiling and this error pattern:

1. Capacity bottleneck — a lightweight 3-layer CNN without residual connections or pretraining will naturally plateau in this range on CIFAR-10; deeper or pretrained architectures would likely perform better, but were outside the scope of a Basic task.
2. Scene-level composition bias — retrieval errors (confusing "plane" with "bird" or "ship") suggest reliance on global scene composition and background color rather than fine-grained object geometry.
3. Dataset coarseness, not mining pool size — the two ablations together indicate that mining strategy and batch size have limited effect on this dataset, most likely because CIFAR-10's ten classes are visually coarse and distinct enough that even random or small-pool negatives are frequently informative by chance, reducing the practical benefit of more sophisticated mining strategies that matter more in fine-grained domains like face recognition.

The t-SNE visualization of the embedding space shows substantial mixing between classes rather than fully separated, discrete clusters — consistent with the ~67% KNN accuracy reported above, which reflects meaningful but imperfect class separation. This overlap likely reflects the same factors discussed above: a lightweight architecture without pretraining, and reliance on coarse visual/compositional cues that do not always align cleanly with class boundaries.

Limitations:
- Both ablations point toward CIFAR-10's coarse class structure — rather than batch size or mining strategy — as the primary reason semi-hard mining showed no advantage here; this conclusion is dataset-specific and should not be generalized to fine-grained similarity tasks.
- The batch=512 result suggests margin (0.3) and learning rate (0.001) may need retuning for larger batch sizes; this interaction was not explored further due to time constraints.
- k=5 KNN accuracy is a proxy for embedding quality, not a direct or complete measure of it.
- Only class-label-based similarity was used as ground truth, a coarse proxy that does not capture within-class variation or cross-class visual similarity a human might still judge as meaningfully "similar."

Conclusion:
Triplet-loss training produced embeddings substantially more discriminative than raw pixel similarity (64–67% vs. 28.72% k-NN accuracy), confirming the network learns meaningful class structure regardless of configuration. Two ablations then tested the specific claim that semi-hard mining improves on random negative selection — and both rejected it: mining showed no accuracy advantage over random selection, and increasing the mining pool (batch size 512) made results worse, not better, ruling out pool scarcity as the cause. The most likely explanation is CIFAR-10's coarse, visually distinct class structure, which makes even random negatives frequently informative. The broader takeaway: a technique's theoretical benefit — established in fine-grained domains like face recognition — does not automatically transfer to coarser-grained datasets, a conclusion reached through direct experimentation rather than assumed from the literature.

References:
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR.
