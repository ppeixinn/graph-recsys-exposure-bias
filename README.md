# Does LightGCN's Advantage Survive Unbiased Evaluation?
### BPR-MF vs. LightGCN on KuaiRand-Pure and KuaiRec

SC4020 Project 1. Most recommender test sets only contain items that the platform's own recommender chose to show, so they reward models that recommend what was already popular. We compare a plain matrix-factorization recommender with a graph neural network recommender on two short-video datasets that also contain feedback the recommender did not select, and ask whether the graph model's advantage holds there.

## Methods
| Method | Description |
|---|---|
| **Most-popular** | Recommends the items with the most training interactions to every user. Non-learned baseline. |
| **BPR-MF** | Matrix factorization trained with the Bayesian Personalized Ranking loss (Rendle et al., UAI 2009). No graph. |
| **LightGCN** | Simplified graph convolution (He et al., SIGIR 2020, [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)). Embeddings are propagated over the normalized user-item bipartite graph and the layer outputs are averaged. With 0 layers it is BPR-MF. |

## Datasets
Both come from Kuaishou, a short-video app. Each has a biased part (videos chosen by the platform recommender) and an unbiased part.

| Dataset | Users | Items | Train positives | Density | Positive | Unbiased test |
|---|---|---|---|---|---|---|
| KuaiRand-Pure ([paper](https://arxiv.org/abs/2208.08696)) | 25,689 | 7,218 | 466,490 | 0.25% | `is_click == 1` | 194,539 clicks on randomly exposed videos |
| KuaiRec ([paper](https://arxiv.org/abs/2202.10842)) | 7,173 | 10,611 | 609,375 | 0.80% | `watch_ratio >= 2` | 216,727 positives in a fully observed block of 1,408 users x 3,326 videos |

MovieLens-100K is used only for quick smoke tests. Download instructions are in [data/README.md](data/README.md).

## Evaluation
- The biased positives of each user are split at random: 80% train / 20% test, then 10% of train held out for validation.
- Every trained model is scored on two test sets: the biased `test` and the `test_unbiased` set. On KuaiRec's unbiased test, items are ranked only among the block videos.
- Full ranking over all items, known positives masked. Metrics: Recall@20, NDCG@20, margin over most-popular, item coverage, popularity of recommendations, training time.
- Model selection uses validation Recall@20 with early stopping; the unbiased test is never used for tuning. Results are reported as mean and standard deviation over several seeds.

## Repository layout
```
src/          Python code (data loading, models, training, experiments)
notebooks/    one notebook per step, with outputs
results/      metrics and figures
data/         datasets (not tracked, see data/README.md)
```

## Setup
Python 3.13 with PyTorch (CPU), numpy, scipy, pandas, matplotlib, scikit-learn and Jupyter.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Download the datasets as described in [data/README.md](data/README.md), then run the notebooks in `notebooks/` in order.

## Team
Group XX (SC4020). Individual contributions are described in `group_XX_contribution.pdf`.
