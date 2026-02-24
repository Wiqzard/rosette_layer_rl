# Single-Layer Transfer from Pretrained Video Models to Reinforcement Learning

## Abstract
We investigate whether a single frozen layer from a pretrained video model, combined with lightweight input/output adapters, can provide a strong inductive bias for reinforcement learning (RL). We propose a layer-selection criterion based on reward predictivity, transition predictivity, and representation rank, then validate selected layers in a controlled RL benchmark. In our prototype, the selected structured mid-layer consistently outperforms alternative layer choices. Replace placeholder numbers below with your final large-model results.

## 1. Introduction
Large pretrained video models encode rich temporal and spatial priors that may be beneficial for decision-making tasks. Most transfer methods use the full backbone or multiple blocks, which is computationally expensive and can be unstable for RL. This work studies a minimal alternative: freeze a **single internal layer**, and train only small adapters before and after it.

### Contributions
1. A practical hypothesis: one carefully chosen frozen layer can outperform many arbitrary layer choices for RL.
2. A layer-selection analysis combining linear reward probing, transition prediction, and feature-rank diagnostics.
3. A reproducible experiment pipeline that compares layer choices with identical training budgets.

## 2. Related Work
- RL with pretrained visual/video representations: `[CITE: pretrained representation RL papers]`
- World models and latent dynamics for control: `[CITE: world model / latent dynamics papers]`
- Probing and layer-wise transfer in foundation models: `[CITE: probing / representational transfer papers]`

Gap: prior work rarely isolates a **single frozen layer** as the core reusable computation for RL and evaluates systematic layer ranking criteria before policy learning.

## 3. Method
### 3.1 Problem Setup
Given observations $o_t$ and actions $a_t$, we define a policy/value head over latent features obtained via:

$$
z_t = f_\ell(g_\theta(o_t)), \quad Q(o_t, a_t) = h_\phi(z_t)_a
$$

- $f_\ell$: frozen candidate layer from a pretrained video backbone (layer index $\ell$)
- $g_\theta$: small input adapter
- $h_\phi$: small output adapter

Only $\theta, \phi$ are trained with RL updates.

### 3.2 Layer Selection Criteria
For each candidate layer, collected on an exploratory dataset:

1. Reward predictivity: linear probe $z_t \to r_t$ (test $R^2$)
2. Transition predictivity: linear probe $[z_t, a_t] \to z_{t+1}$ (test $R^2$)
3. Effective rank: entropy-based rank of feature covariance

Composite score:

$$
S(\ell)=0.45\,\tilde R^2_{\text{reward}}+0.45\,\tilde R^2_{\text{trans}}+0.10\,\text{rank}
$$

where $\tilde R^2$ is a normalized version of $R^2$ in $[0,1]$.

### 3.3 Training
- RL algorithm: off-policy Q-learning with replay buffer
- Same hyperparameters and budget across all layer choices
- Multi-seed evaluation and final ranking by mean return over the last 50 episodes

## 4. Experiments
### 4.1 Experimental Questions
1. Does the analysis score identify layers that perform better in RL?
2. Does a chosen single layer outperform random/bottleneck alternatives under fixed compute?
3. How stable are the results across random seeds?

### 4.2 Setup
Fill in for real run:
- Backbone: `[CogVideoX-1.5 / HunyuanVideo-1.5 / other]`
- Candidate layers: `[indices and module names]`
- Environments: `[list tasks]`
- Budget: `[steps, episodes, wall-clock]`
- Hardware: `[GPU/CPU, memory]`

Prototype artifacts in this repo:
- Layer analysis: `outputs/layer_analysis.csv`
- RL comparison: `outputs/experiment_summary.csv`

## 5. Results
### 5.1 Layer Analysis
Paste or regenerate from `outputs/layer_analysis.csv`.

| Layer | Reward $R^2$ | Transition $R^2$ | Effective Rank | Composite Score |
|---|---:|---:|---:|---:|
| `[fill]` | `[fill]` | `[fill]` | `[fill]` | `[fill]` |

### 5.2 RL Performance
Paste table from `outputs/experiment_summary_table.md`.

| Layer | Mean Return (Last 50) | Std Across Seeds | Mean Return (All Episodes) |
|---|---:|---:|---:|
| `[fill]` | `[fill]` | `[fill]` | `[fill]` |

### 5.3 Interpretation
- The layer with the best composite analysis score achieved `[X%]` higher final return than `[baseline layer]`.
- Transition predictivity appears to correlate with policy learning speed `[quantify]`.
- Bottleneck/random layers underperform due to `[hypothesis]`.

## 6. Limitations
- Prototype currently uses a synthetic video-like benchmark.
- Results on full-scale pretrained video backbones may depend on extraction layer granularity and compute budget.
- The linear probes are simple diagnostics, not causal proof.

## 7. Conclusion
A single frozen layer plus tiny adapters is a promising low-parameter transfer strategy for RL. The proposed layer-selection analysis provides a practical way to narrow candidates before expensive policy training. Next steps are direct validation on CogVideo/HunyuanVideo layers and broader task suites.

## Appendix A. Reproducibility Checklist
- Code: this repository
- Random seeds: `[list]`
- Hyperparameters: `[attach config table]`
- Outputs: `outputs/*.csv`
