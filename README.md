# steem-value-transfer

Code for measuring the couplings of a two-layer material–information network on the Steem
blockchain, for the article

> A. N. Pisarchik and A. E. Hramov, *Copying, capacity and a market for votes: the measured couplings
> of a material–information network* (in preparation, 2026).

In Steem the same accounts follow, post, reblog, reply, vote and transfer tokens, and every operation
is on the chain. The value-transfer theory of material–information networks describes such a system
by a coupling operator W ∝ D^(−θ) A D^(−γ) in each layer and by two channels between the layers
(λ_IM: attention raises production; λ_MI: material resources buy attention). The repository estimates
all of them from public data:

| Part | Design | Script | What it estimates |
|---|---|---|---|
| Information layer | reblog pairs, post fixed effects, category placebo | `src/fit_reblog.py` | source exponent γ |
| | case–control sample of exposed readers, pair fixed effects | `src/fit_recipient_cc.py` | recipient exponents (number of sources, output per source, daily load) |
| | within-reader panel over months | `src/fit_follower_panel.py`, `src/fit_asymmetry.py` | recipient exponent θ net of selection |
| | votes (protocol rule as positive control) | `src/fit_votes_model0.py` | saturating (club) division of voting power |
| Channel I→M | stacked event study of reblogs by big accounts, audience held fixed; distributed lags; permutations | `src/fit_model_c.py`, `src/fit_model_c_dl.py`, `src/perm_model_c.py` | λ_IM: reward of later posts after an information shock |
| Channel M→I | bought bot votes vs paid-but-not-voted placebo, by post age and voter type; dose | `src/fit_mi.py` | λ_MI: organic attention brought by a bought vote |
| Material layer | transfer operator R_v = Σ_u T_uv / O_u and the stake per month | `src/fit_material.py` | postulate P1 and Result 1 (frozen pattern set by unequal row sums) |

Every estimator is first run on synthetic data in the SteemOps format with a known truth
(`validation/`, Appendix A of the paper).

## Contents

| Path | What it is |
|---|---|
| `src/cpois.py` | Poisson regression with fixed effects (conditional likelihood), clustered sandwich errors; used by all fits |
| **Information layer** (February–April 2018) | |
| `src/inventory_steemops.py` | Counts and distributions for one SteemOps month (no unpacking) |
| `src/build_reblog_pairs.py` | Reblog pairs, placebos, recipient-load cells, case–control sample, per-reader monthly summary |
| `src/fit_reblog.py` | Source side: γ, placebos, within-pair difference, shape test |
| `src/fit_recipient_cc.py` | Recipient side: decomposition of the feed load on the case–control sample |
| `src/fit_follower_panel.py` | Recipient side within readers; robustness by size and direction of change |
| `src/follow_decomposition.py`, `src/fit_asymmetry.py` | Follow decisions of the reader versus activity of the sources |
| `src/fetch_vote_rshares.py`, `src/fit_votes_model0.py` | rshares of votes through the API; value of a vote versus the voter's load |
| **Reward data** (December 2017 – June 2018) | |
| `src/probe_api.py` | Checks the API nodes and the reward fields |
| `src/sample_authors.py` | Bots, stratified sample of human authors, vote buyers, their root posts |
| `src/thin_posts.py` | At most 3 posts per author-week plus all bought posts, with weights |
| `src/fetch_post_rewards.py`, `src/check_rewards.py`, `src/parse_post_rewards.py` | Resumable download of rewards and votes (`condenser_api`), progress check, flat table |
| **Channel I→M** | |
| `src/build_panel_4b.py` | Author-week panel: active followers F, posts P, shocks, reach; post-level rewards |
| `src/fit_model_c.py` | Stacked event study (stack × author and stack × week effects, ln F and ln P held fixed) |
| `src/fit_model_c_dl.py` | Distributed leads and lags of ln(1 + reach) |
| `src/perm_model_c.py` | Permutation tests for both designs |
| **Channel M→I** | |
| `src/build_mi_events.py` | Purchases from bid bots, bot votes, refunds, services; organic votes and replies per 12-h bin of post age and voter type |
| `src/fit_mi.py` | Poisson within post: voted / placebo / service, leads, dose, strict placebo |
| **Material layer** | |
| `src/build_material.py` | Monthly transfer operator, inflow, row sums, stake, account classes |
| `src/fit_material.py` | Cross-section in the PRL form, within accounts, timing |
| `validation/` | Synthetic data generators, a mock API, and `validate.sh` |
| `figures/` | Scripts for all figures of the paper |
| `data/README.md` | Archives to download and the derived files |
| `reproduce.sh` | The full sequence of commands (part A: information layer, part B: channels and material layer) |

## Requirements

Python ≥ 3.10. Data assembly and the API download use only the standard library; estimation needs
`numpy`, `pandas`, `scipy`, figures need `matplotlib` (`requirements.txt`). A panel over the follow
graph (~20 million edges) takes 30–60 minutes and a few GB of memory on a laptop.

## Quick check (no data, no network)

```bash
pip install -r requirements.txt
bash validation/validate.sh        # ~15 min
```

Expected (Appendix A of the paper):

- **source and recipient exponents**, truths (γ, θ) = (0.5, 0.5), (0, 0), (0.2, 0.8): source γ ≈ 0.44 /
  0.00 / 0.21; recipient decomposition, number of sources ≈ −0.51 / +0.02 / −0.79; within-reader panel
  +0.00 / −1.00 for within-reader truths 0 / −1 with selection −1 between readers;
- **votes**: club form with k_c ≈ 12 for a protocol break at 10 votes per day;
- **reward data**: `ALL OK` (strata, bots, buyers, download with errors and interruption, sums of rshares);
- **I→M**: stacked event study, mean over lags 0–6 −0.01 for λ = 0 and +0.11 for the true +0.114,
  audience effect μ = 0.82 for the true 0.8; distributed lags, cumulative effect ≈ 0 for λ = 0;
- **M→I**: bought-vote effect −0.01 when there is none and +0.17 for the true +0.20; placebo ≈ 0;
- **material layer**: slope 0.090 for the 0.089 of the simulated PRL model (Eq. of Result 1 gives 0.122);
  0.00 in the null case.

## Reproducing the results

Download the archives (`data/README.md`), then from the data folder:

```bash
SRC=/path/to/steem-value-transfer/src bash /path/to/steem-value-transfer/reproduce.sh A     # information layer
SRC=/path/to/steem-value-transfer/src bash /path/to/steem-value-transfer/reproduce.sh B     # channels, material layer
```

Part B downloads the rewards and votes of the sampled posts from public API nodes (10–20 h; the download resumes where
it stopped when the same command is run again). All other steps are offline.

Known biases, documented in the paper: an engagement is credited to the earliest reblog through which
the reader could have seen the post (γ attenuated by ~10% on synthetic data); in the event study, posts
reblogged by a big account are dropped from the outcome (otherwise the selection of the reblogged post
creates a pre-trend); clustered standard errors of the event study are too small by ~25% on synthetic
data, so the paper reports permutation p-values.

## License

Code: MIT (`LICENSE`). SteemOps data are distributed by their authors (Li et al., 2021); API data are
public chain data. This repository redistributes neither.

## Citation

See `CITATION.cff`.
