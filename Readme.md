# Quantum amplitude estimation: csAE and the geometric-ladder method

Two non-adaptive amplitude-estimation algorithms live in this repository. Both measure the Grover iterator at a fixed set of depths and post-process the outcomes classically, so every circuit can run in parallel; they differ in the depth schedule and in the estimator.

| | csAE | Geometric ladder + exact ML |
|---|---|---|
| Paper | [arXiv:2405.14697](https://arxiv.org/abs/2405.14697) | [arXiv:2609.02715](https://arxiv.org/abs/2609.02715) |
| Depth schedule | powers of two | ratio 1.45 ladder, one extra rung near the cap |
| Estimator | ESPRIT on a sparse virtual array | exact global maximum likelihood (one matrix product) |
| Constants at 95% | 4.3/ε total, 0.26/ε sequential depth | 2.8–3.1/ε total, 0.21/ε sequential depth |
| Where | [`csae/`](csae/): `estimator.py`, `csAE_example.ipynb`, `plots.ipynb` | [`mlqae/`](mlqae/) with its own [README](mlqae/README.md); core module `mlqae/core.py` |
| Live demo | — | https://unitaryfoundation.github.io/csAE/ (source: [`docs/index.html`](docs/index.html)) |

Live demo of the ladder mechanism: **https://unitaryfoundation.github.io/csAE/**

The rest of this file documents csAE, which lives in the `csae/` package (import it from the repository root: `from csae import *`). For the ladder method, start at [`mlqae/README.md`](mlqae/README.md).

## csAE

This repository contains all of the code required to generate the tables and plots provided in arXiv:2405.14697. To recreate the images simply run the provided notebook `csae/plots.ipynb`. This will use the precomputed results stored in pickle files to generate the plots.

Should you want to run your own simulations and learn how to use the compressed sensing amplitude estimation approach, we recommend looking at the notebook `csae/csAE_example.ipynb`. This contains a minimal working example demonstrating the functionality.

All of the data used to generate the results in the paper can be recreated by running the `csae/run_ae_sims.py` python script from the repository root with the following commands. This code block generates the data needed for Fig. 3 and Tab. 1. It runs 500 Monte Carlo trials of the csAE approach for random amplitudes in the range a=(0.1, ..., 0.9). With 12 threads on a workstation this takes about XXX hours to complete.

```
python csae/run_ae_sims.py --save --dir=csae/sims_final/ --nthreads=24 --num_lengths=6 --C=4 --adjacency=5 --num_mc=500 > csae/sims_final/outfileC4.0.dat
python csae/run_ae_sims.py --save --dir=csae/sims_final/ --nthreads=24 --num_lengths=6 --C=8 --adjacency=5 --num_mc=500 > csae/sims_final/outfileC8.0.dat

```

Once the data is generated, simply run the notebook `csae/plots.ipynb` to generate the figures and tables. For your convenience, the precomputed data is already stored so there is no need to rerun these long simulations, except for reproducibility.
