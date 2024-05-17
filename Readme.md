# Readme

This repository contains all of the code required to generate the tables and plots provided in arXiv:xxx.xxxx. To recreate the images simply run the provided notebook plots.ipynb. This will use the precomputed results stored in pickle files to generate the plots.

Should you want to run your own simulations and learn how to use the compressed sensing amplitude estimation approach, all of the data can be recreated by running the "run_ae_sims.py" python script from the command line with the following set of arguments

This first code block generates the data needed for Figs. 3 and 4. It runs 500 Monte Carlo trials of the csAE approach for amplitudes a=[0.1, 0.2, ..., 0.9]. With 4 threads on a small laptop this takes about 4.5 hours to complete.
```
python run_ae_sims.py --save --dir sims_C1.5_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5
python run_ae_sims.py --save --dir sims_C1.8_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.8
```

This second runs three commands that generate the data needed for Figs. XXX and XXX. It runs 500 Monte Carlo trials for differing values of eta, which corresponds to the depolarizing noise rate of the maximum depth circuit. With 4 threads on a small laptop each of these runs takes about 4.5 hours to complete.

```
python run_ae_sims.py --save --dir sims_C1.5_eta1e-3_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5 --eta=1e-3
python run_ae_sims.py --save --dir sims_C1.5_eta1e-4_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5 --eta=1e-4
python run_ae_sims.py --save --dir sims_C1.5_eta1e-5_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5 --eta=1e-5
python run_ae_sims.py --save --dir sims_C1.5_eta1e-6_final --nthreads=4 --num_mc=500 --num_lengths=8 --C=1.5 --eta=1e-6
```

Once the data is generated, simply run the notebook plots.ipynb to generate the figures and tables. For your convenience, the precomputed data is already stored so there is no need to rerun these long simulations, except for reproducibility.