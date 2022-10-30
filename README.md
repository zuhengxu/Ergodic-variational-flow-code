# Ergodic-variational-flow-code
Code that reproduces experiments in [Ergodic variational flows](https://arxiv.org/pdf/2205.07475.pdf)

## Package installation

We implement a practical version of the Ergodic variational flows---Hamiltonian ergodic flow---as a julia package [ErgFlow.jl](https://github.com/zuhengxu/ErgFlow.jl), which can be installed as following:

```julia
using Pkg
Pkg.add("https://github.com/zuhengxu/ErgFlow.jl")
```

## Source code 
Examples run and generate output using Julia v1.8.2.
- `inference/` provides implementation of our competitors (NUTS/standard HMC/three normalizing flow methods), as well as utility functions for performing inference (KSD estimation/mean-field Gaussian VI method/etc.)
- `examples/` provides code to replicate examples and figures 


## How to run the code
Each experiment should be run in its own folder (e.g., `example/1d_cauchy`, `example/banana/`, and etc.), see each detailed instructions in each folder. Specifically, 
* To generate Figure 1 (effect of irrational shift), execute the following:
```bash
cd "example/uniform/"
julia beta_mixture.jl
```
* For instructions on generating plots for all synthetic examples, check the readme files in each of the `1d_cauchy`, `1d_gaussian`, `1d_mixture`, `banana`, `cross`, `neals_funnel`, and `warped_gaussian` folders. 
* To generate plots for all real data examples: `cd` into each of the `heavy_reg`, `linear_regression`, `lin_reg_heavy`, `logistic_reg`, `poiss`,`sparse_regression`, and `sp_reg_big` folders, and then execute `./run.sh`. Note that users should specify the number of threads in `run.sh` based on their own computation resources (currently set to be 18).

