# Ergodic-variational-flow-code
Code that reproduces experiments in [MixFlows: principled variational inference via mixed flows](https://arxiv.org/pdf/2205.07475.pdf)

## Package installation

We implement a practical version of the Ergodic variational flows---Hamiltonian
ergodic flow---as a julia package
[ErgFlow.jl](https://github.com/zuhengxu/ErgFlow.jl), which can be installed by
executing the following code:

```julia
# in Julia REPL
using Pkg
Pkg.add("https://github.com/zuhengxu/ErgFlow.jl")
```

## Code description 
Examples run and generate output using Julia v1.8.1.
- `inference/` provides implementation of our competitors (NUTS/standard HMC/three normalizing flow methods), as well as utility functions for performing inference (KSD estimation/mean-field Gaussian VI method/etc.)
- `examples/` provides code to replicate results and figures for each example


## How to run the code
Each experiment should be run in its own folder (e.g., `example/1d_cauchy`, `example/banana/`, and etc.). Specifically, 
* To generate Figure 1 (effect of irrational shift), execute the following code in bash:
```bash
cd example/uniform/
julia beta_mixture.jl
```
* To generate plots for all synthetic examples, check the `README.md` files in each of the `1d_cauchy`, `1d_gaussian`, `1d_mixture`, `banana`, `cross`, `neals_funnel`, and `warped_gaussian` folders. 
* To generate plots for all real data examples: `cd` into each of the `heavy_reg`, `linear_regression`, `lin_reg_heavy`, `logistic_reg`, `poiss`,`sparse_regression`, and `sp_reg_big` folders, and then execute the following in bash:
```bash
./run.sh
```  
Note that users should modify the number of execution threads in `run.sh` based on their own computational resources (currently set to be 18).

