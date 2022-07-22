# Ergodic-variational-flow-code
Code that reproduce experiments in [Ergodic variational flow](https://arxiv.org/pdf/2205.07475.pdf)

Examples run and generate output using Julia v1.7.1.
- `inference/` provides functions for inferences (NUTS/mean-field Gaussian VI) 
- Ergodic variational flow is implemented as a julia package [ErgFlow.jl](https://github.com/zuhengxu/ErgFlow.jl)
- `examples/` provides code to replicate examples and figures 


## How to run the code
Each experiment should be run in its own folder (e.g., `example/1d_cauchy`, `example/banana/`, and etc.), see each detailed instructions in each folder. Generally, 
- first run `julia main.jl` or `julia --threads $(number of threads) main.jl`  to perform the experiment
- then run `julia plotting.jl` to generate plots
