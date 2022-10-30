## How to run the code
In this folder `example/neals_funnel/`: 
- first run `julia main_2d.jl` or `julia --threads $(number of threads) main_2d.jl`  to compute ELBOs and KSD comparison with NUTS for 2d funnel example
- then run `julia stability.jl` or `julia --threads $(number of threads) stability.jl`  to perform numerical stability analysis
- then run `julia nf.jl` or `julia --threads $(number of threads) nf.jl`  to perform Planar flow analysis
- then run `julia conv.jl` or `julia --threads $(number of threads) conv.jl`  to perform running average convergence analysis
- then run `julia timing.jl` or `julia --threads $(number of threads) timing.jl`  to perform timing analysis
- then run `julia main_multi_d.jl` or `julia --threads $(number of threads) main_multi_d.jl` to compute ELBOs and KSD comparison with NUTS for 5d and 20d funnel examples
- last run `julia plotting.jl` to generate plots