## How to run the code
In this folder `example/banana/`: 
- first run `julia main.jl` or `julia --threads $(number of threads) main.jl`  to compute ELBOs and KSD comparison with NUTS
- then run `julia stability.jl` or `julia --threads $(number of threads) stability.jl`  to perform numerical stability analysis
- then run `julia nf.jl` or `julia --threads $(number of threads) nf.jl`  to perform Planar flow analysis
- then run `julia conv.jl` or `julia --threads $(number of threads) conv.jl`  to perform running average convergence analysis
- then run `julia timing.jl` or `julia --threads $(number of threads) timing.jl`  to perform timing analysis
- then run `julia neo_main.jl` or `julia --threads $(number of threads) neo_main.jl`  to perform NEO tuning
- then run `julia neo_timing.jl` or `julia --threads $(number of threads) neo_timing.jl`  to perform NEO evaluation
- last run `julia plotting.jl` to generate plots