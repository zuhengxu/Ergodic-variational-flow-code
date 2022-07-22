## How to run the code
In this folder `example/neals_funnel/`:
- first run `julia main_2d.jl` or `julia --threads $(number of threads) main_2d.jl`  to perform the experiment for 2D-neal's funnel
- then run `julia stability.jl` or `julia --threads $(number of threads) stability.jl`  to perform numerical stability analysis for 2D-neal's funnel
- run `julia main_multi_d.jl` or `julia --threads $(number of threads) main_multi_d.jl`  to perform the experiment for higher-dimensional neal's funnel
- last run `julia plotting.jl` to generate plots
