#!/bin/bash

echo -e "start banana"
cd banana
julia --threads 18 conv.jl
cd ..
wait

echo -e "cross"
cd cross
julia --threads 18 conv.jl
cd ..
wait

echo -e "neals funnel"
cd neals_funnel
julia --threads 18 conv.jl
cd ..
wait

echo -e "warped Gaussian"
cd warped_gaussian
julia --threads 18 conv.jl
cd ..
wait
# echo -e "start logistic_reg"
# cd logistic_reg
# julia --threads 18 conv.jl
# cd ..
# wait

# echo -e "start poiss"
# cd poiss
# julia --threads 18 conv.jl
# cd ..
# wait

# echo -e "start sp_reg_big"
# cd sp_reg_big
# julia --threads 18 conv.jl
# cd ..
# wait

# echo -e "start lin_reg_heavy"
# cd lin_reg_heavy
# julia --threads 18 conv.jl
# cd ..
# wait