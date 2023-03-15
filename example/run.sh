#!/bin/bash

echo -e "start linear_regression"
cd linear_regression
julia --threads 18 conv.jl
cd ..
wait

echo -e "start heavy_reg"
cd heavy_reg
julia --threads 18 conv.jl
cd ..
wait

echo -e "start lin_reg_heavy"
cd lin_reg_heavy
julia --threads 18 conv.jl
cd ..
wait

echo -e "start logistic_reg"
cd logistic_reg
julia --threads 18 conv.jl
cd ..
wait

echo -e "start poiss"
cd poiss
julia --threads 18 conv.jl
cd ..
wait

echo -e "start sp_reg_big"
cd sp_reg_big
julia --threads 18 conv.jl
cd ..
wait

echo -e "start sparse_regression"
cd sparse_regression
julia --threads 18 conv.jl
cd ..
wait