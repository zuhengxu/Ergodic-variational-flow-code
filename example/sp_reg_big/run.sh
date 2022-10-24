#!/bin/bash

echo -e "start main"
julia --threads 18 main.jl 
wait

echo -e "start NF"
julia --threads 18 nf.jl
wait
echo -e "lala"