#!/bin/bash

echo -e "start main"
julia --threads 18 main.jl 
wait

echo -e "start NF"
julia --threads 18 nf.jl
wait
echo -e "start timing"
julia --threads 18 timing.jl
echo -e "post vis"
julia --threads 18 post_vis.jl 
echo -e "plotting"
julia plotting.jl
wait
echo -e "done"
