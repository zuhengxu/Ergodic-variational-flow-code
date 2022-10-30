using Random, Distributions
include("model_2d.jl")

xs = Matrix(rand(MvNormal(zeros(d), [100 0; 0 1.]), 500)')

for i in 1:500
    xs[i,2] = xs[i,2] + b * xs[i,1]^2 - 100 * b
end

M = vec(mean(xs; dims = 1)) 
V = vec(var(xs; dims = 1)) 

scatter(xs[:,1], xs[:,2])