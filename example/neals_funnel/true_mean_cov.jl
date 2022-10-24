using Random, Distributions
include("model_2d.jl")

xs = zeros(500000, d)
xs[:,1] = rand(Normal(0, sqrt(σ²)), 500000)
for i in 1:500000
    xs[i,2] = rand(Normal(0, exp(xs[i,1]/4)))
end

M = vec(mean(xs; dims = 1))
V = sqrt.(vec(var(xs; dims = 1)))

scatter(xs[:,1], xs[:,2])