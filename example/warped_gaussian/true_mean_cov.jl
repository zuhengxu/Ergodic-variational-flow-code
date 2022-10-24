using Random, Distributions
include("model_2d.jl")

samps = rand(Normal(0,1), (2,50000))
samps[1,:] *= 1
samps[2,:] *= 0.12
rs = sqrt.(sum(samps.^2, dims=1))[1,:]
θs = atan.(samps[2,:], samps[1,:])
θs = θs - rs/2
x0s = rs .* cos.(θs)
y0s = rs .* sin.(θs)

xs = hcat(x0s, y0s)

M = vec(mean(xs; dims = 1)) 
V = vec(var(xs; dims = 1)) 

# scatter(x0s, y0s)
scatter(xs[:,1], xs[:,2])