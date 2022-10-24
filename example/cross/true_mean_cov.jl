using Random, Distributions
include("model_2d.jl")

xs = zeros(50000, d)
for i in 1:50000
    u = rand(Uniform())
    if u <= 0.25
        xs[i,:] = rand(MvNormal([0, 2.], [0.15^2 0; 0 1]))
    elseif u <= 0.5
        xs[i,:] = rand(MvNormal([-2, 0.], [1 0; 0 0.15^2]))
    elseif u <= 0.75
        xs[i,:] = rand(MvNormal([2., 0.], [1 0; 0 0.15^2]))
    else
        xs[i,:] = rand(MvNormal([0, -2.], [0.15^2 0; 0 1]))
    end
end

M = vec(mean(xs; dims = 1))
V = vec(var(xs; dims = 1))

scatter(xs[:,1], xs[:,2])