using Bijectors, Flux, Zygote

function lrelu_layer(xdims::Int; hdims::Int=20)
    nn = Chain(Flux.Dense(xdims, hdims, leakyrelu), Flux.Dense(hdims, hdims, leakyrelu), Flux.Dense(hdims, xdims))
return nn
end

function affine_coupling_layer(shifting_layer, scaling_layer, dims, masking_idx)
    Bijectors.Coupling(θ -> Bijectors.Shift(shifting_layer(θ)) ∘ Bijectors.Scale(scaling_layer(θ)), Bijectors.PartitionMask(dims, masking_idx))
end


function RealNVP_layers(q0, nlayers, d; hdims=20)
    xdims = Int(d/2)
    # println(xdims)
    scaling_layers = [ lrelu_layer(xdims; hdims = hdims) for i in 1:nlayers ]
    shifting_layers = [ lrelu_layer(xdims; hdims = hdims) for i in 1:nlayers ]
    ps = Flux.params(shifting_layers[1], scaling_layers[1]) 
    Layers = affine_coupling_layer(shifting_layers[1], scaling_layers[1], d, xdims+1:d)
    # number of affine_coupling_layers with alternating masking scheme
    for i in 2:nlayers
        Flux.params!(ps, (shifting_layers[i], scaling_layers[i]))
        Layers = Layers ∘ affine_coupling_layer(shifting_layers[i], scaling_layers[i], d, (i%2)*xdims+1:(1 + i%2)*xdims) 
    end
    flow = Bijectors.transformed(q0, Layers)
    return flow, Layers, ps
end

function train_rnvp(q0, logp, logq, d::Int, niters::Int; 
                nlayers = 5, hdims = 20,  
                elbo_size::Int = 10, optimizer = Flux.ADAM(1e-3), kwargs...)

    flow, Layers, ps = RealNVP_layers(q0, nlayers, d; hdims = hdims)
    #define loss
    loss = () -> begin 
        elbo = nf_ELBO(flow, logp, logq; elbo_size = elbo_size)
        return -elbo
    end

    elbo_log, ps_log = vi_train!(niters, loss, ps, optimizer)
    return flow, [[copy(p) for p in ps]], -elbo_log, ps_log
end
