function stability_plot(o::ErgodicFlow, a::HF_params, Ns; 
                        nsample::Int = 100, refresh::Function = ErgFlow.pseudo_refresh_coord, inv_ref::Function = ErgFlow.inv_refresh_coord, 
                        res_dir = "result/", res_name = "stability.jld") 
    EE_fwd = zeros(size(Ns, 1)+1, nsample) 
    EE_bwd = zeros(size(Ns, 1)+1, nsample) 
    for i in 1:size(Ns, 1)
        EE_fwd[i+1, :] .= ErgFlow.error_checking_fwd(o,a, Ns[i]; Nsample = nsample, refresh = refresh, inv_ref = inv_ref) 
        @info "$(Ns[i]) fwd done"
        # println(e)
        EE_bwd[i+1, :] = ErgFlow.error_checking_bwd(o,a, Ns[i]; Nsample = nsample, refresh = refresh, inv_ref = inv_ref) 
        @info "$(Ns[i]) bwd done"
        # println(e1)
    end
    JLD.save(joinpath(res_dir, res_name), "fwd_err", EE_fwd, "bwd_err", EE_bwd, "Ns", vcat([0], Ns))
end
