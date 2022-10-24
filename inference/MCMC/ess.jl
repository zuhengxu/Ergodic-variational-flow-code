using RCall

function multiess(D::Matrix{Float64}; f::Function = x -> x)
```
D is data matrix, each row is a sample
f is the test function, defualt set as identity, which returns ESS for mean

return "ESS = n |Λ|_p/|Σ|_p"
```  
    @rput D
    @rput f
    return rcopy(R"mcmcse:::multiESS(D, g = f)")
end

# ess(mvg; f = x -> norm(x))
function ess(D::Matrix{Float64}; f::Function = x -> x)
```
D is data matrix, each row is a sample
f is the test function operates on a single coord, defualt set as identity, which returns ESS for mean

return "ess = n λ/σ"
```  
    @rput D
    @rput f
    return minimum(rcopy(R"mcmcse:::ess(D, g = f)"))
end



##################
# Examples
####################

# D = randn(10000, 2)
# multiess(D)
# ess(D)