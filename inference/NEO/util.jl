


function logsumexp_sweep(X::Vector{<:Real}, K)
    N = length(X)
    T = zeros(N-K+1)
    @views for i = 1:N-K+1
        t = LogExpFunctions.logsumexp(X[i:i+K-1])
        T[i] = t
    end
    return T
end