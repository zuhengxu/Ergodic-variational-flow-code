######################333
# ksd
######################333
using Base.Threads, Distances

function rbf_kernal(x,y,bw)
    return exp(-(1. / bw) * norm(x-y)^2)
end

function ∇x(x,y,bw)
    return -rbf_kernal(x,y,bw) .* (x .- y) * (2. / bw)
end

function ∇y(x,y,bw)
    return -rbf_kernal(x,y,bw) .* (y .- x) * (2. / bw)
end

function trace_term(x,y,bw)
    r = rbf_kernal(x,y,bw)
    return -(4. / bw^2.) * r * norm(x-y)^2 + (2 * length(x) / bw) * r 
end

function Up(x, y, grd, bw)
    # return rbf_kernal(x, y, bw) * dot(grd(x), grd(y)) + dot(grd(x), ∇y(x,y,bw)) + dot(grd(y), ∇x(x,y,bw)) + trace_term(x,y,bw)
    return rbf_kernal(x, y, bw) * dot(grd(x), grd(y)) + dot(grd(x) .- grd(y), ∇y(x,y,bw)) + trace_term(x,y,bw)
end

function adapt_bw(D)
    # h selected to be the median of the squared Euclidean distance between pairs of sample points
    R = pairwise(SqEuclidean(1e-12), D, dims = 1)
    return median(R)
end
function adapt_bw!(R, D)
    # h selected to be the median of the squared Euclidean distance between pairs of sample points
    R .= pairwise(SqEuclidean(1e-12), D, dims = 1)
    return median(R)
end
# compute the ksd using standard RBF kernel
function ksd(D, grd::Function; bw = -1)
    #=
    D: sample matrix (each row is a data point)
    grd: ∇logp ---score of target distribution
    =#
    if bw < 0  
        bw = adapt_bw(D)
    end
    println(bw)
    N = size(D,1)
    ksd = Threads.Atomic{Float64}(0.0)
    @info "computing KSD" 
    prog_bar = ProgressMeter.Progress(N^2, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i in 1:N 
        @simd for j in 1:N
            @inbounds atomic_add!(ksd, Up(@view(D[i,:]), @view(D[j,:]), grd, bw))
            ProgressMeter.next!(prog_bar)
        end
    end
    # scaled_ones = (1. / N) * ones(N)
    # return sqrt(scaled_ones' * Ups * scaled_ones)
    return sqrt(ksd[])/N
end

function ksd!(R, D, grd::Function; bw = -1)
    #=
    D: sample matrix (each row is a data point)
    grd: ∇logp ---score of target distribution
    =#
    if bw < 0  
        bw = adapt_bw!(R, D)
    end
    N = size(D,1)
    ksd = Threads.Atomic{Float64}(0.0)
    @info "computing KSD" 
    prog_bar = ProgressMeter.Progress(N^2, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i in 1:N 
        @simd for j in 1:N
            @inbounds atomic_add!(ksd, Up(@view(D[i,:]), @view(D[j,:]), grd, bw))
            ProgressMeter.next!(prog_bar)
        end
    end
    # scaled_ones = (1. / N) * ones(N)
    # return sqrt(scaled_ones' * Ups * scaled_ones)
    return sqrt(ksd[])/N
end

