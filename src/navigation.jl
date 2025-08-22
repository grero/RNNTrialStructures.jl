using LinearAlgebra

struct Arena{T<:Real}
    ncols::Int64
    nrows::Int64
    colsize::T
    rowsize::T
end

function signature(arena::Arena{T},h=zero(UInt32)) where T <: Real
   for q in [arena.ncols, arena.nrows, arena.colsize, arena.rowsize]
        for ii in q
            h = crc32c(string(ii), h)
        end
    end 
    h
end

function get_center_position(arena::Arena{T}) where T <: Real
    i = div(arena.ncols,2)+1
    j = div(arena.nrows,2)+1
    get_position(i,j,arena)
end

function get_center(arena::Arena{T}) where T <: Real
    w,h = extent(arena)
    w/2,h/2
end

extent(arena::Arena)  = (arena.ncols*arena.colsize, arena.nrows*arena.rowsize)

"""
Return possible steps
"""
function check_step(i::Int64,j::Int64, ncols::Int64, nrows::Int64) 
    possible_steps = [(0,0)]
    if i > 1 
        push!(possible_steps, (-1, 0))
    end
    if i < ncols
        push!(possible_steps, (1, 0))
    end
    if j > 1
        push!(possible_steps, (0,-1))
    end
    if j < nrows
        push!(possible_steps, (0,1))
    end
    possible_steps
end

"""
    get_coordinate(i::Int64, j::Int64,arena::Arena{T};Δθ=π/4,rng=Random.default_rng()) where T <: Real

Generate a new coordinate in `arena` by choosing a random step subject to the arena constraints.
"""
function get_coordinate(i::Int64, j::Int64,arena::Arena{T};Δθ=π/4,rng=Random.default_rng()) where T <: Real
    possible_steps = check_step(i,j,arena.ncols,arena.nrows)
    Δ = rand(rng, possible_steps)
    (i,j) = (i+Δ[1], j+Δ[2])
    (i,j)
end

function get_coordinate(i::Int64, j::Int64,arena::Arena{T},θhd::T;Δθ::T=T(π/4),rng=Random.default_rng(),p_hd=T(0.5)) where T <: Real
    possible_steps = check_step(i,j,arena.ncols,arena.nrows)
    nsteps = length(possible_steps)
    #figure out the step most aligned with the current head direction
    dx = [cos(θhd),sin(θhd)]
    m = zeros(T, nsteps) 
    for (ii,ps) in enumerate(possible_steps)
        m[ii] = dx[1]*ps[1] + dx[2]*ps[2]
    end
    kk = findall(m.==maximum(m))
    pp = p_hd
    pq = T(1.0/(nsteps-length(kk)))
    jj = 0
    if rand(rng) < pp
        jj = rand(rng,kk)
    else
        for (k,ii) in enumerate(setdiff(1:nsteps, kk))
            if rand(rng) < k*pq 
                jj = ii
                break
            end
        end
    end
    Δ = possible_steps[jj]
    (i,j) = (i+Δ[1], j+Δ[2])
    (i,j)
end

function get_coordinate(arena::Arena{T};rng=Random.default_rng()) where T <: Real
    i = rand(rng, 1:arena.ncols)
    j = rand(rng, 1:arena.nrows)
    (i,j)
end

function get_position(arena::Arena{T};rng=Random.default_rng()) where T <: Real
    i,j = get_coordinate(arena;rng=rng)
    get_position(i,j,arena)
end

function get_position(i::Int64, j::Int64, arena::Arena{T}) where T <: Real
    wc = arena.colsize/2
    wr = arena.rowsize/2
    [(i-1)*arena.colsize + wc, (j-1)*arena.rowsize+wr]
end

function get_head_direction(Δθ::T;rng=Random.default_rng()) where T <: Real
    rand(rng, [-Δθ, zero(T), Δθ])
end

function get_head_direction(Δθ::T,θ::T;rng=Random.default_rng(),p_stay::T=T(0.5)) where T <: Real
    pp = cumsum([(1 - p_stay)/2, p_stay, (1-p_stay)/2])
    cc = [-Δθ, zero(T), Δθ]
    ii = 2 # stop gap; if for some reason the below fails, default to staying
    for jj in 1:length(pp)
        if rand(rng) < pp[jj]
            ii = jj
            break
        end
    end
    cc[ii]
end

struct ViewField{T<:Real}

end

"""
Return a view field, in terms of an angular range, given the grid position `x` and `y`
"""
function (vf::ViewField{T})(x::Int64, y::Int64,θ::T) where T <: Real
    # just to it in a streight forward dumb way
    view_bins = zeros(Bool, 4, 8)
    if x == 1
        if y == 1
            if θ == 9π/4
                view_bins[1,1] = true
                view_bins[2,1] = true
            elseif θ == 3π/2
                view_bins[1,1:2] .= true
            elseif θ == 3π/2+π/4
                view_bins[1,2:end] .= true
                view_bins[2,1] .= true
            elseif θ == 0
                view_bins[1,3:end] .= true
                view_bins[2,2:end] .= true
            elseif θ == π/4
                view_bins[2,3:end] .= true
            elseif θ == π/2
                view_bins[1,4:end] .= true
                view_bins[3,]
                view_bins[2,4:end] .= true
            end

        end
    end

end

struct NavigationTrial{T<:Real} <: AbstractTrialStruct{T}
    min_num_steps::Int64
    max_num_steps::Int64
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    arena::Arena{T}
    angular_pref::AngularPreference{T}
end

# fallback
function NavigationTrial{T}(min_num_steps::Int64, max_num_steps::Int64, arena::Arena{T}, apref::AngularPreference{T})  where T <: Real
    NavigationTrial(min_num_steps, max_num_steps, [:view],[:position],arena, apref)
end

get_name(::Type{NavigationTrial{T}}) where T <: Real = :NavigationTrial

function signature(trial::NavigationTrial{T},h=zero(UInt32)) where T <: Real
    for q in [trial.min_num_steps, trial.max_num_steps, trial.inputs, trial.outputs]
        for ii in q
            h = crc32c(string(ii), h)
        end
    end
    h = signature(trial.arena,h)
    h = signature(trial.angular_pref,h)
    h
end

function get_circle_intersection(origin::Vector{T}, r::T, point::Vector{T}, θ::T) where T <: Real
    (a,b) = (cos(θ),sin(θ))
    (x0,y0) = origin
    (x1,y1) = point

    # find t
    A = a^2 + b^2
    B = 2*a*(x1-x0) + 2*b*(y1-y0)
    C = (x1-x0)^2 + (y1-y0)^2 - r^2
    t = (-B+sqrt(B^2 - 4*A*C))/(2*A)
    point + t*[a,b]
end

 function get_view(pos::Vector{T}, θ::T, arena::Arena{T};fov::T=T(π/2)) where T <: Real
    pos_center = [get_center(arena)...,]
    r = sqrt(sum(pos_center.^2))
    θr = T.([θ-fov/2, θ+fov/2])
    xq = zeros(T,2,2)
    xq[1,:] = get_circle_intersection(pos_center, r, pos, θr[1])
    xq[2,:] = get_circle_intersection(pos_center, r, pos, θr[2])
    xq .-= pos_center
    θq = atan.(xq[:,2], xq[:,1])
    # make sure we get the smallest range
    Qqm1 = extrema(θq)
    Δ1 = Qqm1[2] - Qqm1[1]
    θq2 = copy(θq)
    θq2[θq2.<0] .+= 2π
    Qqm2 = extrema(θq2)
    Δ2 = Qqm2[2] - Qqm2[1]
    if Δ1 < Δ2
        return Qqm1
    else
        return Qqm2
    end
 end

function get_view_old(pos::Vector{T}, θ::T,arena::Arena{T}) where T <: Real
    # just project onto a circle which inscribes the arena
    pos_center = get_center(arena)
    r = sqrt(sum(pos_center.^2))
    θr = [θ-π/4, θ+π/4]

    # head direction
    vp = [cos(θ) sin(θ)]

    #fov
    v = [cos.(θr) sin.(θr)]

    # distance to border
    dl = T(0.01)
    xp = zeros(T, 2, 2)
    for i in 1:2
        xy = [pos...]
        while norm(xy+dl*v[i,:]-pos_center) < r
            xy .+= dl*v[i,:]
        end
        xp[i,:] .= xy
    end

    xq = xp .- pos_center
    θq = atan.(xq[:,2], xq[:,1])
    # make sure we get the smallest range
    Qqm1 = extrema(θq)
    Δ1 = Qqm1[2] - Qqm1[1]
    θq2 = copy(θq)
    θq2[θq2.<0] .+= 2π
    Qqm2 = extrema(θq2)
    Δ2 = Qqm2[2] - Qqm2[1]
    if Δ1 < Δ2
        return Qqm1
    else
        return Qqm2
    end
end

function (trial::NavigationTrial{T})(;rng=Random.default_rng(),Δθstep::T=T(π/4), p_stay=T(1/3), p_hd=T(1/4), kwargs...) where T <: Real
    # random initiarange(-T(π), stop=T(π), step=π/4)li
    θf = range(zero(T), stop=T(2π), step=T(π/4))
    nsteps = rand(rng, trial.min_num_steps:trial.max_num_steps)
    position = zeros(T,2,nsteps)
    (i,j) = get_coordinate(trial.arena;rng=rng) 
    position[:,1] = get_position(i,j,trial.arena)
    viewf = zeros(T, length(trial.angular_pref.μ),nsteps)
    head_direction = zeros(T, length(trial.angular_pref.μ), nsteps)

    θ = rand(rng, θf)
    head_direction[:,1] = trial.angular_pref(θ)
    θq = get_view(position[:,1],θ, trial.arena;kwargs...)
    viewf[:,1] .= mean(trial.angular_pref(range(θq[1], stop=θq[2],length=10)),dims=2)

    Δθ = T.([-Δθstep, 0.0, Δθstep])
    for k in 2:nsteps
        θ += get_head_direction(Δθstep,θ;rng=rng,p_stay=p_stay) 
        i,j = get_coordinate(i,j,trial.arena,θ;rng=rng)
        position[:,k] = get_position(i,j,trial.arena)
        head_direction[:,k] = trial.angular_pref(θ)
        # get view angles
        θq = get_view(position[:,k],θ, trial.arena;kwargs...)
        viewf[:,k] .= mean(trial.angular_pref(range(θq[1], stop=θq[2],length=10)),dims=2)
    end
    position./=[trial.arena.ncols*trial.arena.colsize, trial.arena.nrows*trial.arena.rowsize]
    position .= 0.8*position .+ 0.05 # rescale from 0.05 to 0.85 to avoid saturation
    position, head_direction, viewf
end

num_inputs(trialstruct::NavigationTrial) = length(trialstruct.inputs)*length(trialstruct.angular_pref.μ)
num_outputs(trialstruct::NavigationTrial) = 2  # for position

function compute_error(trialstruct::NavigationTrial{T}, output::Array{T,3}, output_true::Array{T,3}) where T <: Real
    # we should differentiate depending on what the output is. If it is just position, an error is the deviation from the cell center
    # error for each position
    err = fill(T(NaN), size(output,2), size(output,3))
    for (i,(output_t, output_true_t)) in enumerate(zip(eachslice(output,dims=3), eachslice(output_true,dims=3)))
        # find the sequence length
        idxc = findfirst(output_true_t .> T(0.05))
        idx1 = idxc.I[2]
        idx2 = findfirst(dropdims(sum(output_true_t,dims=1),dims=1) .== zero(T))
        if idx2 === nothing
            idx2 = size(output_t,2)
        else
            idx2 = idx2 - 1
        end
        err[1:idx2-idx1+1,i] = dropdims(sum(abs2, output_t[:,idx1:idx2]-output_true_t[:,idx1:idx2],dims=1),dims=1)
    end
    err
end

function performance(trialstruct::NavigationTrial{T}, output::Array{T,3}, output_true::Array{T,3};require_fixation=false) where T <: Real
    err = compute_error(trialstruct, output, output_true)
    # correct if the error is less than half the cell width
    ppq = zeros(T,size(err,1))
    nq = fill(0, size(err,1))
    # need to scale the size appropriately
    wc = 0.8/trialstruct.arena.ncols
    wr = 0.8/trialstruct.arena.nrows
    sΔ = (wc/2)^2 + (wr/2)^2
    for _err in eachcol(err)
        idx = isfinite.(_err)
        ppq[idx] .+= _err[idx] .< sΔ
        nq[idx] .+= 1
    end
    ppq./nq 
end

function generate_trials(trial::NavigationTrial{T}, ntrials::Int64,dt::T; rng=Random.default_rng(), rseed=1, Δθstep::T=T(π/4), fov::T=T(π/2),p_stay=T(1/3), p_hd=T(1/4)) where T <: Real
    args = [(:ntrials, ntrials),(:dt, dt), (:rng, rng), (:rseed, rseed), (:Δθstep, Δθstep),
            (:fov, fov),(:p_stay, p_stay),(:p_hd, p_hd)]
    defaults = Dict{Symbol,Any}(:Δθstep=>T(π/4), :fov=>T(π/2),:p_stay=>T(1/3),:p_hd=>T(1/4))
    h = signature(trial)
    for (k,v) in args
        if !(k in keys(defaults)) || v != defaults[k]
            h = CRC32c.crc32c(string(v), h)
        end
    end
    pushfirst!(args, (:trialstruct, trial))
    Random.seed!(rng, rseed)
    ninputs = length(trial.inputs)*length(trial.angular_pref.μ)
    noutputs = length(trial.outputs)*2 # for position
    max_nsteps = trial.max_num_steps
    TrialIterator(
        function data_provider()
            input = zeros(T, ninputs, max_nsteps, ntrials)
            output = zeros(T, noutputs, max_nsteps, ntrials)
            output_mask = zeros(T, noutputs, max_nsteps, ntrials)
            for i in 1:ntrials
                position, head_direction,viewfield = trial(;rng=rng,Δθstep=Δθstep,fov=fov)
                offset = 0
                if :view in trial.inputs
                    input[offset+1:offset+size(viewfield,1), 1:size(viewfield,2),i]  .= viewfield
                    offset += size(viewfield,1)
                end
                if :head_direction in trial.inputs
                    input[offset+1:offset+size(head_direction,1), 1:size(head_direction,2),i]  .= head_direction
                    offset += size(head_direction,1)
                end
                if :position in trial.outputs
                    output[1:size(position,1), 1:size(position,2),i] .= position
                end
                output_mask[:,1:size(position,2),i] .= one(T)
            end
            input,output,output_mask
        end,NamedTuple(args), h)
end