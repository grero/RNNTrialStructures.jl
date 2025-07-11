abstract type AbstractTrialStruct{T<:Real} end
abstract type AbstractPositionPatternTrial{T<:Real} <: AbstractTrialStruct{T} end

abstract type Trial{T<:Real} end

get_name(::Type{T})  where T <: AbstractTrialStruct{T2} where T2 <: Real = error("Not implemented")
get_name(x::T)  where T <: AbstractTrialStruct{T2} where T2 <: Real = get_name(T)

function create_mask(i, go_cue_onset, grace_period, modifier::T,pmodifier::T=one(T)) where T <: Real
    q = zero(T)
    if i < go_cue_onset
        q = pmodifier
    elseif i < go_cue_onset+grace_period
        q = zero(T)
    else
        q = modifier
    end
    return q
end

penalty(::Type{AbstractTrialStruct}) = error("Not implemented")
# general fallback
penalty(trial::T) where T<: AbstractTrialStruct = penalty(T)

"""
Geenerate a new trial ID for the given trial type
"""
get_trialid(::Type{AbstractTrialStruct}) = error("Not implemented")

struct AngularPreference{T<:Real}
    μ::Vector{T}
    σ::T
    a::T
end

function signature(apref::AngularPreference{T}, h=zero(UInt32)) where T <: Real
    for _μ in apref.μ
        h = crc32c(string(_μ), h)
    end
    h = crc32c(string(apref.σ),h)
    h = crc32c(string(apref.a),h)
    h
end

function (apref::AngularPreference{T})(θ::T) where T <: Real
    b = T(2π)
    κ = apref.σ
    z = exp(κ)
    apref.a.*exp.(κ*cos.(θ .- apref.μ))./z
end

struct PlaceCells{T<:Real}
    μ::Vector{Tuple{T, T}}
    σ::Vector{T}
end

"""
    (pc::PlaceCells)(pos::Tuple{Float64,Float64})

Return the value of each place cell for the position pos
"""
function (pc::PlaceCells{T})(pos::Tuple{T,T}) where T <: Real
    d = map(x->sum(abs2, pos .- x),pc.μ)
    exp.(-T(0.5)*d./pc.σ.^2)
end

struct CategoryTrial{T<:Real} <: AbstractTrialStruct{T}
    input::Matrix{T}
    output::Matrix{T}
    go_cue_onset::Int64
end

num_inputs(trial::CategoryTrial) = size(trial.input,1)

function penalty(::Type{T}) where T <: CategoryTrial
    func(x,y) = sum(abs2, x-y)
end

get_trialid(trial::CategoryTrial) = rand(1:size(trial.output,1))

function matches(trial::CategoryTrial, output, x::Int64, idx::CartesianIndex)
    idx.I[1] == x
end

function CategoryTrial(::Type{T};nlocations=4, nsteps=100, go_cue_onset=50, target_onset=20, target_duration=5, go_cue_duration=nsteps-go_cue_onset+1) where T <: Real
    target_input = zeros(T, nlocations, nsteps)
    target_input[:,target_onset:target_onset+target_duration-1] .= one(T) 
    go_cue_input = zeros(T, 1, nsteps)
    go_cue_input[go_cue_onset:go_cue_onset+go_cue_duration-1] .= one(T) 
    inputs = [target_input;go_cue_input]
    output = zeros(T, nlocations, nsteps)
    output[:,go_cue_onset+1:end] .= one(T) 
    CategoryTrial(inputs, output, go_cue_onset)
end

CategoryTrial(;kvs...) = CategoryTrial(Float64;kvs...)

function (trial::CategoryTrial)(target_id::Int64)
    nlocations = size(trial.output,1)
    target_mask = repeat([[1:nlocations;].==target_id;true],1,1)
    target_mask.*trial.input, target_mask[1:nlocations].*trial.output
end

function (trial::CategoryTrial{T})(target_id::Int64, go_cue_onset::Int64) where T <: Real
    nsteps = size(trial.input,2)
    nlocations = size(trial.output,1)
    target_mask = repeat([1:nlocations;].==target_id,1,1)
    go_cue_input = permutedims(repeat([ifelse(i < go_cue_onset, 0.0, 1.0) for i in 1:nsteps],1,1))
    input = [target_mask.*trial.input[1:nlocations,:];go_cue_input]
    output = target_mask.*go_cue_input
    input, output
end

struct PositionTrial{T<:Real} <: AbstractTrialStruct{T}
    input::Matrix{T}
    output::Matrix{T}
    go_cue_onset::Int64
    positions::Vector{Tuple{T, T}}
    pc::PlaceCells
    go_cue_duration::Int64
end

# TODO: Add trial with an ITI

struct PositionPatternTrial{T<:Real} <: AbstractPositionPatternTrial{T}
    input::Matrix{Float64}
    output::Matrix{Float64}
    go_cue_onset::Int64
    positions::Vector{Tuple{Float64, Float64}}
    pc::PlaceCells
    go_cue_duration::Int64
end

struct ITIPositionPatternTrial{T<:Real} <: AbstractPositionPatternTrial{T}
    input::Matrix{T}
    output::Matrix{T}
    go_cue_onset::Int64
    positions::Vector{Tuple{T, T}}
    pc::PlaceCells
    go_cue_duration::Int64
    iti_onset::Int64
    iti_duration::Int64
end

function PositionPatternTrial(trialstruct::T;kvs...) where T <: AbstractPositionPatternTrial
    args = Any[]
    for k in fieldnames(T)
        if k in keys(kvs)
            push!(args, kvs[k])
        else
            push!(args, getfield(trialstruct, k))
      end
  end
  T(args...)
end

num_inputs(trial::PositionTrial) = length(trial.positions)
num_inputs(trial::AbstractPositionPatternTrial) = length(trial.positions)

penalty(::Type{T}) where T <: PositionTrial = (func(x,y) = sum(abs2, x-y))
penalty(::Type{T}) where T <: AbstractPositionPatternTrial = (func(x,y) = sum(abs2, x-y))

"""
Return a function to compute the penalty using vector averge.

This function interprets the first input as weights attached to each of
the place field positions. The inferred position is then just he weighted average
of those positions, and is compared to the position `y` when computing the penalty.
"""
function penalty(trialstruct::PositionTrial)
    pos = trialstruct.pc.μ
    function func(x::Vector{Float64},y)
        # get the vector average
        xs = x .+ 0.01 # to avoid blowup if all x are close to zero
        zz = [_x.*_pos for (_x,_pos) in zip(xs,pos)]
        z = reduce((z1,z2)->z1.+z2, zz)./sum(xs)
        sum(abs2, z .- y)
    end
end

get_trialid(trial::PositionTrial) = rand(trial.positions)
get_trialid(trial::AbstractPositionPatternTrial) = rand(trial.positions)

function PositionTrial(positions::Vector{Tuple{T, T}}, μ::Vector{Tuple{T, T}},σ::Vector{T};nsteps=100, go_cue_onset=50, target_onset=5, target_duration=5, go_cue_duration=5, kvs...) where T <: Real
    nlocations = length(μ)
    target_input = zeros(T, nlocations, nsteps)
    target_input[:,target_onset:target_onset+target_duration-1] .= one(T) 
    go_cue_input = zeros(T, 1, nsteps)
    go_cue_input[go_cue_onset:go_cue_onset+go_cue_duration-1] .=  one(T)
    inputs = [target_input;go_cue_input]
    output = zeros(T, nlocations, nsteps)
    output[:,go_cue_onset+1:end] .= one(T) 
    PositionTrial(inputs, output, go_cue_onset,positions, PlaceCells(μ, σ),go_cue_duration)
end


function PositionPatternTrial(positions::Vector{Tuple{T, T}}, μ::Vector{Tuple{T, T}},σ::Vector{T};nsteps=100, go_cue_onset=50, target_onset=5, target_duration=5, go_cue_duration=5, kvs...) where T <: Real
    nlocations = length(μ)
    target_input = zeros(t, nlocations, nsteps)
    target_input[:,target_onset:target_onset+target_duration-1] .= one(T) 
    go_cue_input = zeros(T, 1, nsteps)
    go_cue_input[go_cue_onset:go_cue_onset+go_cue_duration-1] .= one(T)
    timeinput = zeros(T, 1, nsteps)
    inputs = [target_input;go_cue_input;timeinput]
    output = zeros(T, nlocations, nsteps)
    output[:,go_cue_onset+1:end] .= one(T) 
    PositionPatternTrial(inputs, output, go_cue_onset,positions, PlaceCells(μ, σ),go_cue_duration)
end

function ITIPositionPatternTrial(positions::Vector{Tuple{T, T}}, μ::Vector{Tuple{T, T}},σ::Vector{T};nsteps=100, go_cue_onset=50, target_onset=5, target_duration=5, go_cue_duration=5, iti_onset=45, iti_duration=5,kvs...) where T <: Real
    pc = PlaceCells(μ,σ)
    fixation_pattern = pc((zero(T), zero(T)))
    nlocations = length(μ)
    target_input = zeros(T, nlocations, nsteps)
    target_input[:,target_onset:target_onset+target_duration-1] .= one(T) 
    go_cue_input = zeros(T, 1, nsteps)
    go_cue_input[go_cue_onset:go_cue_onset+go_cue_duration-1] .= one(T) 
    inputs = [target_input;go_cue_input]
    output = zeros(T, nlocations, nsteps)
    output[:,go_cue_onset+1:iti_onset-1] .= one(T) 
    ITIPositionPatternTrial(inputs, output, go_cue_onset,positions, PlaceCells(μ, σ),go_cue_duration, iti_onset, iti_duration)
end

function (trial::PositionTrial)(input,output, pos::Tuple{T, T}, go_cue_onset::Int64;kvs...) where T <: Real
    nlocations,nsteps = size(trial.output)
    go_cue_duration = trial.go_cue_duration
    posr = trial.pc(pos)
    for i in 1:nsteps
        input[end,i] = ifelse(go_cue_onset <= i < go_cue_onset+go_cue_duration,one(T),zero(T))
        oo = ifelse(i >=go_cue_onset, one(T), zero(T))
        for j in 1:nlocations
            input[j,i] = posr[j]*trial.input[j,i]
        end
        output[:,i] = pos.*oo
    end
end

function (trial::PositionTrial)(pos::Tuple{T, T}, go_cue_onset::Int64;kvs...) where T <: Real
    nlocations,nsteps = size(trial.output)
    go_cue_duration = trial.go_cue_duration
    go_cue_input = permutedims(repeat([ifelse(go_cue_onset <= i < go_cue_onset+go_cue_duration,one(T),zero(T)) for i in 1:nsteps],1,1))
    go_cue_output = permutedims(repeat([ifelse(i >=go_cue_onset, one(T), zero(T)) for i in 1:nsteps],1,1))
    posr = trial.pc(pos)
    input = [posr.*trial.input[1:nlocations,:];go_cue_input]
    output = pos.*go_cue_output
    input, output
end

function (trial::PositionPatternTrial)(pos::Tuple{T, T}, go_cue_onset::Int64;kvs...) where T <: Real
    nlocations,nsteps = size(trial.output)
    go_cue_duration = trial.go_cue_duration
    go_cue_input = permutedims(repeat([ifelse(go_cue_onset <= i < go_cue_onset+go_cue_duration,one(T),zero(T)) for i in 1:nsteps],1,1))
    go_cue_output = permutedims(repeat([ifelse(i >=go_cue_onset, one(T), zero(T)) for i in 1:nsteps],1,1))
    posr = trial.pc(pos)
    timeinput = permutedims(repeat([(i-1)*3.0/(nsteps-1) for i in 1:nsteps],1,1))
    input = [posr.*trial.input[1:nlocations,:];go_cue_input;timeinput]
    output = posr.*go_cue_output
    input, output
end

function (trial::PositionPatternTrial)(input,output, pos::Tuple{T, T}, go_cue_onset::Int64;kvs...) where T <: Real
    nlocations,nsteps = size(trial.output)
    go_cue_duration = trial.go_cue_duration
    posr = trial.pc(pos)
    for i in 1:nsteps
        input[end,i] = ifelse(go_cue_onset <= i < go_cue_onset+go_cue_duration,one(T),zero(T))
        oo = ifelse(i >=go_cue_onset, one(T),zero(T))
        for j in 1:nlocations
            input[j,i] = posr[j]*trial.input[j,i]
            output[j,i] = posr[j]*oo
        end
    end
end


function (trial::ITIPositionPatternTrial)(pos::Tuple{T, T}, go_cue_onset::Int64=trial.go_cue_onset;kvs...) where T <: Real
    fixation = trial.pc((zero(T), zero(T)))
    nlocations,nsteps = size(trial.output)
    go_cue_duration = trial.go_cue_duration
    go_cue_input = permutedims(repeat([ifelse(go_cue_onset <= i < go_cue_onset+go_cue_duration,1.0,0.0) for i in 1:nsteps],1,1))
    go_cue_output = permutedims(repeat([ifelse(i >=go_cue_onset,one(T), zero(T)) for i in 1:nsteps],1,1))
    posr = trial.pc(pos)
    timeinput = permutedims(repeat([(i-1)*T(3.0)/(nsteps-1) for i in 1:nsteps],1,1))
    input = [posr.*trial.input[1:nlocations,:];go_cue_input;timeinput]
    # TODO present fixation here
    output = [posr.*go_cue_output[:,1:trial.iti_onset] fixation.*go_cue_output[:,trial.iti_onset+1:end]]
    input, output
end

"""
    matches(trial::PositionTrial, output, x, idx::CartesianIndex)

Return whether `x` matches output[idx] by comparing the distance of `output[idx]`
"""
function matches(trial::PositionTrial, output, x::Tuple{T, T}, idx::CartesianIndex) where T <: Real
    # we want to compare the entire output
    pos = trial.pc.μ
    oidx = output[:,idx.I[2]]

    #compute vector average
    zz = [_x.*pos for (_x,pos) in zip(oidx,pos)]
    z = reduce((z1,z2)->z1.+z2, zz)
    d = sum(abs2, z .- x)
    m = true
    for _x in trial.positions
        _d = sum(abs2, z .- _x)
        if _d < d
            m = false
            break
        end
    end
    m
end

get_output(trial::PositionTrial, input::Tuple{T, T}) where T <: Real = input
get_output(trial::PositionPatternTrial, input::Tuple{T, T})  where T <: Real = trial.pc(input)

function matches(trial::PositionPatternTrial, output, x::Tuple{T, T}, idx::CartesianIndex) where T <: Real
    # we want to compare the entire output
    oidx = output[:,idx.I[2]]
    y = trial.pc(x)
    d = sum(abs2, oidx .- y)
    m = true
    for _x in trial.positions
        _y = trial.pc(_x)
        _d = sum(abs2, oidx .- _y)
        if _d < d
            m = false
            break
        end
    end
    m
end

struct AngleTrial{T<:Real} <: AbstractTrialStruct{T}
    input::Matrix{T}
    output::Matrix{T}
    go_cue_onset::Int64
    angles::Vector{T}
end

num_inputs(trial::AngleTrial) = length(trial.angles)

#TODO: This is problematic because it doesn't put any absolute constrain on x or y
penalty(::Type{T}) where T <: AngleTrial = (func(x,y) = sum(abs2, x - y))
get_trialid(trial::AngleTrial) = rand(trial.angles)

function matches(trial::AngleTrial, output, x, idx::CartesianIndex)
    # The output should be the cosine and sine of the angle
    xx = [cos(x);sin(x)]
    oidx = output[:,idx.I[2]]
    d = sum(abs2, oidx .- xx)
    m = true
    for _x in trial.angles
        _d = sum(abs2, oidx .- xx)
        if _d < d
            m = false
            break
        end
    end
    m
end

function AngleTrial(angles::Vector{Float64};nsteps=100, go_cue_onset=50, target_onset=5, target_duration=5)
    target_input = fill(0.0, 2, nsteps)
    target_input[:,target_onset:target_onset+target_duration-1] .= 1.0
    go_cue_input = fill(0.0, 1, nsteps)
    go_cue_input[go_cue_onset:end] .= 1.0
    inputs = [target_input;go_cue_input]
    output = fill(0.0, 2, nsteps)
    output[:,go_cue_onset+1:end] .= 1.0
    AngleTrial(inputs, output, go_cue_onset, angles)
end

(trial::AngleTrial)(θ) = trial(θ, trial.go_cue_onset)

function (trial::AngleTrial)(θ::Float64, go_cue_onset::Int64)
    q = [cos(θ), sin(θ)]
    nsteps = size(trial.input,2)
    go_cue_input = permutedims(repeat([ifelse(i < go_cue_onset, 0.0, 1.0) for i in 1:nsteps],1,1))
    input = [q.*trial.input[1:1,:];go_cue_input]
    output = q.*go_cue_input
    input, output
end

struct TwoAngularInputs{T<:Real} <: AbstractTrialStruct{T}
    length::T
    go_cue_onset::T
    stim1_onset::T
    stim2_onset::T
    stim_duration::T
    response1_onset::T
    response2_onset::T
    response_duration::T
    angular_pref::AngularPreference{T}
end

num_outputs(trial::TwoAngularInputs) = length(trial.angular_pref.μ)+1
num_inputs(trial::TwoAngularInputs) = num_outputs(trial) + 1

function penalty(::Type{TwoAngularInputs{T}}) where T <: Real
   func(x,y)  = sum(abs2, x-y;init=zero(T))
end

"""
    matches(trial::TwoAngularInputs{T}, output, θ::T, idx::CartesianIndex) where T <: Real

The output matches the angle `θ` at index `idx` if the weighted output is closer to the
actual `θ` than the average distance between the input preferences.
"""
function matches(trial::TwoAngularInputs{T}, output, θ::T, idx::CartesianIndex) where T <: Real
    Δ = Float32(2π)/length(trial.angular_pref.μ)
    rr = output[:,idx.I[2]]'*trial.angular_pref.μ
    return one(T) - cos(rr-θ) < cos(Δ)
end

"""
    matches(trial::TwoAngularInputs{T}, target_output::Matrix{T}, output::Matrix{T})

Return true if both outputs match the target output during the output duration.
"""
function matches(trial::TwoAngularInputs{T}, target_output::Matrix{T}, output::Matrix{T};dt::T=T(20.0)) where T <: Real
    Δ = T(2π)/length(trial.angular_pref.μ)
    cΔ = cos(Δ)
    θ1 = get_response(trial, target_output, trial.response1_onset)
    θ1p = get_response(trial, output, trial.response1_onset)
    θ2 = get_response(trial, target_output, trial.response2_onset)
    θ2p = get_response(trial, output, trial.response2_onset)

    return cos(θ1-θ1p) >= cΔ && cos(θ2-θ2p) >= cΔ
end

"""
    get_response(trial::TwoAngularInputs{T}, output::Matrix{T}, idx::Int64)

Return the angular output at time step `idx`
"""
function get_response(trial::TwoAngularInputs{T}, output::Matrix{T}, idx::Int64) where T <: Real
    idx0 = idx
    idx1 = idx0 + trial.response_duration-1
    yy = dropdims(mean(output[:, idx0:idx1],dims=2),dims=2)
    θ = yy'*trial.angular_pref.μ
    θ /= sum(yy)
    θ
end

function get_response(trial::TwoAngularInputs{T}, output::Matrix{T}, t::T;dt::T=T(20.0)) where T <: Real
    tt = range(zero(T), step=dt, stop=trial.length-dt)
    idx0 = searchsortedfirst(tt, t)
    idx1 = searchsortedlast(tt, t+trial.response_duration)
    yy = dropdims(mean(output[:, idx0:idx1],dims=2),dims=2)
    θ = yy'*trial.angular_pref.μ
    θ /= sum(yy)
    θ
end


function (trial::TwoAngularInputs{T})(θ1::T, θ2::T, go_cue_onset=trial.go_cue_onset;dt=T(20.0)) where T <: Real
    N = round(Int64, trial.length/dt)
    nout = length(trial.angular_pref.μ)
    nin = nout + 1
    
    input = zeros(T,nin,N)
    output = zeros(T, nout,N)
    stim1_onset = round(Int64, trial.stim1_onset/dt)
    stim1_offset = round(Int64, (trial.stim1_onset+trial.stim_duration)/dt)
    stim2_onset = round(Int64, trial.stim2_onset/dt)
    stim2_offset = round(Int64, (trial.stim2_onset+trial.stim_duration)/dt)

    input[1:end-1,stim1_onset:stim1_offset] .= trial.angular_pref(θ1)[:,1:1]
    input[1:end-1,stim2_onset:stim2_offset] .= trial.angular_pref(θ2)[:,1:1]
    # fixation
    _go_cue_onset = round(Int64, go_cue_onset/dt)
    input[end, 1:_go_cue_onset] .= one(T)

    # response
    response1_onset = round(Int64, trial.response1_onset/dt)
    response1_offset = round(Int64, (trial.response1_onset+trial.response_duration)/dt)
    response2_onset = round(Int64, trial.response2_onset/dt)
    response2_offset = round(Int64, (trial.response2_onset+trial.response_duration)/dt)
    output[:,response1_onset:response1_offset] .= trial.angular_pref(θ1)[:,1:1]
    output[:,response2_onset:response2_offset] .= trial.angular_pref(θ2)[:,1:1]
    output .= 0.6*output .+ 0.2 # output go from 0.2 to 0.8
    input, output
end

function get_trialid(trialstruct::TwoAngularInputs{T};rng=Random.default_rng()) where T <: Real
    T(2π)*rand(rng, T), T(2π)*rand(rng, T)
end

struct MultipleAngleTrial{T<:Real} <: AbstractTrialStruct{T}
    input_onset::Vector{Int64}
    input_offset::Vector{Int64}
    response_onset::Vector{Int64}
    response_offset::Vector{Int64}
    nangles::Int64
    tdim::Int64
    dt::T
    preference::AngularPreference{T}
end

get_name(::Type{MultipleAngleTrial{T}}) where T <: Real = :MultipleAngleTrial

function signature(trial::MultipleAngleTrial{T},h=zero(UInt32)) where T <: Real
    for q in [trial.input_onset, trial.input_offset, trial.response_onset, trial.response_offset]
        for ii in q
            h = crc32c(string(ii), h)
        end
    end
    h = crc32c(string(trial.nangles),h)
    h = crc32c(string(trial.tdim),h)
    h = crc32c(string(trial.dt),h)
    h = signature(trial.preference,h)
    h
end

function MultipleAngleTrial(first_onset::T, input_duration::T, delays::Vector{T}, output_duration::T, nangles::Int64, dt::T, preference::AngularPreference{T}) where T <: Real
    nn = length(preference.μ)

    input_duration = round(Int64, input_duration/dt)
    output_duration = round(Int64, output_duration/dt)
    input_onset = Vector{Int64}(undef, nangles)
    input_offset = Vector{Int64}(undef, nangles)
    response_onset = Vector{Int64}(undef, nangles)
    response_offset = Vector{Int64}(undef, nangles)
    tdim = round(Int64, first_onset/dt) 
    for (ii,delay) in enumerate(delays)
        input_onset[ii] = tdim 
        tdim += input_duration
        input_offset[ii] = tdim-1
        tdelay = round(Int64, delay/dt)
        tdim += tdelay-1
    end
    for ii in 1:nangles
        response_onset[ii] = tdim
        tdim += output_duration
        response_offset[ii] = tdim-1
    end
    MultipleAngleTrial(input_onset, input_offset, response_onset, response_offset,nangles, tdim, dt, preference)
end

function penalty(trialstruct::MultipleAngleTrial{T}) where T <: Real
     function func(x,y)
        sum(abs2, x-y,init=zero(T))
    end
end

function get_go_cue_onset(trial::MultipleAngleTrial{T};dt=T(20.0)) where T <: Real
    round(Int64, (trial.input_onset + trial.nangles*trial.input_duration + sum(trial.delay))/dt)
end

function get_output_mask(trial::MultipleAngleTrial{T},go_cue_onset=zero(T);pre_cue_multiplier=T(1.0),
                                                                            post_cue_multiplier=pre_cue_multiplier,
                                                                            grace_period=T(0.0)) where T <: Real
    nn = trial.nangles
    nsteps = trial.tdim 
    _grace_period = round(Int64, grace_period/tria.dt)
    _go_cue_onset = trial.response_onset[1] + round(Int64, go_cue_onset/dt)
    output_mask = zeros(T, 1, nsteps)
    output_mask[1,:] = [create_mask(i, _go_cue_onset, _grace_period, post_cue_multiplier, pre_cue_multiplier) for i in 1:nsteps]
    output_mask
end

num_inputs(trial::MultipleAngleTrial) = length(trial.preference.μ)+1
num_outputs(trial::MultipleAngleTrial) = length(trial.preference.μ)+1
num_stimuli(trial::MultipleAngleTrial) = trial.nangles

get_trialid(trial::MultipleAngleTrial{T};rng=Random.default_rng()) where T <: Real = T(2π).*rand(rng, T,trial.nangles)

function get_trialid(trial::MultipleAngleTrial{T}, constraint_factor::T;rng=Random.default_rng()) where T <: Real
    if constraint_factor == zero(T)
        return get_trialid(trial;rng=rng)
    end
    if trial.nangles > 2
        error("Constraints only work for 2 angles currently")
    end
    θ1 = T(2π)*rand(rng, T)
    if trial.nangles == 1
        return [θ1]
    end
    0.0 <= constraint_factor <= 1.0 || error("Constraint factor should be beween 0 and 1")
    # only really works for two inputs
    a = T(π)
    d = (T(2.0) - 2*constraint_factor)*rand(rng,T) + constraint_factor 
    s = rand(rng, [-T(1.0),T(1.0)])
    θ2 = mod(θ1 + a*d*s,T(2π))
    [θ1,θ2]
end

function get_nsteps(trial::MultipleAngleTrial{T},n::Int64, dt,go_cue_onset=zero(T)) where T <: Real
    input_duration = round(Int64, trial.input_duration/dt)
    output_duration = round(Int64, trial.output_duration/dt)
    tdim = round(Int64, trial.input_onset/dt) 
    for delay in trial.delay 
        tdim += input_duration
        tdelay = round(Int64, delay/dt)
        tdim += tdelay
    end
    tdim += round(Int64, go_cue_onset/dt)
    for _ in 1:n
        tdim += output_duration
    end
    tdim
end

function (trial::MultipleAngleTrial{T})(θ::Vector{T},go_cue_onset::Int64=0) where T <: Real
    nt = length(θ)
    nn = length(trial.preference.μ)
    nsteps = trial.tdim
    nsteps += go_cue_onset
    input = zeros(T, nn+1, nsteps)
    output = fill(T(0.05), nn+1, nsteps)
    for (ii,_θ) in enumerate(θ)
        idx1 = trial.input_onset[ii]
        idx2 = trial.input_offset[ii]
        input[1:end-1,idx1:idx2] .= trial.preference(_θ)
        idx1 = trial.response_onset[ii] + go_cue_onset
        idx2 = trial.response_offset[ii] + go_cue_onset

        output[1:end-1,idx1:idx2] .+= 0.75*trial.preference(_θ)
    end
    # set fixation input
    input[end, 1:trial.response_onset[1]+go_cue_onset-1] .= 1.0
    # have the model reproduce the fixation output as well
    # ideally we shouldn't have this, but it could help the model earn faster
    output[end, 1:trial.response_onset[1]+go_cue_onset-1] .+= 0.75
    input, output
end

function matches(trial::MultipleAngleTrial{T}, output::AbstractMatrix{T}, output_true::AbstractMatrix{T}) where T <: Real
    angular_pref = trial.preference
    θ = angular_pref.μ
    nθ = length(θ)
    Δ = Float32(2π)/length(angular_pref.μ)
    pp = fill(false, trial.nangles)
    for jj in axes(pp,1)
        idx1 = trial.response_onset[jj]
        idx2 = trial.response_offset[jj]
        rr = dropdims(mean(output[1:nθ,idx1:idx2],dims=2),dims=2)
        sumrr = sum(rr)
        θrr = atan(sum(rr.*sin.(θ))/sumrr, sum(rr.*cos.(θ))/sumrr) 
        rr_true = mean(output_true[1:nθ,idx1:idx2],dims=2)

        sumrr_true = sum(rr_true)
        θrr_true = atan(sum(rr_true.*sin.(θ))/sumrr_true, sum(rr_true.*cos.(θ))/sumrr_true) 

        pp[jj] = cos(θrr - θrr_true) > cos(Δ)
    end
    return pp
end


function matches(trial::MultipleAngleTrial{T}, output::AbstractArray{T,3}, output_true::AbstractArray{T,3};require_fixation=true) where T <: Real
    angular_pref = trial.preference
    θ = angular_pref.μ
    nθ = length(θ)
    θ = [θ;zero(T)]
    sθ = sin.(θ)
    cθ = cos.(θ)
    Δ = Float32(2π)/length(angular_pref.μ)
    pp = zeros(T, trial.nangles)
    W = zeros(T, size(output)...)
    for jj in axes(pp,1)
        fill!(W, zero(T))
        idx1 = trial.response_onset[jj]
        idx2 = trial.response_offset[jj]
        W[1:nθ, idx1:idx2,:] .= one(T)
        sumrrn = sum(W, dims=2)
        rr = mean(W.*output, dims=2) # this creates NaNs
        sumrr = sum(rr, dims=1)
        θrr = atan.(sum(rr.*sθ,dims=1)./sumrr, sum(rr.*cθ,dims=1)./sumrr)

        rr_true = mean(W.*output_true, dims=2)
        sumrr_true = sum(rr_true)
        θrr_true = atan.(sum(rr_true.*sθ,dims=1)./sumrr_true, sum(rr_true.*cθ,dims=1)./sumrr_true)

        if require_fixation
            fill!(W,zero(T))
            # allow 5 points before response onset
            W[1:nθ,1:idx1-5,:] .= one(T)
            rrt = maximum(maximum(W.*output,dims=2),dims=1)
            pp[jj] = mean((rrt .< 0.2f0).*(cos.(θrr .- θrr_true) .> cos(Δ)))
        else
            pp[jj] = mean(cos.(θrr .- θrr_true) .> cos(Δ))
        end
    end
    return pp
end

