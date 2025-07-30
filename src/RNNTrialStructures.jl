module RNNTrialStructures
using Random
using StatsBase
using CRC32c

include("trialstructures.jl")

# TODO: This probably be made into a macro to make things more much simple
mutable struct TrialIterator
    data_provider::Function
    args::NamedTuple
    arghash::UInt32
    state::Int64
end

TrialIterator(func,args, h) = TrialIterator(func, args, h,0)


function (trial::TrialIterator)()
    trial.state += 1
    trial.data_provider()
end

function Base.iterate(trial::TrialIterator, state)
    trial.state = state
    trial.state += 1
    trial.data_provider(), trial.state
end

function Base.iterate(trial::TrialIterator)
    trial.data_provider(),0
end

function generate_trials(trialstruct::MultipleAngleTrial{T}, ntrials::Int64;randomize_go_cue=true, randomize_grace_period=false, go_cue_onset_min::T=zero(T), go_cue_onset_max::T=go_cue_onset_min,
                                                      grace_period_min::T=zero(T), grace_period_max=grace_period_min,
                                                      post_cue_multiplier::T=T(2.0), pre_cue_multiplier::T=one(T),stim_onset_min::Vector{T}=zeros(T,trialstruct.nangles), stim_onset_max::Vector{T}=stim_onset_min,
                                                      stim_onset_step::Vector{T}=fill(trialstruct.dt, trialstruct.nangles), σ=zero(T), constraint_factor::T=T(0.0),rng=Random.default_rng(), rseed=1234) where T <: Real

    args = [(:ntrials, ntrials),(:randomize_go_cue, randomize_go_cue),(:randomize_grace_period, randomize_grace_period),(:go_cue_onset_min, go_cue_onset_min),
            (:go_cue_onset_max, go_cue_onset_max),(:stim_onset_min, stim_onset_min),(:stim_onset_max, stim_onset_max),(:stim_onset_step, stim_onset_step),(:grace_period_min, grace_period_min), (:grace_period_max, grace_period_max),
            (:post_cue_multiplier, post_cue_multiplier),(:pre_cue_multiplier,pre_cue_multiplier),(:σ, σ), (:constraint_factor, constraint_factor),(:rng, rng),(:rseed, rseed)]
    defaults = Dict(:stim_onset_min => zeros(T, trialstruct.nangles),
                    :stim_onset_max => zeros(T, trialstruct.nangles),
                    :stim_onset_step => fill(trialstruct.dt, trialstruct.nangles))
    #generate a hash of the arguments
    h = signature(trialstruct)
    for (k,v) in args
        if !(k in keys(defaults)) || v != defaults[k]
            h = crc32c(string(v),h)
        end
    end
    # add the first argument after
    pushfirst!(args, (:trialstruct, trialstruct))
    Random.seed!(rng, rseed)
    randomize_stim_2 = any(stim_onset_max .> stim_onset_min)
    TrialIterator(function data_provider()
        dt = trialstruct.dt
        ninput = num_inputs(trialstruct)
        noutput = num_outputs(trialstruct)
        nsteps = trialstruct.tdim
        # account for the fact that varying go-cue could extend the trial
        nsteps += round(Int64, ceil(go_cue_onset_max/dt))
        input = zeros(T, ninput, nsteps, ntrials)
        output = fill(T(0.05), noutput, nsteps, ntrials)
        output_mask = zeros(T, noutput, nsteps, ntrials)
        # by default, the go-cue onset coincides  with the first response onset
        go_cue_onset = trialstruct.response_onset[1]-1
        stim_onset_Δ = fill(0, trialstruct.nangles)
        for i in 1:ntrials
            trialid = get_trialid(trialstruct,constraint_factor;rng=rng)
            if randomize_go_cue
                _go_cue_onset = rand(round(Int64, go_cue_onset_min/dt):round(Int64,go_cue_onset_max/dt))
            else
                _go_cue_onset = 0 
            end
            if randomize_stim_2
                for j in 1:trialstruct.nangles
                    _step = round(Int64, stim_onset_step[j]/dt)
                    stim_onset_Δ[j] = rand(round(Int64, stim_onset_min[j]/dt):_step:round(Int64,stim_onset_max[j]/dt))
                end
            end
            if randomize_grace_period
                _grace_period = rand(round(Int64, grace_period_min/dt):round(Int64, grace_period_max/dt))
            else
                _grace_period = round(Int64, grace_period_min/dt)
            end
            _input, _output = trialstruct(trialid, _go_cue_onset, stim_onset_Δ)
            input[:,1:size(_input,2),i] = _input
            output[:,1:size(_output,2),i] = _output
            aa = [create_mask(i, go_cue_onset+_go_cue_onset, _grace_period, post_cue_multiplier, pre_cue_multiplier) for i in 1:nsteps, j in 1:noutput]

            output_mask[:,:,i] .= permutedims(aa)
         end
        input .+= σ.*randn(rng, size(input)...)
        input,output, output_mask
    end,NamedTuple(args), h)
end

end # module RNNTrialStructures
