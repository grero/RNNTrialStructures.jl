module RNNTrialStructures
using Random
using StatsBase
using CRC32c

include("trialstructures.jl")

mutable struct TrialIterator
    data_provider::Function
    arghash::UInt32
    state::Int64
end

TrialIterator(func,h) = TrialIterator(func, h,0)


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

function generate_trials(trialstruct::AbstractTrialStruct{T}, ntrials::Int64;randomize_go_cue=true, randomize_grace_period=false, go_cue_onset_min::T=zero(T), go_cue_onset_max::T=go_cue_onset_min,
                                                      grace_period_min::T=zero(T), grace_period_max=grace_period_min,
                                                      post_cue_multiplier::T=T(2.0), pre_cue_multiplier::T=one(T),
                                                      σ=zero(T), constraint_factor::T=T(0.0),rng=Random.default_rng(), rseed=1234) where T <: Real

    Random.seed!(rng, rseed)
    #generate a hash of the arguments
    h = zero(UInt32)
    h = crc32c(string(ntrials), h)
    h = crc32c(string(randomize_go_cue), h)
    h = crc32c(string(randomize_grace_period),h)
    h = crc32c(string(go_cue_onset_min),h)
    h = crc32c(string(go_cue_onset_max),h)
    h = crc32c(string(grace_period_min),h)
    h = crc32c(string(grace_period_max),h)
    h = crc32c(string(post_cue_multiplier),h) 
    h = crc32c(string(pre_cue_multiplier),h) 
    h = crc32c(string(σ),h)
    h = crc32c(string(constraint_factor),h)
    h = crc32c(string(rng),h)
    h = crc32c(string(rseed),h)

    TrialIterator(function data_provider()
        dt = trialstruct.dt
        ninput = num_inputs(trialstruct)
        noutput = num_outputs(trialstruct)
        nsteps = trialstruct.tdim
        # account for the fact that varying go-cue could extend the trial
        nsteps += round(Int64, ceil(go_cue_onset_max/dt))
        input = zeros(T, ninput, nsteps, ntrials)
        output = zeros(T, noutput, nsteps, ntrials)
        output_mask = zeros(T, noutput, nsteps, ntrials)
        # by default, the go-cue onset coincides  with the first response onset
        go_cue_onset = trialstruct.response_onset[1]-1
        for i in 1:ntrials
            trialid = get_trialid(trialstruct,constraint_factor;rng=rng)
            if randomize_go_cue
                _go_cue_onset = rand(round(Int64, go_cue_onset_min/dt):round(Int64,go_cue_onset_max/dt))
            else
                _go_cue_onset = 0 
            end
            if randomize_grace_period
                _grace_period = rand(round(Int64, grace_period_min/dt):round(Int64, grace_period_max/dt))
            else
                _grace_period = round(Int64, grace_period_min/dt)
            end
            _input, _output = trialstruct(trialid, _go_cue_onset)
            input[:,1:size(_input,2),i] = _input
            output[:,1:size(_output,2),i] = _output
            aa = [create_mask(i, go_cue_onset+_go_cue_onset, _grace_period, post_cue_multiplier, pre_cue_multiplier) for i in 1:nsteps, j in 1:noutput]

            output_mask[:,:,i] .= permutedims(aa)
         end
        input .+= σ.*randn(rng, size(input)...)
        input,output, output_mask
    end,h)
end

end # module RNNTrialStructures
