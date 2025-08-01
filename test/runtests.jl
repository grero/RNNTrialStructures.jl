using RNNTrialStructures
using Random
using StableRNGs
using Test

@testset "Place cells" begin
    pc = RNNTrialStructures.PlaceCells([(0.5, 0.5), (-0.5, -0.5)],[1.0, 1.0])
    output = pc((0.0, 0.0))
    @test output ≈ [0.7788007830714049, 0.7788007830714049]
end

@testset "Trials" begin
    @testset "Category" begin
        trial = RNNTrialStructures.CategoryTrial(;nlocations=4,nsteps=50, go_cue_onset=30, target_onset=5)
        input, output = trial(1, 30)
        @test size(input) == (5,50)
        @test size(output) == (4,50)
        @test all(input[1,5:9] .≈ 1.0)
        @test all(input[2:end-1,:] .≈ 0.0)
        @test all(input[end,30:end] .≈ 1.0)
        @test all(input[end,1:29] .≈ 0.0)
        @test all(output[1,30:end] .≈ 1.0)
        @test all(output[2:end,:] .≈ 0.0)
        pfunc = RNNTrialStructures.penalty(typeof(trial))
        @test pfunc([0.0,0.0], [1.0,0.0]) ≈ sum(abs2, [1.0, 0.0])
        @test RNNTrialStructures.matches(trial, output, 1, CartesianIndex(1,31))

    end

    @testset "Position" begin
        trial = RNNTrialStructures.PositionTrial([(-10.0, 10.0)], [(-10.0,10.0),(10.0, -10.0)],[5.0, 5.0];nsteps=50, go_cue_onset=30, target_onset=5)
        input, output = trial((-10.0, 10.0), 30)
        @test size(input) == (3,50)
        @test size(output) == (2,50)
        @test all(input[1:2,5:9] .≈ [1.0, 1.1253517471925912e-7])
        @test all(input[3,1:29] .≈ 0.0)
        @test all(input[3,30:34] .≈ 1.0)
        @test all(output[:,30:end] .≈ (-10.0, 10.0))
        @test all(output[:,1:29] .≈ [0.0,0.0])
        pfunc = RNNTrialStructures.penalty(typeof(trial))
        @test pfunc([0.0, 0.0], [-10.0, 10.0]) ≈ sum(abs2, [-10.0, 10.0])
        @test pfunc([10.0, 10.0], [10.0, 10.0]) ≈ 0.0
        @test RNNTrialStructures.matches(trial, output, (-10.0, 10.0), CartesianIndex(1,31))
    end

    @testset "Angle" begin
        trial = RNNTrialStructures.AngleTrial([0.0, π/4];nsteps=50, go_cue_onset=30, target_onset=5)
        θ = 1/4 
        input, output = trial(θ, 30)
        @test size(input) == (3,50)
        @test size(output) == (2,50)
        @test all(input[1:1,5:9] .≈ cos(θ))
        @test all(input[3,1:29] .≈ 0.0)
        @test all(input[3,30:end] .≈ 1.0)
        @test all(output[1,30:end] .≈ cos(θ))
        @test all(output[2,30:end] .≈ sin(θ))
        @test all(output[:,1:29] .≈ 0.0)
        pfunc = RNNTrialStructures.penalty(typeof(trial))
        @test pfunc(0.0, θ) ≈ sum(abs2, 0.0 - θ) 
        @test pfunc(θ, θ) ≈ 0.0
        @test pfunc(0.0, 0.25) ≈ abs2(0.0-0.25) 
        @test pfunc(0.0, 0.5) ≈ abs2(0.0-0.5)
        idx = CartesianIndex(1,31)
        @test RNNTrialStructures.matches(trial, output, θ, idx)
        # mutate slightly
        output[idx] -= 0.1
        @test RNNTrialStructures.matches(trial, output, θ, idx)
    end
    
    @testset "RandomSequenceTrial" begin
        apref = RNNTrialStructures.AngularPreference(collect(range(0.0f0, stop=2.0f0*π, length=32)), 4.1f0, 0.8f0)
        trialstruct = RNNTrialStructures.RandomSequenceTrial(20.0f0, 20.0f0, 20.0f0, 20.0f0, 2, 9, apref)
        rng = StableRNG(1234) 
        θ = RNNTrialStructures.get_trialid(trialstruct, 2, 5, rng)
        @test θ ≈ Float32[3.1415927, 1.5707964]
        Random.seed!(rng, 1234)
        θ = RNNTrialStructures.get_trialid(trialstruct, rng)
        @test length(θ) == 6
        @test θ ≈ Float32[-2.3134663, 1.9634168, 0.58281016, 1.3592651, 0.077911615, -2.211473]
        trial_generator = RNNTrialStructures.generate_trials(trialstruct, 256, 20.0f0;σ=0.03f0,rng=rng)
        @test isa(trial_generator, RNNTrialStructures.TrialIterator)
        @test trial_generator.arghash == 0x444662f0
        @test trial_generator.args.trialstruct == trialstruct
        @test trial_generator.args.ntrials == 256
        @test trial_generator.args.dt == 20.0f0 
        @test trial_generator.args.σ == 0.03f0 
        @test trial_generator.args.rng == rng
        x,y,w = trial_generator()
        nsteps = RNNTrialStructures.get_nsteps(trialstruct,trialstruct.max_seq_length, 20.0f0)
        @test size(x,2) == size(y,2) == size(w,2) ==  nsteps
        @test size(x,3) == size(y,3) == size(w,3) == 256
        pp = RNNTrialStructures.performance(trialstruct, y, y)
        @test pp ≈ 1.0f0

        @test RNNTrialStructures.get_name(trialstruct) == :RandomSequenceTrial
        sig = RNNTrialStructures.signature(trialstruct)
        @test sig == 0xd423d481
    end
end
