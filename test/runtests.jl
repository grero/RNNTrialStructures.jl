using RNNTrialStructures
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
end
