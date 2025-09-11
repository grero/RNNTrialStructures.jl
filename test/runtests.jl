using RNNTrialStructures
using Random
using StableRNGs
using Test

@testset "NavigationUtils" begin
    (θ1,θ2),(i,j) = RNNTrialStructures.order_angles(0.4268554, -0.25289375)
    @test θ2 ≈ 0.4268554 
    @test θ1 ≈ -0.25289375
    @test i == 2
    @test j == 1

    (θ1,θ2),(i,j) = RNNTrialStructures.order_angles( 6.061203f0,0.36774418f0)
    @test θ1 ≈ -0.22198230424989873
    @test θ2 ≈ 0.36774418f0
    @test i == 1
    @test j == 2

    # non-overlapping
    obstructed_angles = Tuple{Float32, Float32}[(0.7873215, 1.3179023), (1.7504921, 2.483581)]
    oo,ii = RNNTrialStructures.consolidate_view(obstructed_angles)
    @test all(oo[1] .≈ obstructed_angles[1])
    @test all(oo[2] .≈ obstructed_angles[2])
    @test ii[1] == ((1,1),(1,2)) 
    @test ii[2] == ((2,1),(2,2))

    # overlapping
    obstructed_angles = Tuple{Float32, Float32}[(0.7873215, 1.8179023), (1.7504921, 2.483581)]
    oo,ii = RNNTrialStructures.consolidate_view(obstructed_angles)
    @test length(oo) == 1
    @test all(oo[1] .≈ (0.7873215f0, 2.483581f0))
    @test ii[1] == ((1,1),(2,2))

end
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
        @test pp[1] ≈ 1.0f0

        @test RNNTrialStructures.get_name(trialstruct) == :RandomSequenceTrial
        sig = RNNTrialStructures.signature(trialstruct)
        @test sig == 0xd423d481
    end
end

@testset "Navigation" begin
    arena = RNNTrialStructures.Arena(5,5,1.0f0, 1.0f0)
    @test RNNTrialStructures.signature(arena)  == 0xb0173a4c

    @test RNNTrialStructures.get_center_position(arena) == [2.5f0, 2.5f0]

    spiral_path = RNNTrialStructures.traverse_outwards(arena)
    @test spiral_path == Tuple{Float32, Float32}[(2.5, 2.5), (1.5, 2.5), (1.5, 1.5), (2.5, 1.5), (3.5, 1.5), (3.5, 2.5), (3.5, 3.5), (2.5, 3.5), (1.5, 3.5), (0.5, 3.5), (0.5, 2.5), (0.5, 1.5), (0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5), (4.5, 0.5), (4.5, 1.5), (4.5, 2.5), (4.5, 3.5), (4.5, 4.5), (3.5, 4.5), (2.5, 4.5), (1.5, 4.5), (0.5, 4.5)]

    possible_steps = RNNTrialStructures.check_step(1,1,arena)
    @test possible_steps == [(0,0), (1,0),(0,1)]

    possible_steps = RNNTrialStructures.check_step(5,5,arena)
    @test possible_steps == [(0,0), (-1,0), (0,-1)]

    arena = RNNTrialStructures.MazeArena(10,10,1.0f0,1.0f0,[[(2,2)],[(4,2)],[(4,4)],[(2,4)]])
    possible_steps = RNNTrialStructures.check_step(1,2,arena)
    # an obstacle to our right; we can only move up or down
    @test possible_steps == [(0,0),(0,1),(0,-1)]

    arena = RNNTrialStructures.MazeArena(10,10,1.0f0,1.0f0,[[(3,3),(4,3),(4,4),(3,4)],[(7,3),(8,3),(8,4),(7,4)],[(7,7),(8,7),(8,8),(7,8)],[(3,7),(4,7),(4,8),(3,8)]])

    @test RNNTrialStructures.signature(arena) == 0x74ba5f43

    center_pos = RNNTrialStructures.get_center(arena)
    @test center_pos == (5.0f0, 5.0f0)

    apref = RNNTrialStructures.AngularPreference(collect(range(0.0f0, stop=2.0f0*π, length=16)), 5.0f0, 0.8f0);

    pp, dp = RNNTrialStructures.get_obstacle_intersection([1.5f0, 4.5f0], [1.0471976f0-Float32(π/3)/2], arena, 1.0471976f0, Float32(π/3))
    @test all(pp[1] .≈ (6.0f0, 7.0980763f0))
    @test dp[1] ≈ 5.1961527f0

    obstacle_points = RNNTrialStructures.get_obstacle_points(arena)
    res = RNNTrialStructures.inview(obstacle_points[4], [1.5f0, 4.5f0], 1.0471976f0, Float32(π/3))
    @test res == ones(Bool, 4)

    res = RNNTrialStructures.inview(obstacle_points[3], [1.5f0, 4.5f0], 1.0471976f0, Float32(π/3))
    @test res == [false, false, false, true]

    res = RNNTrialStructures.inview(obstacle_points[2], [1.5f0, 4.5f0], 1.0471976f0, Float32(π/3))
    @test res == zeros(Bool, 4)

    trialstruct = RNNTrialStructures.NavigationTrial(5,10,[:distance, :view],[:position], arena,apref)
    @test RNNTrialStructures.num_inputs(trialstruct) == 32 
    @test RNNTrialStructures.num_outputs(trialstruct) == 2

    @test RNNTrialStructures.signature(trialstruct) == 0x95b53b84 

    rng = StableRNG(1234)
    position, head_direction, viewf, movement,dist = trialstruct(;rng=rng) 
    @test size(position,2) == size(head_direction,2) == size(viewf,2) == size(dist,2) == 9
    @test size(position,1) == 2
end
