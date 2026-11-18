# RNNTrialStructures
[![CI](https://github.com/grero/RNNTrialStructures.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/grero/RNNTrialStructures.jl/actions/workflows/ci.yml)

[![codecov](https://codecov.io/gh/grero/RNNTrialStructures.jl/graph/badge.svg?token=4LN4RIG3eX)](https://codecov.io/gh/grero/RNNTrialStructures.jl)
## Usage

### Multiple angular input trial
Create an instance of a trial structure where two angles are presented with a 500 ms delay between them. These two angles need to be reported back after an additional delay of 500 ms.

```julia
using RNNTrialStructures

# The angles are encoded via 32 input units with angular preference spanning the circle
apref = RNNTrialStructures.AngularPreference(collect(range(0.0f0, stop=2.0f0*π, length=32)), 4.1f0, 0.8f0)

input_onset = 300.0f0
input_duration = 300.0f0
delay1 = 500.0f0
delay2 = 500.0f0
output_duration=500.0f0
dt = 20.0f0

trialstruct = RNNTrialStructures.MultipleAngleTrial(input_onset,
                                                   input_duration, 
                                                   [delay1, delay2], output_duration, 2, dt, apref)
```

### Navigation trial

Create a trial structure in which an agent is passively moved around a defined arena.

```julia
using RNNTrialStructures

apref = RNNTrialStructures.AngularPreference(collect(range(0.0f0, stop=2.0f0*π, length=32)), 4.1f0, 0.8f0)

# default maze with 10 x 10 bin floor and 4 pillars.
arena = RNNTrialStructures.MazeArena()

# each trial consists of a random number of steps, between min_num_steps and max_num_steps.
min_num_steps = 20
max_num_steps = 50
# the inputs are distance to the walls of the arena along 16 rays spanning the field of view of the agent, the self-movement from one time step to the next and the texture of the walls where the 16 gaze rays intersect the maze. The output is the conjunction of the agent's location in the arena and the location of the intersection of its centrial gaze ray.
 trialstruct = RNNTrialStructures.NavigationTrial(min_num_steps,max_num_steps,[:distance, :movement, :texture], [:conjunction], arena,apref)

```
Other possible values for the outputs are `:position` and `:gaze`.

![Illustration of the four pillar maze](assets/default_arena_layout.png)
Illustration of the default four pillar maze.
