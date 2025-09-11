# RNNTrialStructures
[![CI](https://github.com/grero/RNNTrialStructures.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/grero/RNNTrialStructures.jl/actions/workflows/ci.yml)

## Usage

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
