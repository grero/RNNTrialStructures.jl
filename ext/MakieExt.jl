module MakieExt
using RNNTrialStructures
using Makie
import Makie.SpecApi as S

function Makie.convert_arguments(::Type{<:AbstractPlot}, arena::RNNTrialStructures.TexturedArena{T}) where T <: Real
	# draw the outline
	w = arena.ncols*arena.colsize
	h = arena.nrows*arena.rowsize
	l = S.Lines(Rect(0.0, 0.0, w, h), color=:black)
	p = [S.Poly(_points, color=:gray) for _points in RNNTrialStructures.get_obstacle_points(arena)]
	#S.Axis(plots=[l;p], aspect=1)
	[l;p]
end
end
