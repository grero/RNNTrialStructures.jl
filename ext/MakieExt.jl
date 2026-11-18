module MakieExt
using RNNTrialStructures
using Makie
import Makie.SpecApi as S

function Makie.convert_arguments(::Type{<:AbstractPlot}, arena::RNNTrialStructures.TexturedArena{T}) where T <: Real
	# draw the outline
	w = arena.ncols*arena.colsize
	h = arena.nrows*arena.rowsize
	cr = (minimum(arena.textures), maximum(arena.textures))
	if length(arena.textures) == 4
		# only texture for pillars
		color =:black
	else
		color = arena.textures[5:end]
	end

	l = S.LineSegments(Rect(0.0, 0.0, w, h), color=color, colorrange=cr)
	p = [S.Poly(_points, color=tt, colorrange=cr) for (tt,_points) in zip(arena.textures,RNNTrialStructures.get_obstacle_points(arena))]
	#S.Axis(plots=[l;p], aspect=1)
	[l;p]
end
end
