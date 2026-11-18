using LinearAlgebra
using GeometryBasics: Point2f, Rect

abstract type AbstractArena{T<:Real} end
abstract type AbstractMazeArena{T<:Real} <: AbstractArena{T} end
struct Arena{T<:Real} <: AbstractArena{T}
    ncols::Int64
    nrows::Int64
    colsize::T
    rowsize::T
end

struct MazeArena{T<:Real} <: AbstractMazeArena{T}
    ncols::Int64
    nrows::Int64
    colsize::T
    rowsize::T
    obstacles::Vector{Vector{Tuple{Int64,Int64}}} # vector of vector of points defining the borders of the obstacles
end

struct TexturedArena{T<:Real} <: AbstractMazeArena{T}
    ncols::Int64
    nrows::Int64
    colsize::T
    rowsize::T
    obstacles::Vector{Vector{Tuple{Int64,Int64}}} # vector of vector of points defining the borders of the obstacles
    textures::Vector{T}  # texture associated with each wall/obstacle
end

function TexturedArena(arena, obstacles::Vector{Vector{Tuple{Int64, Int64}}}, texture_values::Vector{T}) where T <: Real
    # one texture value per obstacle
    TexturedArena{T}(arena.ncols, arena.nrows, arena.colsize, arena.rowsize, obstacles, texture_values)
end

function TexturedArena(arena::MazeArena{T}) where T <: Real
    texture_values = [T(i) for i in 1:length(arena.obstacles)]
    TexturedArena(arena,texture_values)
end

function TexturedArena(arena::MazeArena{T}, texture_values::Vector{T}) where T <: Real
    TexturedArena{T}(arena.ncols, arena.nrows, arena.colsize, arena.rowsize, arena.obstacles, texture_values)
end

function get_texture(pos::Vector{T}, θ::AbstractVector{T}, arena::TexturedArena{T}, θ0::T, fov::T) where T <: Real
    pp,dm,doid = get_obstacle_intersection(pos, θ, arena, θ0, fov)
    # oid returns a number whose 
    oid = round.(Int64, doid) 
    Δ = doid .- oid
    doid .= Δ .+ [_oid > 0 ? arena.textures[_oid] : zero(T) for _oid in oid]
    pp,dm,doid
end

function get_texture(pos::Vector{T}, θ::AbstractVector{T}, arena::MazeArena{T}, θ0::T, fov::T) where T <: Real
    pp,dm,oid = get_obstacle_intersection(pos, θ, arena, θ0, fov)
    pp,dm, T.(oid)
end

function get_texture(pos::Vector{T}, θ::AbstractVector{T}, arena::Arena{T}, θ0::T, fov::T) where T <: Real
    pp,dm,oid = get_obstacle_intersection(pos, θ, arena, θ0, fov)
    pp, dm, T.(oid)
end

function (arena::AbstractArena{T})(x::T,y::T, θ::AbstractVector{T}, θ0::T, fov::T) where T <: Real
    # distance
    pp, dm, tt = get_texture([x,y], θ, arena, θ0, fov)
    θv = get_view([x,y],θ0, arena;fov=fov)
    θv[1], dm, tt
end

"""
    MazeArena()

Return a 10 × 10 grid arena with four obstacles
"""
function MazeArena()
    arena = RNNTrialStructures.MazeArena(10,10,1.0f0,1.0f0,
                                          [[(3,3),(4,3),(4,4),(3,4)],
                                          [(7,3),(8,3),(8,4),(7,4)],
                                          [(7,7),(8,7),(8,8),(7,8)],
                                          [(3,7),(4,7),(4,8),(3,8)]])
    arena
end

function signature(arena::Arena{T},h=zero(UInt32)) where T <: Real
   for q in [arena.ncols, arena.nrows, arena.colsize, arena.rowsize]
        for ii in q
            h = crc32c(string(ii), h)
        end
    end 
    h
end

function signature(arena::MazeArena{T},h=zero(UInt32)) where T <: Real
   for q in [arena.ncols, arena.nrows, arena.colsize, arena.rowsize, arena.obstacles]
        for ii in q
            h = crc32c(string(ii), h)
        end
    end 
    h
end

function get_center_position(arena::Arena{T}) where T <: Real
    i = div(arena.ncols,2)+1
    j = div(arena.nrows,2)+1
    get_position(i,j,arena)
end

function get_center(arena::AbstractArena)
    w,h = extent(arena)
    w/2,h/2
end

extent(arena::AbstractArena)  = (arena.ncols*arena.colsize, arena.nrows*arena.rowsize)

function assign_bin(x,y, arena::Arena;binsize=arena.colsize)
    i = round(Int64, floor(y/binsize))+1
    j = round(Int64, floor(x/binsize))+1
    j,i, (j-1)*arena.nrows + i
end

function assign_bin(x,y, arena::MazeArena;binsize=arena.colsize)
    i = round(Int64, floor(y/binsize))+1
    j = round(Int64, floor(x/binsize))+1
    nn = num_floor_bins(arena;binsize=binsize)
    # TODO: Remove the bins occupied by the pillars
    xx,yy = (0.0, 0.0)
    k = 1
    finished = false
    while xx < arena.ncols*arena.colsize
        j2 = round(Int64, floor(xx/binsize))+1
        jj = round(Int64,floor(xx/arena.colsize))+1
        yy = 0.0
        while yy < arena.nrows*arena.rowsize
            ii = round(Int64,floor(yy/arena.rowsize))+1
            i2 = round(Int64, floor(yy/binsize))+1
            if (j2,i2) == (j,i)
                finished = true
                break
            end
            do_include = true
            for pp in arena.obstacles
                if (jj,ii) in pp
                    do_include = false
                    break
                end
            end
            if do_include
                k += 1
            end
            yy += binsize
        end
        if finished
            break
        end
             
        xx += binsize
    end
    j,i, k
end

function wall_bins(arena::AbstractArena)
    #left wall
    bins = [(0,i) for i in 1:arena.nrows]
    # top wall
    append!(bins, [(j,arena.nrows) for j in 1:arena.ncols])
    # right wall
    append!(bins, [(arena.ncols,j) for j in arena.nrows:-1:1])
    # bottom wall
    append!(bins, [(i, 0) for i in arena.ncols:-1:1])
    reverse(bins)
end

function surface_bins(arena::MazeArena)
    wbins = wall_bins(arena)
    for ob in arena.obstacles
        # TODO: Treat the sides differently
        append!(wbins, ob)
    end
    wbins
end

function assign_surface_bin(x,y, arena;binsize=arena.colsize,binsize_wall=binsize)
    #loop through each surface
    xm = arena.ncols*arena.colsize
    ym = arena.nrows*arena.rowsize
    dm = Inf
    pm = (0.0, 0.0)
    lm = 0
    # walls
    xx = [0.0, 0.0, xm, xm]
    yy = [0.0, ym, ym, 0.0]
    pp = collect(zip(xx,yy))
    offset = 0
    for (p1,p2) in zip(pp, circshift(pp,-1))
        pl = find_line_intersection((x,y), p1,p2)
        _dl = norm((x,y) .- pl)
        if _dl <  dm
            dm = _dl
            pm = pl
            lm = offset+round(Int64, floor((norm(pl .- p1))./binsize_wall))+1
        end
        offset += round(Int64, floor(norm(p2 .- p1)/binsize_wall))
    end
    # pillars
    for pp in get_obstacle_points(arena)
        for (p1,p2) in zip(pp, circshift(pp,-1))
            pl = find_line_intersection((x,y), p1,p2)
            if (p1 <= pl <= p2) || (p2 <= pl <= p1)
                _dl = norm((x,y) .- pl)
                if _dl <  dm
                    dm = _dl
                    pm = pl
                    # TODO: This doesn't work if the binsize is non-uniform
                    lm = offset + round(Int64, floor((norm(pl .- p1))./binsize))+1
                end
            end
            offset += round(Int64, floor(norm(p2 .- p1)/binsize))
        end
    end
    pm, dm, lm
end

function num_surface_bins(arena;binsize=arena.colsize, binsize_wall=binsize)
    nn = 2*round(Int64, floor(arena.ncols*arena.colsize/binsize_wall))
    nn += 2*round(Int64, floor(arena.nrows*arena.rowsize/binsize_wall))
    opoints = get_obstacle_points(arena)
    for pp in opoints
        ll = 0
        for (p1,p2) in zip(pp, circshift(pp,-1))
            ll += norm(p2 .- p1)
        end
        nn += round(Int64,ll/binsize)
    end
    nn
end

function num_floor_bins(arena::Arena;binsize=arena.colsize)
    nn = round(Int64, arena.ncols*arena.colsize*arena.nrows*arena.rowsize/binsize^2)
end

function num_floor_bins(arena::MazeArena;binsize=arena.colsize)
    nn = round(Int64, arena.ncols*arena.colsize*arena.nrows*arena.rowsize/binsize^2)
    for pp in arena.obstacles
        nn -= round(Int64,length(pp)*arena.colsize*arena.rowsize/binsize^2)
    end
    nn
end

"""
Return a path starting from the middel and moving in a spiral outwards
"""
function traverse_outwards(arena::Arena{T}) where T <: Real
    i,j = get_center_position(arena)
    _path = [(i,j)]
    step = 1
    ss = -1
    ee = extent(arena)
    while (0 < i <= arena.ncols) && (0 < j <= arena.ncols)
        stop = false
        for k in 1:step
            i += ss 
            if !(0 < i <= ee[1])
                stop = true
            else
                push!(_path, (i,j))
            end
        end
        if stop
            break
        end
        for k in 1:step
            j += ss 
            if !(0 < j <= ee[2])
                stop = true
            else
                push!(_path, (i,j))
            end
        end
        ss = -ss
        step = step+1
    end
    _path
end

"""
Return possible steps
"""
function check_step(i::Int64,j::Int64, ncols::Int64, nrows::Int64) 
    possible_steps = [(0,0)]
    if i > 1 
        push!(possible_steps, (-1, 0))
    end
    if i < ncols
        push!(possible_steps, (1, 0))
    end
    if j > 1
        push!(possible_steps, (0,-1))
    end
    if j < nrows
        push!(possible_steps, (0,1))
    end
    possible_steps
end

function check_step(i,j,arena::Arena)
    check_step(i,j,arena.ncols, arena.nrows)
end

"""
    validate_step(i::Int64, j::Int64, Δi::Int64, Δj::Int64, arena::Arena)

Validate the step (i+Δi, j+Δj) in the context of `arena`, returning the validated Δi and Δj.

If a step is not within the bounds of `arena`, the step is set to zero.

juliadoc````
    julia> arena = Arena(5,5,1.0,1.0)
    julia> validate_step(1,2, -1, -1, arena)
    (0,-1) 
    julia> validate_step(2,1, -1, -1, arena)
    (-1,0)
    julia> validate_step(5,2,1,1,arena)
    (0,1)
    julia validate_step(2,5,1,1,arena)
    (1,0)
````
"""
function validate_step(i::Int64, j::Int64, Δi::Int64, Δj::Int64, arena::Arena)
    Δi = ifelse(0 < i+Δi < arena.ncols, Δi, 0)
    Δj = ifelse(0 < j+Δj < arena.nrows, Δj, 0)
    Δi, Δj
end

function validate_step(i::Int64, j::Int64, vv::Vector{T}, arena::Arena{T}) where T <: Real
    ss = [-1 1 0 0;
           0 0 -1 1]
    (Δi,Δj) = round.(Int64, ss*vv)
    validate_step(i,j, Δi,Δj, arena)
end

function check_step(i::Int64, j::Int64, arena::MazeArena{T}) where T <: Real
    in_any(x) = any([in(obstacle)(x) for obstacle in arena.obstacles]) # array of functions
    possible_steps = [(0,0)]
    if (i < arena.ncols) && !(in_any((i+1,j)))
        push!(possible_steps, (1,0))
    end
    if (i > 1) && !(in_any((i-1,j)))
        push!(possible_steps, (-1,0))
    end
    if (j < arena.nrows) && !(in_any((i,j+1)))
        push!(possible_steps, (0,1))
    end
    if (j > 1) && !(in_any((i,j-1)))
        push!(possible_steps, (0,-1))
    end
    possible_steps
end

function validate_step(i::Int64, j::Int64, vv::Vector{T}, arena::MazeArena{T}) where T <: Real
    ss = [-1 1 0 0;
           0 0 -1 1]
    Δ = ss*vv
    pp = Point2f(i,j) .+ Δ
    did_hit = false
    for obstacle in arena.obstacles
        rr = Rect(Point2f.(obstacle))
        if pp in rr 
            did_hit = true
            break
        end
    end
    if did_hit
        return (zero(T), zero(T))
    end
    return Δ
end

function get_obstacle_points(arena::AbstractMazeArena{T}) where T <: Real
    ncols = arena.ncols
    colsize = arena.colsize
    rowsize = arena.rowsize
    nrows = arena.nrows

    points = Vector{Tuple{T,T}}[]
    for obstacle in arena.obstacles
        # expand every grid point
        xmin = minimum([_p[1] for _p in obstacle])
        ymin = minimum([_p[2] for _p in obstacle])
        xmax = maximum([_p[1] for _p in obstacle])
        ymax = maximum([_p[2] for _p in obstacle])

        #lower left
        p0 = ((xmin-1)*colsize, (ymin-1)*rowsize)
        # lower right
        p1 = (xmax*colsize, (ymin-1)*rowsize)
        # upper right
        p2 = (xmax*colsize, ymax*rowsize)
        # upper left
        p3 = ((xmin-1)*colsize, ymax*rowsize)
        push!(points, [p0,p1,p2,p3])
    end
    points
end

get_obstacle_points(arena::Arena{T}) where T = Tuple{T,T}[]

get_num_obstacles(arena::Arena) = 0
get_num_obstacles(arena::MazeArena) = length(arena.obstacles)

"""
    get_coordinate(i::Int64, j::Int64,arena::Arena{T};Δθ=π/4,rng=Random.default_rng()) where T <: Real

Generate a new coordinate in `arena` by choosing a random step subject to the arena constraints.
"""
function get_coordinate(i::Int64, j::Int64,arena::Arena{T};Δθ=π/4,rng=Random.default_rng()) where T <: Real
    possible_steps = check_step(i,j,arena.ncols,arena.nrows)
    Δ = rand(rng, possible_steps)
    (i,j) = (i+Δ[1], j+Δ[2])
    (i,j)
end

function get_coordinate(i::Int64, j::Int64,arena::AbstractArena{T},θhd::T;Δθ::T=T(π/4),rng=Random.default_rng(),p_hd=T(0.5)) where T <: Real
    possible_steps = check_step(i,j,arena)
    nsteps = length(possible_steps)
    #figure out the step most aligned with the current head direction
    dx = [cos(θhd),sin(θhd)]
    m = zeros(T, nsteps) 
    for (ii,ps) in enumerate(possible_steps)
        m[ii] = dx[1]*ps[1] + dx[2]*ps[2]
    end
    kk = findall(m.==maximum(m))
    pp = p_hd
    pq = T(1.0/(nsteps-length(kk)))
    jj = 0
    if rand(rng) < pp
        jj = rand(rng,kk)
    else
        for (k,ii) in enumerate(setdiff(1:nsteps, kk))
            if rand(rng) < k*pq 
                jj = ii
                break
            end
        end
    end
    Δ = possible_steps[jj]
    (i,j) = (i+Δ[1], j+Δ[2])
    (i,j)
end

function get_coordinate(arena::Arena{T};rng=Random.default_rng()) where T <: Real
    i = rand(rng, 1:arena.ncols)
    j = rand(rng, 1:arena.nrows)
    (i,j)
end

function get_coordinate(arena::AbstractMazeArena{T};rng=Random.default_rng()) where T <: Real
    valid_points = Tuple{Int64, Int64}[]
    for i in 1:arena.ncols
        for j in 1:arena.nrows
            valid_point = true
            for obstacle in arena.obstacles
                if (in(obstacle)((i,j)))
                    valid_point = false
                    break
                end
            end
            if valid_point
                push!(valid_points, (i,j))
            end
        end
    end
    rand(rng, valid_points)
end

function get_position(arena::AbstractArena{T};rng=Random.default_rng()) where T <: Real
    i,j = get_coordinate(arena;rng=rng)
    get_position(i,j,arena)
end

function get_position(i::Int64, j::Int64, arena::AbstractArena)
    wc = arena.colsize/2
    wr = arena.rowsize/2
    [(i-1)*arena.colsize + wc, (j-1)*arena.rowsize+wr]
end

function get_head_direction(Δθ::T;rng=Random.default_rng()) where T <: Real
    rand(rng, [-Δθ, zero(T), Δθ])
end

function get_head_direction(Δθ::T,θ::T;rng=Random.default_rng(),p_stay::T=T(0.5)) where T <: Real
    pp = cumsum([(1 - p_stay)/2, p_stay, (1-p_stay)/2])
    cc = [-Δθ, zero(T), Δθ]
    ii = 2 # stop gap; if for some reason the below fails, default to staying
    for jj in 1:length(pp)
        if rand(rng) < pp[jj]
            ii = jj
            break
        end
    end
    cc[ii]
end

struct ViewField{T<:Real}

end

"""
Return a view field, in terms of an angular range, given the grid position `x` and `y`
"""
function (vf::ViewField{T})(x::Int64, y::Int64,θ::T) where T <: Real
    # just to it in a streight forward dumb way
    view_bins = zeros(Bool, 4, 8)
    if x == 1
        if y == 1
            if θ == 9π/4
                view_bins[1,1] = true
                view_bins[2,1] = true
            elseif θ == 3π/2
                view_bins[1,1:2] .= true
            elseif θ == 3π/2+π/4
                view_bins[1,2:end] .= true
                view_bins[2,1] .= true
            elseif θ == 0
                view_bins[1,3:end] .= true
                view_bins[2,2:end] .= true
            elseif θ == π/4
                view_bins[2,3:end] .= true
            elseif θ == π/2
                view_bins[1,4:end] .= true
                view_bins[3,]
                view_bins[2,4:end] .= true
            end

        end
    end

end

abstract type AbstractNavigationTrial{T<:Real} <: AbstractTrialStruct{T} end

struct NavigationTrial{T<:Real} <: AbstractNavigationTrial{T}
    min_num_steps::Int64
    max_num_steps::Int64
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    arena::AbstractArena{T}
    angular_pref::AngularPreference{T}
end

struct ActiveNavigationTrial{T<:Real} <: AbstractNavigationTrial{T}
    min_num_steps::Int64
    max_num_steps::Int64
    inputs::Vector{Symbol}
    outputs::Vector{Symbol}
    arena::AbstractArena{T}
    angular_pref::AngularPreference{T}
end

"""
Create a copy of `trialstruct` but replacing fields from `kwargs`
"""
function clone(trialstruct::NavigationTrial{T};kwargs...) where T <: Real
    args = Any[]
    qwargs = Dict(kwargs)
    for k in fieldnames(NavigationTrial)
        if k in keys(qwargs)
            v = qwargs[k]
        else
            v = getfield(trialstruct, k)
        end
        push!(args, v)
    end
    NavigationTrial{T}(args...)
end

# fallback
function NavigationTrial(min_num_steps::Int64, max_num_steps::Int64, arena::AbstractArena{T}, apref::AngularPreference{T})  where T <: Real
    NavigationTrial{T}(min_num_steps, max_num_steps, [:view],[:position],arena, apref)
end

get_name(::Type{NavigationTrial{T}}) where T <: Real = :NavigationTrial

function sort_inputs(::Union{Type{NavigationTrial}, Type{NavigationTrial{T}}}, inputs) where T <: Real
    ordered_inputs =  [:view, :head_direction, :movement, :distance, :texture]
    inputs = sort(inputs, by=x->findfirst(ordered_inputs.==x))
    inputs
end

function sort_outputs(::Type{NavigationTrial}, outputs)
    ordered_outputs = [:position, :head_direction, :distance, :texture, :gaze, :conjuction]
    outputs = sort(outputs, by=x->findfirst(ordered_outputs.==x))
end

function Base.length(trial::NavigationTrial, input::Symbol)
    l = 0
    if input == :movement
        l = 4
    elseif input == :view
        l = 16
    elseif input == :texture
        l = 16
    elseif input == :distance
        l = 16
    elseif input == :head_direction
        l = 16
    end
    l
end

function signature(trial::NavigationTrial{T},h=zero(UInt32);respect_order=true) where T <: Real
    # TODO: The order of inputs and outputs should not matter here
    #     : sort using the order that the trial function uses
    if respect_order
        inputs = trial.inputs
        outputs = trial.outputs
    else
        inputs = sort_inputs(NavigationTrial, trial.inputs)
        outputs = sort_outputs(NavigationTrial, trial.outputs)
    end
    for q in [trial.min_num_steps, trial.max_num_steps, inputs, outputs]
        for ii in q
            h = crc32c(string(ii), h)
        end
    end
    h = signature(trial.arena,h)
    h = signature(trial.angular_pref,h)
    h
end

function get_circle_intersection(arena::AbstractArena{T}, point::Vector{T}, θ::T) where T <: Real
    pos_center = [get_center(arena)...,]
    r = sqrt(sum(pos_center.^2))
    get_circle_intersection(pos_center, r, point, θ)
end

function get_circle_intersection(origin::Vector{T}, r::T, point::Vector{T}, θ::T) where T <: Real
    (a,b) = (cos(θ),sin(θ))
    (x0,y0) = origin
    (x1,y1) = point

    # find t
    A = a^2 + b^2
    B = 2*a*(x1-x0) + 2*b*(y1-y0)
    C = (x1-x0)^2 + (y1-y0)^2 - r^2
    t = (-B+sqrt(B^2 - 4*A*C))/(2*A)
    point + t*[a,b]
end

 function get_view(pos::Vector{T}, θ::T, arena::Arena{T};kwargs...) where T <: Real
    pos_center = [get_center(arena)...,]
    get_view(pos, θ, pos_center;kwargs...)
 end

 function get_view(pos::Vector{T}, θ::T, pos_center;fov::T=T(π/2)) where T <: Real
    θ1 = θ-fov/2
    θ2 = θ+fov/2
    get_view(pos, θ1, θ2, pos_center)
 end

 function get_view(pos::Vector{T}, θ1::T, θ2::T, pos_center) where T <: Real
    r = sqrt(sum(pos_center.^2))
    xq = zeros(T,2,2)
    xq[1,:] = get_circle_intersection(pos_center, r, pos, θ1)
    xq[2,:] = get_circle_intersection(pos_center, r, pos, θ2)
    xq .-= pos_center
    θq = atan.(xq[:,2], xq[:,1])
    # make sure we get the smallest range
    #θq[θq.<0] .+= 2π
    Qqm1 = extrema(θq)
    Δ1 = Qqm1[2] - Qqm1[1]
    θq2 = copy(θq)
    θq2[θq2.<0] .+= T(2π)
    Qqm2 = extrema(θq2)
    Δ2 = Qqm2[2] - Qqm2[1]
    if Δ1 < Δ2
        Qqm = Qqm1
    else
        Qqm = Qqm2
    end
    # if both angles are negative, shift them
    if all(Qqm .< 0)
        Qqm = Qqm .+ T(2π)
    end
    return [Qqm], Tuple{T,T}[]
 end

 function get_smallest_wedge(θ1, θ2)
    θq = [θ1,θ2]
    Qqm1 = extrema(θq)
    Δ1 = Qqm1[2] - Qqm1[1]
    θq2 = copy(θq)
    θq2[θq2.<0] .+= 2π
    Qqm2 = extrema(θq2)
    Δ2 = Qqm2[2] - Qqm2[1]
    if Δ1 < Δ2
        Qqm = Qqm1
    else
        Qqm = Qqm2
    end
    # if both angles are negative, shift them
    if all(Qqm .< 0)
        Qqm = Qqm .+ 2π 
    end
    return Qqm
 end

 """
 Find the closest point from `po` to the line from `p0` to `p1`
 """
 function find_line_intersection(po, p0, p1)
    x0,y0 = p0
    x1,y1 = p1
    x,y = po
    a = (x-x0)*(x1-x0) + (y-y0)*(y1-y0)
    a /= (x1-x0)^2 + (y1-y0)^2
    (x0 + a*(x1-x0), y0+a*(y1-y0))
 end

 function find_line_intersection(po, v::Vector{T}, p0, p1) where T <: Real
    x0,y0 = p0
    x1,y1 = p1
    xp,yp = po
    vx,vy = v
    # make sure we respect the direction of v
    s = one(T)
    if abs(vx) > 0
        # this can return Inf if vx==vy and y1-y0 == x1 -x0
        a = (yp - y0 + s*(x0-xp)*vy/vx)/(y1-y0 - s*(x1-x0)*vy/vx)
    elseif abs(vy) > 0
        a = (xp - x0 + s*(y0-yp)*vx/vy)/(x1-x0 - s*(y1-y0)*vx/vy)
    end
    Δx = a*(x1-x0)
    Δy = a*(y1-y0)

    #a = max(min(a,one(T)),zero(T))

    (x0 + Δx, y0 + Δy)
 end

 function project_to_view(pos, θ::T, p, fov::T) where T <: Real
    vv = [cos(θ), sin(θ)]
    # orthogonal to line of sight
    vp = [cos(θ+π/2), sin(θ+π/2)]

    #find the distance by projecting onto vv
    dd = (p .- pos)'*vv
    # the the projected plane size at this distance
    l0 = dd*tan(fov/2)
    dl = 2*l0
    dp = (p .- pos)'*vp
    dd, dp
    # associate an angle with this
    ϕ = atan(dp, dd)
    dd, dp, ϕ
 end

 shift_angle(θ) = ifelse(θ < 0, θ+2π, θ)

 """
 Find the smallest angular distance between θ1 and θ2. 
 
 If the angles are separated by more than 180 degree, switch the relationship. 
 That is if θ1 is π/2 and θ2 is 3π/2+0.1, then we order θ2 before θ1
representing θ2 as -0.47π

Thanks to Mistral.ai for helping me sort this one out!
 """
 function order_angles(θ1::T, θ2::T) where T <: Real
    d1 = θ1 - θ2
    i1,i2 = (0,0)
    if abs(d1) > π
        if θ2 > θ1
            θs = θ2-2π
            θb = θ1
            i1 = 2
            i2 = 1
        else
            θb = θ2
            θs = θ1-2π
            i2 = 2
            i1 = 1
        end
    else
        if θ2 > θ1
            θs = θ1
            θb = θ2
            i1 = 1
            i2 = 2
        else
            θs = θ2
            θb = θ1
            i1 = 2
            i2 = 1
        end
    end
    (θs, θb), (i1,i2)
 end

 """
 Return true if θ1 is ordered before θ2
 """
 function compare_angles(θ1::T, θ2::T) where T <: Real
    d1 = θ1 - θ2
    res = false
    if abs(d1) > π
        if θ2 > θ1
            res = false
        else
            res = true 
        end
    else
        if θ2 > θ1
            res = true
        else
            res = false
        end
    end
    res
 end


 """
 Merge overlapping views
 """
 function consolidate_view(oo::Vector{Tuple{T,T}}) where T <: Real
    _oo,_ii = order_angles(oo[1]...)
    oo2 = [_oo]
    ii = [((1,_ii[1]),(1,_ii[2]))]
    for (i1,oc1) in enumerate(oo[2:end])
        merged = false
        for (i2,ooe) in enumerate(oo2)
            if compare_angles(oc1[1], ooe[2])
                amin,i_min = ifelse(compare_angles(ooe[1], oc1[1]), (ooe[1],i2), (oc1[1],i1+1))
                amax,i_max = ifelse(compare_angles(oc1[2], ooe[2]), (ooe[2],i2), (oc1[2],i1+1))
                oo2[end],jk = order_angles(amin,amax)
                ii[end]  = ((i_min,jk[1]), (i_max,jk[2]))
                merged = true
                break
            end
        end
        if !merged
            _oo,_ii = order_angles(oc1...)
            push!(oo2, _oo)
            push!(ii, ((i1+1,_ii[1]), (i1+1,_ii[2])))
        end
    end
    oo2,ii
 end

 function find_closest_neighbour(points::Vector{Tuple{T,T}}, idx::Int64, pos::Union{Tuple{T,T},Vector{T}}, θ::T,fov::T) where T <: Real
    nn = length(points)
    j1 = mod(idx,nn)+1
    _, dp1,_ = project_to_view(pos, θ, points[j1], fov)
    j2 = mod(idx-2,nn)+1
    _, dp2,_ = project_to_view(pos, θ, points[j2], fov)
    if abs(dp1)  < abs(dp2)
        p1 = points[j1]
    else
        p1 = points[j2] 
    end
    p1
 end

 function is_occluded(oo::Vector{Tuple{T,T}}) where T <: Real
    keep = fill(false, length(oo))
    for (i,oc2) in enumerate(oo)
        keep[i] = !is_occluded(oc2, oo[keep])
    end
    keep
 end

 function is_occluded(oc2::Tuple{T,T}, oo::Vector{Tuple{T,T}}) where T <: Real
    _is_occluded = false
    for oc1 in oo
        if oc1 == oc2
            continue
        end
        if oc1[1] < oc2[1] < oc2[2] < oc1[2]
            # oc2 is contained with oc1; remove i# overlapping
            _is_occluded = true
            break
        end
    end
    _is_occluded
 end

 """
    is_occluded(point::Tuple{T,T}, points::Vector{Tuple{T,T}}, pos::Union{Tuple{T,T},Vector{T}}, θ::T) where T <: Real

Return true if `point` is occluded by at least one edge formed by neighbouring `points` from the perspective of `pos`
with view direction `θ`.
 """
 function is_occluded(point::Tuple{T,T}, points::Vector{Tuple{T,T}}, pos::Union{Tuple{T,T},Vector{T}}, θ::T) where T <: Real
    np = length(points)
    v = [cos(θ),sin(θ)]
    #check whether a point if behind any edge
    occluded = false
    d = point .- pos
    dnn = [(d./norm(d))...]
    for (i1,i2) in zip(circshift(1:np,1), 1:np)
        p1 = points[i1]
        p2 = points[i2]
        m = points[i2] .- points[i1]
        m = m./norm(m)
        # check if point is on this line
        dv = point .- p1
        dv = dv./norm(dv)
        mq = dv[1]*m[1] + dv[2]*m[2] 
        if mq ≈ 1.0f0 
            continue
        end
        ϕ = atan(m[2],m[1])
        n = [cos(ϕ-T(π/2)), sin(ϕ -T(π/2))]
        # project onto normal vector
        dq = d[1]*n[1] + d[2]*n[2]
        # find the intersection
        pp = find_line_intersection(pos, dnn, p1,p2)
        dn = pp .- pos
        dpn = dn[1]*n[1] + dn[2]*n[2]
        if (p1 <= pp <= p2) || (p2 <= pp <= p1)
            if abs(dq) > abs(dpn)
                occluded = true
                break
            end
        end
    end
    occluded
 end

 """
    inview(p::Tuple{T,T}, pos::Vector{T}, θ::T, fov::T) where T <: Real

Return true if the point `p` is within the view cone given by angle `theta` and field-of-view angle `fov` from points `pos`
 """
 function inview(p::Tuple{T,T}, pos::Vector{T}, θ::T, fov::T) where T <: Real
    v = [cos(θ), sin(θ)]
    dp = p .- pos
    dp = dp./norm(dp)
    cosϕ = dp[1]*v[1] + dp[2]*v[2]
    cosϕ >= cos(fov/2)
 end

 function inview(p::Vector{Tuple{T,T}}, pos::Vector{T}, θ::T, fov::T) where T <: Real
    res = fill(false, length(p))
    for (ii,pp) in enumerate(p)
        res[ii] = inview(pp, pos, θ, fov)
    end
    res
 end

 function get_obstacle_intersection(pos::Vector{T}, θ::AbstractVector{T}, arena::AbstractMazeArena{T},θ0::T, fov::T) where T <: Real
    # TODO Do distance to walls as well
    w,h = extent(arena)
    obstacle_points = get_obstacle_points(arena)
    res = [inview(points, pos, θ0, fov) for points in obstacle_points]
    wall_points = [(zero(T), zero(T)),(w, zero(T)), (w, h), (zero(T),h)]
    pps = Vector{Tuple{T,T}}(undef, length(θ))
    dms = zeros(T, length(θ))
    dbs = zeros(T,length(θ))
    oid = fill(0, length(θ))

    for (jj,_θ) in enumerate(θ)
        pp,d_min,d_pp = get_intersection(pos, _θ, wall_points, θ0,fov)
        for (ii,(rr,points)) in enumerate(zip(res, obstacle_points))
            if any(rr)
                _pp, _dm,_d_pp = get_intersection(pos, _θ, points, θ0,fov)
                if _dm < d_min
                    pp = _pp
                    d_min = _dm
                    d_pp = _d_pp
                    oid[jj] = ii
                end
            end
        end
        pps[jj] = pp
        dms[jj] = d_min 
        dbs[jj] = oid[jj] + d_pp
    end
    pps, dms, dbs 
 end

 function get_obstacle_intersection(pos::Vector{T}, θ::AbstractVector{T}, arena::Arena{T},θ0::T,fov::T) where T <: Real
    w,h = extent(arena)
    wall_points = [(zero(T), zero(T)),(w, zero(T)), (w, h), (zero(T),h)]
    pps = Vector{Tuple{T,T}}(undef, length(θ))
    dms = zeros(T, length(θ))
    dbs = zeros(T,length(θ))
    for (jj,_θ) in enumerate(θ)
        pps[jj],dms[jj],dbs[jj] = get_intersection(pos, _θ, wall_points, θ0,fov)
    end
    pps, dms, dbs
 end
 
 """
 Find the first point where a ray from `pos` along `θ` intersects with the polygon represntedy by `points`
 """
 function get_intersection(pos::Vector{T}, θ::T, points::Vector{Tuple{T,T}}, θ0::T,fov::T) where T <: Real
    ppos = convert(Vector{Float64}, pos)
    θp = Float64(θ)
    θ0p = Float64(θ0)
    fovp = Float64(fov)
    np = length(points)
    v = [cos(θp),sin(θp)]
    v0 = [cos(θ0p),sin(θ0p)]
    # loop through each edge
    d_min = Inf
    pp = (NaN, NaN)
    ϵ = eps(Float64)
    db = zero(T)
    d_pp = zero(T)
    # keep track of the distance along the periphery of the obstacle
    for (i1,i2) in zip(circshift(1:np,1), 1:np)
        p1 = convert(Tuple{Float64, Float64}, points[i1])
        p2 = convert(Tuple{Float64,Float64}, points[i2])
        # make sure that at least point is within the cone
        
        _pp = find_line_intersection(ppos, v, p1,p2)
        _dpp = _pp .- ppos
        _dpp = _dpp./norm(_dpp)
        # we need the angle between _dpp and v
        # ϕ is in allocentric coordinates, θ
        #ϕ = atan(_dpp[2],_dpp[1])
        # angle between the point and the ray
        cosϕ = _dpp[1]*v0[1] + _dpp[2]*v0[2]
        # are we within the cone of visibility?
        # this should in general always be true, except we can travel along v in both directions
        # in find_line_intersection. So we need to make that we are still within the cone
        vq = cosϕ >= cos(fov/2)-2ϵ
        #vq = compare_angles(θ0-fov/2, ϕ) && compare_angles(ϕ, θ0+fov/2)
        # use only valid points, i.e. points actually on the edge
        if vq && ((p1 <= _pp <= p2) || (p2 <= _pp <= p1))
            d = norm(_pp .- pos)
            if d < d_min
                d_min = d
                pp = _pp
                d_pp = db + norm(pp .- p1)
            end
        end
        # distance from the first point to pp
        db += norm(p2 .- p1)
    end
    # add some texture to the texture
    # just to a sinusoid here with 5 periods
    db = T(0.1) + T(0.1*sin(2π*5*d_pp/db))
    convert(Tuple{T,T}, pp), T(d_min),db 
 end

 function get_view(pos::Vector{T}, θ::T, arena::MazeArena{T};fov::T=T(π/2)) where T <: Real
    # check if any obstacle is in the view
    obstructed_angles = Tuple{T,T}[]
    #obstructed_points = Tuple{Tuple{T,T}, Tuple{T,T}}[]
    obstructed_points = Tuple{T,T}[]
    obstructed_obstacle = Int64[]
    obstacle_points = get_obstacle_points(arena)
    pos_center = [get_center(arena)...]
    for (ii,obstacle) in enumerate(obstacle_points)

        a_min,a_max = (T(Inf),T(-Inf))
        # assume polygon, i.e. each successive point is connnected
        angles = T[]
        points = Tuple{T,T}[]
        # TODO: Assign view bins to each pillar
        for (jj,p) in enumerate(obstacle)
            dd,dp,ϕ = project_to_view(pos, θ, p, fov)
            dq = norm(pos .- p)
            if dd < 0
                continue
            end
            valid = false 
            if ϕ < -fov/2
                ϕ = -fov/2
                v = [cos(θ+ϕ), sin(θ+ϕ)]
                # need to find the point
                p1 = find_closest_neighbour(obstacle, jj, pos, θ, fov)
                pp = find_line_intersection(pos, v, p, p1)
                # make sure we are somewhere on the line
                if (p < pp < p1) || (p1 < pp < p)
                    valid = true
                end
            elseif ϕ > fov/2
                ϕ = fov/2
                v = [cos(θ+ϕ), sin(θ+ϕ)]
                p1 = find_closest_neighbour(obstacle, jj, pos, θ, fov)
                pp = find_line_intersection(pos, v, p, p1)
                if (p < pp < p1) || (p1 < pp < p)
                    valid = true
                end
                # need to find the point
            else
                pp = p
                valid = true
            end

            # what part of the obstacle does this correspond to?

            # we only care about this obstacle if it is in front of the agent
            push!(angles, mod(ϕ + θ,2π))
            if valid
                push!(points, pp)
            end
        end
        if !isempty(angles)
            angle_min, i_min = findmin(angles)
            angle_max, i_max = findmax(angles)
            # convert this to allocentric
            angle_min, angle_max = first(first(get_view(pos, angle_min, angle_max, pos_center)))
            (angle_min,angle_max),i12 = order_angles(angle_min, angle_max)
            i_min,i_max = [i_min,i_max][[i12...]]
        else
            angle_min, angle_max = (zero(T), zero(T))
            i_min,i_max = (0,0)
        end
        append!(obstructed_points, points)
        append!(obstructed_obstacle, fill(ii, length(points)))
        if angle_min < angle_max
            #angle_min, angle_max, i12 = order_angles(angle_min, angle_max)
            push!(obstructed_angles, (angle_min, angle_max))
            #push!(obstructed_points, (points[i_min], points[i_max]))
        end
    end
    # filter out obstructed points
    obstructed_points_filtered = Tuple{T,T}[]
    candidate_obstacles =  unique(obstructed_obstacle)
    for (pp,iq) in zip(obstructed_points, obstructed_obstacle)
        _is_obstructed = false
        for jq in candidate_obstacles
            _is_obstructed = is_occluded(pp, obstacle_points[jq], pos, θ)
            if _is_obstructed
                break
            end
        end
        if !_is_obstructed
            push!(obstructed_points_filtered, pp)
        end
    end
    ocid = Vector{Int64}[]
    if length(obstructed_angles) > 1
        sort!(obstructed_angles, by=a->a[1],lt=compare_angles)
        obstructed_angles,ocid = consolidate_view(obstructed_angles)
        # also re-arrange the points
        #obstructed_points = [(obstructed_points[i1][j1], obstructed_points[i2][j2]) for ((i1,j1),(i2,j2)) in ocid]
        # the obstructed points should now contain the points in front. i.e. no obscured by other occlusions
    end
    θ12 = get_view(pos,θ,pos_center;fov=fov)
    # θ12 are in allocentric coordinates
    θ1,θ2 = first(first(θ12))
    θs = [(θ1,θ2)]
    if length(obstructed_angles) > 0
        
        # check if the view is completely blocked
        for o_angle in obstructed_angles
            if θ1 ≈ o_angle[1] && o_angle[2] ≈ θ2
                # single occlusion completely blocking the view
                return Tuple{T,T}[], Tuple{T,T}[]
            end
        end
        θs = Tuple{T,T}[]
        # θ1 forms the minimum. Include an angle from θ1 to the first touch point
        if compare_angles(θ1, obstructed_angles[1][1])
        #if θ1 < obstructed_angles[1][1]
        #if shift_angle(θ1) < shift_angle(obstructed_angles[1][1])
            push!(θs, order_angles(θ1, obstructed_angles[1][1])[1])
        end
        for (a,b) in zip(1:length(obstructed_angles)-1, 2:length(obstructed_angles))
            push!(θs, order_angles(obstructed_angles[a][2], obstructed_angles[b][1])[1])
        end
        if compare_angles(obstructed_angles[end][2], θ2)
        #if θ2 > obstructed_angles[end][2]
        #if shift_angle(θ2) > shift_angle(obstructed_angles[end][2])
            push!(θs, order_angles(obstructed_angles[end][2],θ2)[1])
        end
    end
    θs, obstructed_points_filtered
 end

function (trial::NavigationTrial{T})(;rng=Random.default_rng(),Δθstep::T=T(π/4), p_stay=T(1/3), p_hd=T(1/4), fov=T(π/3), do_rescale=true, binsize=1.0, binsize_wall=binsize, kwargs...) where T <: Real
    # random initiarange(-T(π), stop=T(π), step=π/4)li
    arena = trial.arena
    n_place_bins = num_floor_bins(arena;binsize=binsize) 
    n_gaze_bins = num_surface_bins(arena;binsize=binsize,binsize_wall=binsize_wall)
    # recompute based on binsize
    ncols = round(Int64, arena.ncols*arena.colsize/binsize)
    nrows = round(Int64, arena.nrows*arena.rowsize/binsize)
    arena_diam = sqrt(sum(abs2, extent(arena)))
    θf = range(zero(T), stop=T(2π), step=T(π/4))
    nsteps = rand(rng, trial.min_num_steps:trial.max_num_steps)
    conjunction = fill(T(0.2), n_gaze_bins, n_place_bins, nsteps)
    position = zeros(T,2,nsteps)
    (i,j) = get_coordinate(trial.arena;rng=rng) 
    position[:,1] = get_position(i,j,trial.arena)
    viewf = zeros(T, length(trial.angular_pref.μ),nsteps)
    head_direction = zeros(T, length(trial.angular_pref.μ), nsteps)
    movement = zeros(T,4,nsteps)  # up,down,left,right
    texture = zeros(T, 16, nsteps)
    gaze = zeros(T, 2*16, nsteps) # flat dimensions to be consistent with the rest of the code
    gazem = zeros(T, 2, nsteps) # mean gaze
    # the estimate distance from the agent to each of the view points
    # if we are encoding distance like this, we need a way to indicate unknown distance for points 
    # not in view
    # alternatively, fix the number of points for the view (like a fovea) and then encode the distance
    # for each of these fixed points
    dist = zeros(T, 16,nsteps)

    θ = rand(rng, θf)
    head_direction[:,1] = trial.angular_pref(θ)
    inputs_outputs = union(trial.inputs, trial.outputs)
    compute_view = :view in inputs_outputs
    compute_distance = (:distance in inputs_outputs) || (:texture in inputs_outputs) || (:gaze in inputs_outputs)

    if compute_view
        θq,_ = get_view(position[:,1],θ, trial.arena;fov=fov,kwargs...)
        for _θq in θq
            viewf[:,1] .= mean(trial.angular_pref(range(_θq[1], stop=_θq[2],length=10)),dims=2)
        end
    end
    # use the full field of view here

    if compute_distance
        θs = range(θ-fov/2, stop=θ+fov/2, length=size(dist,1))
        xp, dp, texture[:,1] = get_texture(position[:,1], θs, arena, θ, fov)
        dist[:,1] .= dp./arena_diam
        for (j,_pp) in enumerate(xp)
            gaze[2*(j-1)+1:2*j,1] .= _pp
        end
        # digitize central gaze
        gg = dropdims(mean(reshape(gaze[:,1], 2, 16)[:,8:9],dims=2),dims=2)
        gidx = assign_surface_bin(gg...,arena;binsize=binsize,binsize_wall=binsize_wall)[3]
        _pidx,pidx = assign_bin(position[:,1]...,arena;binsize=binsize)
        conjunction[gidx,pidx,1] = T(0.8)
        gazem[:,1] = gg
    end
    
    Δθ = T.([-Δθstep, zero(T), Δθstep])
    for k in 2:nsteps
        θ += get_head_direction(Δθstep,θ;rng=rng,p_stay=p_stay) 
        i1,j1 = get_coordinate(i,j,trial.arena,θ;rng=rng, p_hd=p_hd)
        if i1 - i > 0
            movement[1,k] = i1-i 
        elseif i1 - i < 0
            movement[2,k] = i-i1 
        end
        if j1 - j > 0
            movement[4,k] = j1-j
        elseif j1 - j < 0
            movement[3,k] = j-j1
        end
        i = i1
        j = j1
        position[:,k] = get_position(i,j,trial.arena)
        head_direction[:,k] = trial.angular_pref(θ)
        # get view angles
        if compute_view
            _θq = get_view(position[:,k],θ, trial.arena;kwargs...)
            θq, posq = _θq
            for _θq in θq
                viewf[:,k] .+= mean(trial.angular_pref(range(_θq[1], stop=_θq[2],length=10)),dims=2)
            end
        end
        if compute_distance
            θs = range(θ-fov/2, stop=θ+fov/2, length=size(dist,1))
            xp, dp,texture[:,k] = get_texture(position[:,k], θs, arena,θ, fov)
            dist[:,k] .= dp./arena_diam
            for (jj,_pp) in enumerate(xp)
                gaze[2*(jj-1)+1:2*jj,k] .= _pp
            end
            gg = dropdims(mean(reshape(gaze[:,k], 2, 16)[:,8:9],dims=2),dims=2)
            _,dm, gidx = assign_surface_bin(gg...,arena;binsize=binsize,binsize_wall=binsize_wall)
            _pidx,pidx = assign_bin(position[:,k]...,arena;binsize=binsize)
            # convert to linear index
            if gidx > n_gaze_bins
                @show dm
            end

            conjunction[gidx,pidx,k] = T(0.8)
            gazem[:,k] = gg
        end
    end
    if compute_distance
        idx = findall(!isfinite, dist)
        for ii in idx
            ii0 = ii.I[1]
            if ii.I[1] < size(dist,1)-2
                ii2  = ii.I[1]+2
                ii1 = ii.I[1]+1
                dm = dist[ii.I[1]+2, ii.I[2]] - dist[ii.I[1]+1, ii.I[2]]
                dist[ii] = dist[ii.I[1]+1,ii.I[2]] - dm
                # TODO: Do the same for gaze
                dg = gaze[2*(ii2-1)+1:2*(ii2)] .- gaze[2*(ii1-1)+1:2*ii1]
                gaze[2*(ii0-1)+1:2*ii0,ii.I[2]] = gaze[2*(ii1-1)+1:2*ii1, ii.I[2]] .- dg
            else
                ii2  = ii.I[1]-1
                ii1 = ii.I[1]-2
                dm = dist[ii.I[1]-1, ii.I[2]] - dist[ii.I[1]-2, ii.I[2]]
                dist[ii] = dist[ii.I[1]-1,ii.I[2]] + dm

                dg = gaze[2*(ii2-1)+1:2*(ii2)] .- gaze[2*(ii1-1)+1:2*ii1]
                gaze[2*(ii0-1)+1:2*ii0,ii.I[2]] = gaze[2*(ii1-1)+1:2*ii1, ii.I[2]] .+ dg
            end
        end
    end
    # hack: because of finite precision, we sometimes get Infs in the distance. Use a crude interpolation for now
    #normalize position and gaze to the size of the maze 
    if do_rescale
        position ./= [trial.arena.ncols*trial.arena.colsize, trial.arena.nrows*trial.arena.rowsize]
        position .= 0.8*position .+ 0.05 # rescale from 0.05 to 0.85 to avoid saturation
        gaze[1:2:end,:] ./= trial.arena.ncols*trial.arena.colsize
        gaze[2:2:end,:] ./= trial.arena.nrows*trial.arena.rowsize
        gazem ./= [trial.arena.ncols*trial.arena.colsize, trial.arena.nrows*trial.arena.rowsize]
        gazem .= 0.8*gazem .+ 0.05


        texture ./= max(maximum(texture),1.0)
    end
    conjunction = reshape(conjunction, size(conjunction,1)*size(conjunction,2), size(conjunction,3))
    position, head_direction, viewf, movement, dist, texture, gaze, conjunction, gazem
end

function restore_scale(trialstruct::NavigationTrial{T}, X::Matrix{T}) where T <: Real
    width = trialstruct.arena.ncols*trialstruct.arena.colsize
    height = trialstruct.arena.nrows*trialstruct.arena.rowsize
    [width, height].*(X .- T(0.05))./T(0.8)
end

function num_inputs(trialstruct::NavigationTrial)
    n = 0
    if :head_direction in trialstruct.inputs
        n += length(trialstruct.angular_pref.μ)
    end
    if :view in trialstruct.inputs
        n += length(trialstruct.angular_pref.μ)
    end
    if :movement in trialstruct.inputs
        n += 4
    end
    if :position in trialstruct.inputs
        n += 2
    end
    if :distance in trialstruct.inputs
        n += 16
    end
    if :texture in trialstruct.inputs
        n += 16
    end
    if :gaze in trialstruct.inputs
        n += 32
    end
    n
end

function output_sizes(trialstruct::NavigationTrial;binsize=trialstruct.arena.colsize, binsize_wall=binsize)
    n = fill(0, length(trialstruct.outputs))
    for (ii,output) in enumerate(trialstruct.outputs)
        if output == :conjunction
            #TODO: What discretization to use here 
            # We can use the same discretization that we use for the floor
            # For instance, if the floor uses 10 × 10 bins, we use 10 bins for
            # each of the walls, and I guess two bins for the pillars
            n_surface = num_surface_bins(trialstruct.arena;binsize=binsize,binsize_wall=binsize_wall)
            n_place = num_floor_bins(trialstruct.arena;binsize=binsize)
            n[ii] = n_surface*n_place
        elseif output == :head_direction
            n[ii] = length(trialstruct.angular_pref.μ)
        elseif output == :view
            n[ii] = length(trialstruct.angular_pref.μ)
        elseif output == :movement
            n[ii] = 4
        elseif output == :position
            n[ii] = 2
        elseif output == :distance
            n[ii] = 16
        elseif output == :texture
            n[ii] = 16
        elseif output == :gaze
            n[ii] = 2 # output gaze is only the main direction
        end
    end
    n
end

function num_outputs(trialstruct::NavigationTrial;binsize=trialstruct.arena.colsize, binsize_wall=binsize)
    n = 0
    if :conjunction in trialstruct.outputs
        #TODO: What discretization to use here 
        # We can use the same discretization that we use for the floor
        # For instance, if the floor uses 10 × 10 bins, we use 10 bins for
        # each of the walls, and I guess two bins for the pillars
        n_surface = num_surface_bins(trialstruct.arena;binsize=binsize,binsize_wall=binsize_wall)
        n_place = num_floor_bins(trialstruct.arena;binsize=binsize)
        n += n_surface*n_place
    end
    if :head_direction in trialstruct.outputs
        n += length(trialstruct.angular_pref.μ)
    end
    if :view in trialstruct.outputs
        n += length(trialstruct.angular_pref.μ)
    end
    if :movement in trialstruct.outputs
        n += 4
    end
    if :position in trialstruct.outputs
        n += 2
    end
    if :distance in trialstruct.outputs
        n += 16
    end
    if :texture in trialstruct.outputs
        n += 16
    end
    if :gaze in trialstruct.outputs
        n += 2 # output gaze is only the main direction
    end
    n
end

function compute_error(trialstruct::NavigationTrial{T}, output::Array{T,3}, output_true::Array{T,3}) where T <: Real
    # we should differentiate depending on what the output is. If it is just position, an error is the deviation from the cell center
    # error for each position
    nout = output_sizes(trialstruct)
    err = fill(T(NaN), length(nout), size(output,2)*size(output,3))
    offset_a = 0
    for (i,(output_t, output_true_t)) in enumerate(zip(eachslice(output,dims=3), eachslice(output_true,dims=3)))
        # find the sequence length
        idxc = findfirst(output_true_t .> T(0.05))
        idx1 = idxc.I[2]
        idx2 = findfirst(dropdims(sum(output_true_t,dims=1),dims=1) .<= zero(T))
        if idx2 === nothing
            idx2 = size(output_t,2)
        else
            idx2 = idx2 - 1
        end
        Δ = output_t[:,idx1:idx2]-output_true_t[:,idx1:idx2]
        offset = 0
        for (jj,_n) in enumerate(nout)
            vidx = (offset+1):(offset+_n)
            err[jj, offset_a+1:offset_a+idx2-idx1+1] = dropdims(sum(abs2,Δ[vidx,:,:],dims=1),dims=1)
            offset += _n
        end
        offset_a += idx2-idx1+1
    end
    err[:,1:offset_a]
end

function performance(trialstruct::NavigationTrial{T}, output::Array{T,3}, output_true::Array{T,3};require_fixation=false) where T <: Real
    err = compute_error(trialstruct, output, output_true)
    # correct if the error is less than half the cell width
    ppq = zeros(T,size(err,1))
    nq = fill(0, size(err,1))
    # need to scale the size appropriately
    wc = 0.8/trialstruct.arena.ncols
    wr = 0.8/trialstruct.arena.nrows
    sΔ = (wc/2)^2 + (wr/2)^2
    for _err in eachcol(err)
        idx = isfinite.(_err)
        ppq[idx] .+= _err[idx] .< sΔ
        nq[idx] .+= 1
    end
    ppq./nq 
end

function generate_trials(trial::NavigationTrial{T}, ntrials::Int64,dt; rng=Random.default_rng(), rseed=1, hd_step=T(π/4), fov=T(π/2),p_stay=T(1/3), p_hd=T(1/4),binsize=trial.arena.colsize, binsize_wall=binsize) where T <: Real
    Δθstep = T(hd_step)
    p_stay = T(p_stay)
    dt = T(dt)
    p_hd = T(p_hd)
    fov = T(fov)
    args = [(:ntrials, ntrials),(:dt, dt), (:rng, rng), (:rseed, rseed), (:Δθstep, Δθstep),
            (:fov, fov),(:p_stay, p_stay),(:p_hd, p_hd),(:binsize,binsize),(:binsize_wall, binsize_wall)]
    defaults = Dict{Symbol,Any}(:Δθstep=>T(π/4), :fov=>T(π/2),:p_stay=>T(1/3),:p_hd=>T(1/4), :binsize=>trial.arena.colsize, :binsize_wall=>trial.arena.colsize)
    h = signature(trial)
    for (k,v) in args
        if !(k in keys(defaults)) || v != defaults[k]
            h = CRC32c.crc32c(string(v), h)
        end
    end
    pushfirst!(args, (:trialstruct, trial))
    Random.seed!(rng, rseed)
    ninputs = num_inputs(trial)
    noutputs = num_outputs(trial;binsize=binsize,binsize_wall=binsize_wall)
    max_nsteps = trial.max_num_steps
    TrialIterator(
        function data_provider()
            input = -1*ones(T, ninputs, max_nsteps, ntrials)
            output = -1*ones(T, noutputs, max_nsteps, ntrials)
            output_mask = zeros(T, noutputs, max_nsteps, ntrials)
            for i in 1:ntrials
                position, head_direction,viewfield,movement,dist,texture,gaze,conjunction,gazem = trial(;rng=rng,Δθstep=Δθstep,fov=fov, p_stay=p_stay, p_hd=p_hd,binsize=binsize, binsize_wall=binsize_wall)
                offset = 0
                if :view in trial.inputs
                    input[offset+1:offset+size(viewfield,1), 1:size(viewfield,2),i]  .= viewfield
                    offset += size(viewfield,1)
                end
                if :head_direction in trial.inputs
                    input[offset+1:offset+size(head_direction,1), 1:size(head_direction,2),i]  .= head_direction
                    offset += size(head_direction,1)
                end
                if :movement in trial.inputs
                    input[offset+1:offset+size(movement,1), 1:size(movement,2),i]  .= movement
                    offset += size(movement,1)
                end
                if :distance in trial.inputs
                    input[offset+1:offset+size(dist,1), 1:size(dist,2),i]  .= dist 
                    offset += size(dist,1)
                end
                if :texture in trial.inputs
                    input[offset+1:offset+size(texture,1), 1:size(texture,2),i] .= texture 
                    offset += size(texture,1)
                end
                if :gaze in trial.inputs
                    input[offset+1:offset+size(gaze,1), 1:size(gaze,2),i] .= texture 
                    offset += size(gaze,1)
                end

                offset = 0
                if :position in trial.outputs
                    output[offset+1:offset+size(position,1), 1:size(position,2),i] .= position
                    offset += size(position,1)
                end
                if :head_direaction in trial.outputs
                    output[offset+1:offset+size(head_direction,1), 1:size(head_direction,2),i]  .= head_direction
                    offset += size(head_direction,1)
                end
                if :distance in trial.outputs
                    output[offset+1:offset+size(dist,1), 1:size(dist,2),i]  .= dist 
                    offset += size(dist,1)
                end
                if :texture in trial.outputs
                    output[offset+1:offset+size(texture,1), 1:size(texture,2),i] .= texture 
                    offset += size(texture,1)
                end
                if :gaze in trial.outputs
                    output[offset+1:offset+size(gazem,1), 1:size(gaze,2),i] .= gazem
                    offset += size(gazem,1)
                end
                if :conjunction in trial.outputs
                    output[offset+1:offset+size(conjunction,1), 1:size(conjunction,2), i] .= conjunction
                end
                output_mask[:,1:size(position,2),i] .= one(T)
            end
            input,output,output_mask
        end,NamedTuple(args), h)
end
