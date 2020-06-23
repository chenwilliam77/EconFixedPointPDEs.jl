using Test
include(joinpath(dirname(@__FILE__), "../../src/includeall.jl"))

@testset "Initializing one-dimensional derivatives" begin
    m = Li2020()
    delete!(get_derivatives(m), :∂p_∂w)
    delete!(get_derivatives(m), :∂²p_∂w²)
    init_derivatives!(m, Dict{Symbol, Vector{Int}}(:p => [1]))
    @test haskey(get_derivatives(m), :∂p_∂w)
    @test !haskey(get_derivatives(m), :∂²p_∂w²)
    m = Li2020()
    delete!(get_derivatives(m), :∂p_∂w)
    delete!(get_derivatives(m), :∂²p_∂w²)
    init_derivatives!(m, Dict{Symbol, Vector{Int}}(:p => [1, 2]))
    @test haskey(get_derivatives(m), :∂p_∂w)
    @test haskey(get_derivatives(m), :∂²p_∂w²)
    m = Li2020()
    delete!(get_derivatives(m), :∂p_∂w)
    delete!(get_derivatives(m), :∂²p_∂w²)
    init_derivatives!(m, Dict{Symbol, Vector{Int}}(:p => [2]))
    @test !haskey(get_derivatives(m), :∂p_∂w)
    @test haskey(get_derivatives(m), :∂²p_∂w²)
end

@testset "Initializing two-dimensional derivatives" begin
    m = Li2020()
    m.state_variables[:x] = 2
    delete!(get_derivatives(m), :∂p_∂w)
    delete!(get_derivatives(m), :∂²p_∂w²)

    # Just request 1 derivative
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(1, 0)]))
    @test haskey(get_derivatives(m), :∂p_∂w)
    @test !haskey(get_derivatives(m), :∂²p_∂w²)
    delete!(get_derivatives(m), :∂p_∂w)
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(1, 0), (2, 0)]))
    @test haskey(get_derivatives(m), :∂p_∂w)
    @test haskey(get_derivatives(m), :∂²p_∂w²)
    delete!(get_derivatives(m), :∂p_∂w)
    delete!(get_derivatives(m), :∂²p_∂w²)
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(2, 0)]))
    @test !haskey(get_derivatives(m), :∂p_∂w)
    @test haskey(get_derivatives(m), :∂²p_∂w²)
    delete!(get_derivatives(m), :∂²p_∂w²)

    # Just request 1 derivative
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(0, 1)]))
    @test haskey(get_derivatives(m), :∂p_∂x)
    @test !haskey(get_derivatives(m), :∂²p_∂x²)
    @test length(get_derivatives(m)) == 1
    delete!(get_derivatives(m), :∂p_∂x)
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(0, 1), (0, 2)]))
    @test haskey(get_derivatives(m), :∂p_∂x)
    @test haskey(get_derivatives(m), :∂²p_∂x²)
    @test length(get_derivatives(m)) == 2
    delete!(get_derivatives(m), :∂p_∂x)
    delete!(get_derivatives(m), :∂²p_∂x²)
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(0, 2)]))
    @test haskey(get_derivatives(m), :∂²p_∂x²)
    @test length(get_derivatives(m)) == 1
    delete!(get_derivatives(m), :∂²p_∂x²)

    # Now do multiple variables
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(1, 1)]))
    @test haskey(get_derivatives(m), :∂²p_∂w∂x)
    @test length(get_derivatives(m)) == 1
    delete!(get_derivatives(m), :∂²p_∂w∂x)
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(1, 0), (0, 1)]))
    @test haskey(get_derivatives(m), :∂p_∂w)
    @test haskey(get_derivatives(m), :∂p_∂x)
    @test length(get_derivatives(m)) == 2
    delete!(get_derivatives(m), :∂p_∂w)
    delete!(get_derivatives(m), :∂p_∂x)
    init_derivatives!(m, Dict{Symbol, Vector{Tuple{Int,Int}}}(:p => [(2, 0), (0, 2)]))
    @test haskey(get_derivatives(m), :∂²p_∂w²)
    @test haskey(get_derivatives(m), :∂²p_∂x²)
    @test length(get_derivatives(m)) == 2
    delete!(get_derivatives(m), :∂²p_∂w²)
    delete!(get_derivatives(m), :∂²p_∂x²)
end

@testset "Standard derivatives for continuous-time models" begin
    @test standard_derivs(1) == Vector{Int}([1, 2])
    @test standard_derivs(2) == Vector{Tuple{Int, Int}}([(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)])
end
