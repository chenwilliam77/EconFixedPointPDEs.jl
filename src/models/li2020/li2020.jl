"""
```
Li2020{T} <: AbstractRepModel{T}
```

The `Li2020` type defines the structure of the simple New Keynesian DSGE
model described in 'Bayesian Estimation of DSGE Models' by Sungbae An and Frank
Schorfheide.

### Fields

#### Parameters
* `parameters::Vector{AbstractParameter}`: Vector of all time-invariant model
  parameters.

* `keys::OrderedDict{Symbol,Int}`: Maps human-readable names for all model
  parameters to their indices in `parameters`.

#### Inputs to Measurement and Equilibrium Differential Equations

The following fields are dictionaries that map human-readable names to indices.

* `stategrid::OrderedDict{Symbol,Int}`: Maps each state variable to its dimension on a Cartesian grid

* `endogenous_variables::OrderedDict{Symbol,Int}`: Maps endogenous variables
    calculated during the solution of differential equations to an index

* `endogenous_variables_augmented::OrderedDict{Symbol,Int}`: Maps endogenous variables
    that can be calculated after solving the equilibrium differential equations

* `exogenous_shocks::OrderedDict{Symbol,Int}`: Maps each shock to a column in
  the measurement equation

* `observables::OrderedDict{Symbol,Int}`: Maps each observable to a row in the
  model's measurement equation matrices.

* `pseudo_observables::OrderedDict{Symbol,Int}`: Maps each pseudo-observable to
  a row in the model's pseudo-measurement equation matrices.

#### Model Specifications and Settings

* `spec::String`: The model specification identifier, \"an_schorfheide\", cached
  here for filepath computation.

* `subspec::String`: The model subspecification number, indicating that some
  parameters from the original model spec (\"ss0\") are initialized
  differently. Cached here for filepath computation.

* `settings::Dict{Symbol,Setting}`: Settings/flags that affect computation
  without changing the economic or mathematical setup of the model.

* `test_settings::Dict{Symbol,Setting}`: Settings/flags for testing mode

#### Other Fields

* `rng::MersenneTwister`: Random number generator. Can be is seeded to ensure
  reproducibility in algorithms that involve randomness (such as
  Metropolis-Hastings).

* `testing::Bool`: Indicates whether the model is in testing mode. If `true`,
  settings from `m.test_settings` are used in place of those in `m.settings`.

* `observable_mappings::OrderedDict{Symbol,Observable}`: A dictionary that
  stores data sources, series mnemonics, and transformations to/from model units.
  DSGE.jl will fetch data from the Federal Reserve Bank of
  St. Louis's FRED database; all other data must be downloaded by the
  user. See `load_data` and `Observable` for further details.

* `pseudo_observable_mappings::OrderedDict{Symbol,PseudoObservable}`: A
  dictionary that stores names and transformations to/from model units. See
  `PseudoObservable` for further details.
"""
mutable struct Li2020{T} <: AbstractNLCTModel{T}
    parameters::ParameterVector{T}                         # vector of all time-invariant model parameters
    keys::OrderedDict{Symbol,Int}                          # human-readable names for all the model
                                                           # parameters and steady-states
    stategrid::OrderedDict{Symbol,Int}                     # dimension number of state variable

    differential_variables::OrderedDict{Symbol,Int}
    endogenous_variables::OrderedDict{Symbol,Int}
    exogenous_shocks::OrderedDict{Symbol,Int}
    observables::OrderedDict{Symbol,Int}
    pseudo_observables::OrderedDict{Symbol,Int}

    spec::String                                           # Model specification number (eg "m990")
    subspec::String                                        # Model subspecification (eg "ss0")
    settings::Dict{Symbol,Setting}                         # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}                    # Settings/flags for testing mode
    rng::MersenneTwister                                   # Random number generator
    testing::Bool                                          # Whether we are in testing mode or not

    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end

description(m::Li2020) = "Julia implementation of model defined in 'Public Liquidity and Financial Crises' (2020) by Wenhao Li: Li2020, $(m.subspec)"

"""
`init_model_indices!(m::Li2020)`

Arguments:
`m:: Li2020`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and observables.
"""
function init_model_indices!(m::Li2020)
    # Stategrid
    stategrid = collect([:w])

    # Exogenous shocks
    exogenous_shocks = collect([:K_sh, :N_sh]) # capital shock K, liquidity shock N

    # Variables for differential equations
    differential_variables = collect([:p])

    # Endogenous variables
    endogenous_variables = collect([:Q, :ψ, :xK, :yK, :xg, :yg, :σp, :σ, :σh])

    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    for (i,k) in enumerate(stategrid);              m.stategrid[k]              = i end
    for (i,k) in enumerate(exogenous_shocks);       m.exogenous_shocks[k]       = i end
    for (i,k) in enumerate(endogenous_variables);   m.endogenous_variables[k]   = i end
    for (i,k) in enumerate(differential_variables); m.differential_variables[k] = i end
    for (i,k) in enumerate(observables);            m.observables[k]            = i end
    for (i,k) in enumerate(pseudo_observables);     m.pseudo_observables[k]     = i end
end

function Li2020(subspec::String = "ss0";
                custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                testing = false)

    # Model-specific specifications
    spec               = split(basename(@__FILE__),'.')[1]
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = Li2020{Float64}(
            # model parameters and steady state values
            Vector{AbstractParameter{Float64}}(), OrderedDict{Symbol,Int}(),

            # model indices
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),

            spec,
            subspec,
            settings,
            test_settings,
            rng,
            testing,
            OrderedDict{Symbol,Observable}(),
            OrderedDict{Symbol,PseudoObservable}())

    # Set settings
    model_settings!(m)
    # default_test_settings!(m)
    for custom_setting in values(custom_settings)
        m <= custom_setting
    end

    # Set observable and pseudo-observable transformations
    # init_observable_mappings!(m)
    # init_pseudo_observable_mappings!(m)

    # Initialize parameters
    init_parameters!(m)

    init_model_indices!(m)
    init_subspec!(m)

    return m
end


"""
```
init_parameters!(m::Li2020)
```
Initializes the model's parameters.
"""
function init_parameters!(m::Li2020)
    # Productivity
    m <= parameter(:AH, 0.15, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(),
                   Uniform(0., 1e3), fixed = false, description = "Banker's productivity")
    m <= parameter(:AL, 0.125, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household's productivity")

    # Bank run and fire sales
    m <= parameter(:λ, 0.025, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.),
                   fixed = false, description = "Probability of liquidity shock")
    m <= parameter(:β, 0.25, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.), fixed = false,
                   description = "Probability of running on a deposit account")
    m <= parameter(:α, 0.21, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.), fixed = false,
                   description = "Illiquidity discount on capital during fire sale")
    m <= parameter(:ϵ, 1e-3, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.), fixed = false,
                   description = "Fraction of remaining wealth for bankers after bankruptcy")
    m <= parameter(:θ, 1e-5, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.), fixed = false,
                   description = "Probability of non-zero exposure to liquidity shock")

    # Shadow value of liquidity
    m <= parameter(:π, 0.19, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.), fixed = false,
                   description = "Illiquidity discount on illiquid safe asset")

    # Macro parameters
    m <= parameter(:δ, 0.1, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.),
                   fixed = false, description = "Depreciation rate of capital")
    m <= parameter(:ρ, 0.04, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Discount rate")
    m <= parameter(:σK, 0.033, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3), fixed = false,
                   description = "Volatility of capital shocks")
    m <= parameter(:χ, 3., (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3), fixed = false,
                   description = "Adjustment cost of internal investment function")

    # Stationary distribution
    m <= parameter(:η, 0.05, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.), fixed = false,
                   description = "Rate of retirement for bankers")
end



"""
```
model_settings!(m::Li2020)
```
creates the model's settings.
"""
function model_settings!(m::Li2020)

    # Investment functions
    m <= Setting(:Φ, quadratic_investment, "Internal investment function")
    m <= Setting(:∂Φ, derivative_quadratic_investment_li2020, "Derivative of internal investment function")

    # Numerical settings
    m <= Setting(:v₀, 3e-8, "Parameter for damping function")
    m <= Setting(:vp_function, x -> get_setting(m, :v₀) ./ x, "Dampling function to avoid corners")
    m <= Setting(:p₀_perturb, 1e-14, "Perturbation of boundary condition for q at w = 0")
    m <= Setting(:N, 100, "Grid size")
    m <= Setting(:stategrid_method, :exponential, "Type of grid to construct for state variables")
    m <= Setting(:stategrid_dimensions, OrderedDict{Symbol, Tuple{Float64, Float64, Int}}(:w => (1e-3, 1., get_setting(m, :N))),
                 "Information about the dimensions of the state space")
    m <= Setting(:stategrid_splice, 0.2, "Li (2020) constructs the grid in two parts. This value is where the first half stops.")
    m <= Setting(:boundary_conditions, OrderedDict{Symbol, Vector{Float64}}(:p => [0.; 0.]),
                 "Boundary conditions for differential equations.")
    m <= Setting(:max_iterations, 12, "Maximum number of fixed point iterations")
    m <= Setting(:ode_reltol, 1e-4, "Relative tolerance for ODE integration")
    m <= Setting(:ode_abstol, 1e-12, "Absolute tolerance for ODE integration")
    m <= Setting(:essentially_one, 0.999,
                "If ψ is larger than this value, then we consider it essentially one for some numerical purposes.")

    # Other settings for initialization
    m <= Setting(:nojump_parameters, [:ρ, :AH, :AL, :σK, :χ, :δ], "Keys of parameters used when solving the no-jump equilibrium.")

    # Calibration targets
    m <= Setting(:avg_gdp, 0.145, "Average GDP")
    m <= Setting(:liq_gdp_ratio, 0.39, "Average Liquidity/GDP ratio in the data")
    m <= Setting(:gov_bond_gdp_level, 0.3, "Ratio of Government Bonds to GDP")

    # Simulation settings
    m <= Setting(:dt, 1. / 12., "Simulation interval as a fraction of a 1 year")

    # ODE integrator
    m <= Setting(:ode_integrator, DP5(), "Numerical ODE integrator for no-jump solution") # DP5 is Matlab's ode45
end
