"""
```
BruSan{T} <: AbstractNLCTFPModel{T}
```

The `BruSan` type defines the structure of a standard "Brunnermeier-Sannikov" style model
augmented to feature jumps.

### Fields

#### Parameters
* `parameters::Vector{AbstractParameter}`: Vector of all time-invariant model
  parameters.

* `keys::OrderedDict{Symbol,Int}`: Maps human-readable names for all model
  parameters to their indices in `parameters`.

#### Inputs to Measurement and Equilibrium Functional Equations

The following fields are dictionaries that map human-readable names to indices.

* `state_variables::OrderedDict{Symbol,Int}`: Maps each state variable to its dimension on a Cartesian grid

* `endogenous_variables::OrderedDict{Symbol,Int}`: Maps endogenous variables
    calculated during the solution of functional equations to an index

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
mutable struct BruSan{T} <: AbstractNLCTFPModel{T}
    parameters::ParameterVector{T}                         # vector of all time-invariant model parameters
    keys::OrderedDict{Symbol,Int}                          # human-readable names for all the model
                                                           # parameters and steady-states
    state_variables::OrderedDict{Symbol,Int}               # dimension number of state variable

    functional_variables::OrderedDict{Symbol,Int}
    derivatives::OrderedDict{Symbol,Int}
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

description(m::BruSan) = "Julia implementation of model defined in 'Public Liquidity and Financial Crises' (2020) by Wenhao Li: BruSan, $(m.subspec)"

"""
`init_model_indices!(m::BruSan)`

Arguments:
`m:: BruSan`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and observables.
"""
function init_model_indices!(m::BruSan)
    # State_Variables
    state_variables = collect([:η])

    # Exogenous shocks
    exogenous_shocks = collect([:K_sh]) # capital shock K

    # Variables for functional equations
    functional_variables = collect([:q, :vₑ, :vₕ])

    # Derivatives of variables
    init_derivatives!(m, OrderedDict{Symbol, Vector{Int}}(:q => standard_derivs(1), :vₑ => standard_derivs(1), :vₕ => standard_derivs(1)),
                      state_variables)
    derivatives = keys(get_derivatives(m))

    # Endogenous variables
    endogenous_variables = collect([:φ_e, :φ_h, :σ_q, :μ_q, :σ_η, :μ_η, :ςₑ, :ςₕ, :ι, :Φ, :μ_K, :dr_f,
                                    :σ_vₑ, :σ_vₕ, :μ_vₑ, :μ_vₕ])

    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    for (i,k) in enumerate(state_variables);        m.state_variables[k]      = i end
    for (i,k) in enumerate(exogenous_shocks);       m.exogenous_shocks[k]     = i end
    for (i,k) in enumerate(endogenous_variables);   m.endogenous_variables[k] = i end
    for (i,k) in enumerate(functional_variables);   m.functional_variables[k] = i end
    for (i,k) in enumerate(derivatives);            m.derivatives[k]          = i end
    for (i,k) in enumerate(observables);            m.observables[k]          = i end
    for (i,k) in enumerate(pseudo_observables);     m.pseudo_observables[k]   = i end
end

function BruSan(subspec::String = "ss0";
                custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                testing = false)

    # Model-specific specifications
    spec               = split(basename(@__FILE__),'.')[1]
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = BruSan{Float64}(
            # model parameters and steady state values
            Vector{AbstractParameter{Float64}}(), OrderedDict{Symbol,Int}(),

            # model indices
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),

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
init_parameters!(m::BruSan)
```
Initializes the model's parameters.
"""
function init_parameters!(m::BruSan)
    # Technology
    m <= parameter(:aₑ, 1., (0., Inf), (0., 1e3), ModelConstructors.Untransformed(),
                   Uniform(0., 1e3), fixed = false, description = "Banker's productivity")
    m <= parameter(:aₕ, 0.7, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household's productivity")
    m <= parameter(:δ, 0.1, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.),
                   fixed = false, description = "Depreciation rate of capital")
    m <= parameter(:σ, 0.033, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3), fixed = false,
                   description = "Volatility of capital shocks")
    m <= parameter(:χ₁, 1., (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3), fixed = false,
                   description = "Adjustment cost parameter 1 of internal investment function")
    m <= parameter(:χ₂, 1., (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3), fixed = false,
                   description = "Adjustment cost parameter 2 of internal investment function")

    # Preferences
    m <= parameter(:ρₑ, 0.04, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Expert discount rate")
    m <= parameter(:ρₕ, 0.04, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household discount rate")
    m <= parameter(:νₑ, 0.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Expert death rate")
    m <= parameter(:νₕ, 0.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household death rate")
    m <= parameter(:γₑ, 1.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Expert risk aversion coefficient")
    m <= parameter(:γₕ, 1.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household risk aversion coefficient")
    m <= parameter(:ψₑ, 1.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Expert inverse Frisch elasticity")
    m <= parameter(:ψₕ, 1.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household inverse Frisch elasticity")

    # Stationary distribution
    m <= parameter(:τ, 0.0, (0., 1.), (0., 1.), ModelConstructors.Untransformed(), Uniform(0., 1.), fixed = false,
                   description = "Redistributive wealth tax on experts")
end

"""
```
model_settings!(m::BruSan)
```
creates the model's settings.
"""
function model_settings!(m::BruSan)

    # Numerical settings for grid
    m <= Setting(:N, 150, "Grid size")
    m <= Setting(:stategrid_method, :exponential, "Type of grid to construct for state variables")
    m <= Setting(:stategrid_dimensions, OrderedDict{Symbol, Tuple{Float64, Float64, Int}}(:η => (1e-3, 1. - 1e-3, get_setting(m, :N))),
                 "Information about the dimensions of the state space")
    m <= Setting(:stategrid_splice, 0.1, "The grid is constructed in two parts. This value is where the first half stops.")

    # Numerical settings for no jump equilibrium
    m <= Setting(:boundary_conditions, OrderedDict{Symbol, Vector{Float64}}(:q => [0.; 0.]),
                 "Boundary conditions for differential equations.")
    m <= Setting(:ode_reltol, 1e-4, "Relative tolerance for ODE integration")
    m <= Setting(:ode_abstol, 1e-12, "Absolute tolerance for ODE integration")
    m <= Setting(:ode_integrator, Tsit5(), "Numerical ODE integrator for no-jump solution")
    m <= Setting(:backup_ode_integrators, [DP5(), RK4(), Euler()], "Back up numerical ODE integrators for no-jump solution")
    m <= Setting(:interpolant, Gridded(Linear()), "Interpolation method for value functions")
    m <= Setting(:max_q₀_perturb, 1.05, "Maximum perturbation for the initial value of q at η = 0")

    # Numerical settings for functional iteration
    m <= Setting(:tol, 1e-4, "Tolerance for functional iteration")
    m <= Setting(:learning_rate, 0.4, "Learning rate for the update of functional variables after each loop")
    m <= Setting(:max_iter, 40, "Maximum number of loops during functional iteration")
    m <= Setting(:error_method, :total_error, "Method for calculating error at the end of each loop during functional iteration")
    m <= Setting(:q₀_perturb, 0.006, "Perturbation of boundary condition for q at 1e-3")
    m <= Setting(:κp_grid, Vector{Float64}(undef, 0),
                 "Vector of guesses for κp used during iteration for new κp and xK within each functional loop")
    m <= Setting(:p_interpolant, Gridded(Linear()), "Interpolant method for fitted p.")
    m <= Setting(:xK_interpolant, Gridded(Linear()), "Interpolant method for xK after solving xK for each guess of κp.")
    m <= Setting(:xg_interpolant, Gridded(Linear()), "Interpolant method for xg after solving xK and κp.")
    m <= Setting(:κp_interpolant, Gridded(Linear()), "Interpolant method for κp after solving xK and κp.")
    m <= Setting(:Q̂_interpolant, Gridded(Linear()), "Interpolant method for Q̂.")
    m <= Setting(:inside_iteration_nlsolve_tol, 1e-6, "Tolerance for nlsolve in the inside iteration")
    m <= Setting(:inside_iteration_nlsolve_max_iter, 400, "Maximum number of iterations for nlsolve for the inside iteration")
    m <= Setting(:xg_tol, 2e-3, "Lower but permissible tolerance for xg fixed point in the inside iteration")
    m <= Setting(:yg_tol, 1e-5, "Lowest permissible value for yg")
    m <= Setting(:p_tol, 1e-8, "Tolerance when calculating p after completing an inside iteration")
    m <= Setting(:firesale_bound, 0.99, "Upper bound for size of firesale jumps that do not wipe out a banker's net worth")
    m <= Setting(:firesale_interpolant, Gridded(Linear()), "Interpolant method for the firesale value.")
    m <= Setting(:N_GH, 10, "Number of nodes for Gauss-Hermite quadrature.")
    m <= Setting(:Q̂_tol, 1e-5, "Tolerance for approximation of Q̂ via Gauss-Hermite quadrature")
    m <= Setting(:Q̂_max_it, 1000, "Maximum number of iterations for approximation of Q̂.")
    m <= Setting(:dt, 1. / 12., "Simulation interval for Q̂ approximation as a fraction of a 1 year")
end
