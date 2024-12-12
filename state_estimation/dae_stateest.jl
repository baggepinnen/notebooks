### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 42056c78-903c-4dc5-a0b0-6c67ea0ce6f5
# ╠═╡ show_logs = false
using Pkg; Pkg.add(url="https://github.com/baggepinnen/SeeToDee.jl"); Pkg.add(url="https://github.com/baggepinnen/LowLevelParticleFilters.jl"); Pkg.add([
	"Test"
	"Random"
	"LinearAlgebra"
	"Statistics"
	"StaticArrays"
	"Distributions"
	"Plots"
	"NonlinearSolve"
])

# ╔═╡ c08f7fbc-4638-11ee-3944-434b1f71c165
using LowLevelParticleFilters, SeeToDee, Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions, Plots, NonlinearSolve

# ╔═╡ 624402d7-7a79-44bb-a813-32edef49db6e
md"""
# State estimation for high-index DAEs

### Introduction

In a [previous blog post](https://info.juliahub.com/tune-kalman-filter), we spoke about how to tune a Kalman filter, a widely used state estimator for linear dynamical systems. Unfortunately, not all practically occurring systems are represented sufficiently well with linear dynamics affected by Gaussian noise. In fact, such assumptions are almost never satisfied completely, even if they may be good enough in many situations. In this post, we will take a step up in sophistication and consider nonlinear systems expressed as differential-algebraic equations, and explore the challenges and opportunities that arise when performing state estimation and virtual sensing for this richer set of systems.

### Differential-Algebraic Equations

Modern equation-based modeling tools, such as ModelingToolkit, allow the user to quickly and in a component-based fashion, model complex real-world systems. Such tools often produce differential-algebraic equation models (DAEs), i.e., the dynamics may include both differential equations _and_ algebraic equations. The standard Kalman filter, and most of its nonlinear extensions, do not deal with algebraic equations, so how do we estimate the state for such a DAE system?

Before we proceed, we define some notion. Let the dynamic equations of the system we want to estimate the state of be written as
```math
Mẋ(t) = f(x(t), u(t))
```
where ``M`` is called a "mass matrix", ``x`` is the state (technically, since we are dealing with DAEs, it's a _descriptor_) and ``u`` is the input. If we take ``M = I``, we have a standard ODE system, if ``M`` is invertible, we can easily convert this system to an ODE system like ``ẋ= M^{-1} f(x, u)``, so the interesting case is when ``M`` is _singular_. A commonly occurring special case is when ``M`` has the form
```math
\begin{bmatrix}
I & 0 \\
0 & 0
\end{bmatrix}
```
i.e., the system can be written as 
```math
\begin{bmatrix}
ẋ_δ \\
0
\end{bmatrix} = f(x, u)
```
where ``ẋ_δ`` includes the derivatives of the differential state variables only. In this form, we have ``f`` containing a number of ordinary differential equations, followed by one or several algebraic equations ``0 = g(x, u)``.

Differential-equation solvers that support DAEs often make use of a nonlinear root finder that continuously solves the equation ``0 = g(x, u)`` in order to simulate the system.

Now, how do we estimate the state descriptor ``x`` in a DAE system? This question is much more nuanced than when we're dealing with systems of ordinary differential equations only. How come?

When we design a state estimator, we must come up with some model of the disturbances that affect the dynamics and the measurements. This disturbance model is usually designed to include both unmeasured inputs that affect the system, but also to mitigate the effects of _model error_. Modeling any physical system perfectly well is impossible, and we must somehow account for the fact that our dynamic equations are an imperfect representation of the true dynamics that govern the system. As detailed in the previous post, the Kalman filter models these disturbances as Gaussian noise, but for nonlinear DAE systems, things can quickly become more challenging.
"""

# ╔═╡ d1a4d317-8b27-4f7e-bb6c-02742cf5ab58
md"""
## An example
To ground the continued discussion, we introduce an example system, the _Cartesian pendulum_. The most common representation of the dynamics of a simple pendulum is ``\ddot\phi = -\frac{g}{l}\sin\phi + \tau``. In this representation, we use the angle ``\phi`` of the pendulum as our coordinate, and ``\tau`` is some torque applied to the joint. However, we can model the pendulum also in Cartesian (``x,y``) coordinates, in which case the dynamic equations may be represented as
```math
\begin{align}
\dot x   &= x_v \\
\dot y   &= y_v \\
\dot x_v &= -λx + f_x \\
\dot y_v &= -λy + f_y - g \\
0        &= x^2 + y^2 - l^2
\end{align}
```
where ``x,y`` are the coordinates of the tip of the pendulum, ``f_x, f_y`` are forces acting on the tip, ``λ`` is the tension force in the pendulum rod, ``g`` is the gravitiational acceleration and ``l`` is the length of the pendulum. Notice how there's no differential equation for ``λ``, instead there is an algebraic equation that says that the tip of the pendulum is at a distance ``l`` from the pivot
```math
0 = x^2 + y^2 - l^2 \quad\Leftrightarrow\quad \sqrt{x^2 + y^2} = l
```
from this algebraic equation, it's possible to compute the tension force (if we know the mass of the pendulum).

An animation of this pendulum is shown below, this animation is produced by the very last code in this post, but the figure is reproduced here to establish a picture of how the system looks and feels.
"""

# ╔═╡ 1573960e-89b2-4a0a-b5a3-033dc7c414d9
md"""
Some numerical integrators do not like this representation of the pendulum, the problem is that it has a high DAE _index_. The index of a DAE is roughly the number of times you need to differentiate the algebraic equation with respect to time before you get a differential equation. This representation of the pendulum has index 3 since our algebraic equation is a _position constraint_. If we differentiate this equation, we get another algebraic equation that says that the _velocity_ is tangential to the circle the pendulum is tracing. If we differentiate one more time, we get an algebraic equation that says that the _acceleration_ of the pendulum is centripetal (points towards the center point). Finally, if we differentiate one more time, we get a differential equation. Now, why don't we always differentiate sufficiently many times to get a differential equation so that we can forget about DAEs altogether? The problem is that while the differentiated system is mathematically equivalent, we now have to integrate one extra equation, and integration is associated with some small numerical error. Over time, this integrated error grows, causing our original index-3 constraint ``\sqrt{x^2 + y^2} = l`` to be violated. The more times we differentiate (and thus have to integrate), the bigger this problem becomes. This can be compared to trying to estimate a position by integrating a noisy accelerometer measurement twice, this typically leads to a very noisy estimate of the position. The figure below illustrates the problem of integrating small random errors.
"""

# ╔═╡ 9510e624-99d0-45bb-acfd-4da7810b7f9c
let
	Random.seed!(123)
	Ts = 0.5
	r = randn(80)
	ri = cumsum(r) .* Ts
	rii = cumsum(ri) .* Ts
	plot([r ri rii], label=["Random errors" "Once integrated" "Twice integrated"], framestyle=:zerolines, size=(200,200))
end

# ╔═╡ 0a1ac989-e739-414f-af3a-7af31bf771a6
md"""
If all we are doing is simulating the system, this problem can be mitigated by making the integrator tolerance very small so that the numerical drift is small, but what about when we are performing state estimation?
"""

# ╔═╡ fe5001a3-5037-4100-8f4d-f947d49c727c
md"""
### A new foe has appeared

Imagine that we obtain measurements of the ``x,y`` coordinates of the pendulum, perhaps from a camera. These are going to be noisy and associated with some probability-density function. Lets say that the noise is Gaussian and distributed according to ``N(0, R_y)`` where ``R_y`` is a 2×2 covariance matrix. We immediately spot a problem, the measurement noise has two degrees of freedom, but the real system only has one. Our noisy measurements are likely going to lie outside of the circle ``\sqrt{x^2 + y^2} = l`` and thus violate the constraint! If we differentiate the algebraic equations twice, to obtain an index-1 DAE, we hide this problem since we can solve the algebraic equation that relates ``x,y`` and ``\lambda`` through centripetal acceleration for a feasible ``λ`` given any pair of ``x, y`` measurements, but our original index-3 position constraint will be silently violated.
"""

# ╔═╡ a12d6f59-8b4a-423d-9979-37b089a23de8
md"""
The problems don't end here though, how do we represent the posterior distrubution of the state? The Kalman filter represents this as a multivariate normal distribution, but no normal distrubution has support only on a circle, and we can thus never represent the posterior uncertainty as a multivariate normal in the ``x,y`` space. We canot even compute the standard mean of several distinct points on the circle without getting a point that lies inside the circle and thus violates the constraint (This is what the Unscented Kalman filter (UKF) whould have done for this system).

How do we solve this problem? Unfortunately, there is not a single easy answer, and the best approach is context dependent. For the pendulum, the answer is obvious, model the system in polar coordinates instead of Cartesian coordinates. For other systems, the answer is not so straightforward. If we know that we are modeling an index-3 system like the Cartesian pendulum, we could imagine the following list of partial solutions
- Use an integrator that handles index 3 algebraic equations and keep the original formulation of the dynamics without differentiating the algebraic equation. Then project each the measurement onto the feasible space of ``\sqrt{x^2 + y^2} = l``. 
- Represent posterior distributions voer the state as samples rather than as normal distributions.
- Represent the input disturbances as dynamically feasible inputs to the system (in spirit with the previous blog post). This prevents the system from moving away from the index-3 constraint manifold due to disturbances.


In the rest of this post, we will make use of all three of the above suggested points. However, it would be possible to make use of an index-1 formulation provided that the dynamically feasible disturbances are modeled already during the index-reduction stage.
"""

# ╔═╡ 5df4068d-a4f2-4f74-9ec4-0ad921cf4242
md"""
## In practice

Let's stop talking and implement some state estimation! We will use a [particle filter](https://en.wikipedia.org/wiki/Particle_filter) as our state estimator, a sequential Monte-Carlo method that represents distributions as collections of samples ("particles"). The particle filter is implemented in [LowLevelParticleFilters.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl) The integration of the continuous-time dynamics will be done using a direct-collocation method from the package [SeeToDee.jl](https://github.com/baggepinnen/SeeToDee.jl).

We start by loading the required pacakges:
"""

# ╔═╡ d56d8c16-ed6c-4736-85bf-0c60a8ef0826
begin
	Random.seed!(0)
	gr(fmt=:png)
end

# ╔═╡ c1ec5b9c-2bf7-4bdf-b5ad-2990f5826069
md"""
We then define the dynamic equations, our ``Mẋ(t) = f(x(t), u(t))``
"""

# ╔═╡ 5e003076-c235-4355-899f-037250afa915
"A Cartesian pendulum of length 1 in DAE form"
function pend(state, f, p, t=0)
    x,y,u,v,λ = state
    g = 9.82
    SA[
        u 				# = ẋ
        v 				# = ẏ
        -λ*x + f[1] 	# = u̇
        -λ*y - g + f[2] # = v̇
        x^2 + y^2 - 1 	# = 0
    ]
end

# ╔═╡ ebef9c1c-c9cc-4886-9e93-71409da485e3
md"""
Next up, we define some of the properties of the state estimator. We integrate with a fixed interval of ``T_s = 0.01`` seconds, and use ``N=100`` particles to represent our state distribution. We also choose some covariance properties of the measurement noise ``\sigma_y`` and the input disturbance ``\sigma_w``. We will model the dynamic disturbance as entering like Cartesian forces, similar to how the control input enters. This is a dynamically feasible disturbance which the DAE integrator can integrate the same way as it integrates the control input forces. Since the integrator we use treats inptus as constant during the integration step (Zero-order-Hold), this implicitly makes the assumption that disturbance forces are also piecewise constant. Since the integration time step ``T_s`` is small, this is an acceptable model in this case and we will get a good approximation to a continuous-time noise process.
"""

# ╔═╡ 924095f4-51e6-4b0a-b045-5282f2c2b433
begin
	nx = 4 			# Dinemsion of differential state
	na = 1 			# Number of algebraic variables
	nu = 2 			# Dinemsion of input
	ny = 2 			# Dinemsion of measurements
	Ts = 0.01
	N  = 100 		# Number of particles
	p  = nothing 	# We have no parameters to keep track of
	σy = 0.1 		# Measurement noise std dev
	σy_model = 0.2  # We model slightly larger measurement noise than we actually have
	σw = 5.0 		# Dynamics noise std dev
	dg = MvNormal(zeros(2), σy_model)
end;

# ╔═╡ ff56ae63-ab60-4ce1-b4b3-23bdff078f68
md"""
We then implement a function that models the measurement process. We assume that we can measure the coorinates ``x,y``. Here we sample a random value from the measurement-noise distribution and then project the result measurement onto the circle traced by the pendulum. The resulting distribution is thus no longer Gaussian, but that is not a problem for a particle filter!
"""

# ╔═╡ bdde4cfd-f546-4c89-9428-6478c296e238
@inbounds function measurement(x,u,p,t, noise=false)
    if noise
        y = SA[x[1] + σy*randn(), x[2] + σy*randn()]
        y ./ norm(y) # Measurements projected to be consistent with algebraic constraint
    else
        SA[x[1], x[2]]
    end
end

# ╔═╡ cffd10c8-c111-4288-aad5-4ec0e5e3d60e
md"""
Here, we chose to apply measurement noise with standard deviation ``σ_y = `` $(σy) to our measurements, but let the model in the filter use ``σ_{y_{model}} = `` $(σy_model), this is a common trick in particle filtering to improve the performance of the estimator without increasing the number of particles, and thus the computational requirements, too much.
"""

# ╔═╡ 2bcf761a-a598-4dc7-af3d-e5eb32e97846
function measurement_likelihood(x, u, y, p, t)
    logpdf(dg, measurement(x, u, p, t)-y) # A simple measurement model with normal additive noise
end

# ╔═╡ bd6c047d-9346-46f7-b59c-0e5d8a8ba877
md"""
Next, we discretize the continuous-time dynamics using the direct-collocation method. This method will happily accept algebraic equations, even if they are index 3.
"""

# ╔═╡ 6995f5c3-e878-4f54-b18f-b6ebc8de14bd
discrete_dynamics = SimpleColloc(pend, Ts, nx, na, nu; n=3, abstol=1e-8, solver = NewtonRaphson())

# ╔═╡ 30aee77f-0517-4019-98e4-ff3e47f5a8eb
md"""
We then compute an initial condition and create an initial state distribution from which the particle filter will sample particles when initializing the filter.
"""

# ╔═╡ 8ad859e3-950d-4313-9820-ed437e3e2190
begin
	x0 = SeeToDee.initialize(discrete_dynamics, [1.0,0,0,0,0], p) # This makes susre that the initial condition satisfies the algebraic equation.
	x0 = SVector(x0...)
	d0 = MvNormal(x0, 0)   # Initial state Distribution, zero variance
end;

# ╔═╡ 29d34dcd-b609-4c08-9182-ccfeaba6f246
md"""
We also need to define the dynamics of each particle. The function `dynamics` below makes use of the discretized dynamic equations, but adds noise handling if the filter calls for dynamics noise to be added. The noise is sampled from a normal distribution, and added to the control input.
"""

# ╔═╡ 18457a70-0994-4d75-ac89-31e3d31d9fd1
function dynamics(x, u, p, t, noise=false)
    if noise
        # Add input noise (acts like Cartesian forces)
        un = @SVector randn(length(u))
        xp = discrete_dynamics(x, u + σw*un, p, t)
    else
        xp = discrete_dynamics(x, u, p, t)
    end
    xp
end

# ╔═╡ 78194091-3a3d-469f-a483-412c01d393ab
md"""
We now pacakge all the settings and functions into a `AdvancedParticleFilter` struct
"""

# ╔═╡ fa7a6f4a-84da-4110-ace6-ce72e3dab051
apf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, p, d0, threads=false);

# ╔═╡ 9a17ba77-ed08-409d-86bf-9a799e8c01cb
md"""
### Simulation and filtering

With the filter constructed, we can simulate a draw from the posterior distribution implied by the dynamics and noise distributions. We generate some interesting-looking control inputs `u`, and pass this to the `simulate` function. This function will add noise to the simulation in accodance with the functions `dynamics` and `measurement` defined above. To perform the actual filtering, we call `forward_trajectory` which performs filtering along the entire trajecory all at once. The result of the filtering is a solution object that contains all the particles. In order to easily visualize the result, we compute the mean over all the particles at each time step using the function `mean_trajectory`.[^1] The result of the simulation and the filtering is shown below.

[^1]: Computing the mean of points on a circle will in general lead to a point which is not on the circle, we thus project the computed mean back onto the circle before plotting. The theoretically correct way of computing this mean is to compute a generalized mean under the metric implied by measureing distances along the circle. 
"""

# ╔═╡ 35d86ddb-ee2d-47f7-9f97-474635458ac2
begin
t = range(0, length=250, step=Ts)
# Generate input
ui = [10sign.(sin.(5 .* t.^2)) randn(length(t))] |> eachrow .|> collect
x,u,y = simulate(apf, ui; dynamics_noise=true, measurement_noise=true)

u = SVector{2, Float64}.(u) # For slightly higher-performance filtering
y = SVector{2, Float64}.(y)

X = reduce(hcat, x)'
Y = reduce(hcat, y)'
U = reduce(hcat, u)'

@time "Particle Filtering" sol = forward_trajectory(apf, u, y, p);

Xmean = mean_trajectory(sol)
for i in axes(Xmean, 1)
	Xmean[i, 1:2] ./= norm(Xmean[i, 1:2]) # Project onto circle
end

state_names = ["x" "y" "xᵥ" "yᵥ" "λ"]
	
figX = plot(X, label=state_names, layout=nx+na)
plot!(Xmean, label="Xmean")
# plot!(Xmode, label="Xmode")
figY = plot(Y, label="Y")
figU = plot(U, label="U")

plot(figX, figY, figU, layout=@layout([a{0.6h}; b{0.2h}; c{0.2h}]), size=(800,1100))
end

# ╔═╡ 53f68faf-695e-4e65-9a8b-6e4dbb95eb34
md"""
Instead of plotting only the mean of the particles, we could plot all the individual particles as well. Below, we do this and plot the true evolution of the state on top.
"""

# ╔═╡ 467caa21-6188-4b2a-914d-1b329ee99413
begin
Xpart = stack(sol.x)
fig_part = plot(layout=nx+na, plot_title="Particle trajectories")
for i = 1:nx+na
    scatter!(Xpart[i, :, :]', label="X$i", sp=i, m=(1,:black, 0.5), lab=false)
end
	plot!(X, label=state_names, c=1)
fig_part
end

# ╔═╡ 4c8afc2f-c2a7-4a04-847d-f5f38011bebc
md"""
Due to the random force input, we see that the tension force in the pendulum rod ``λ`` is rather noisy. We also see that some of the particle trajectories appears to suddenly die out, in particular in the plot for ``y``. This is a characteristic of the particle filter, it internally makes use of a resampling procedure in order to discard particles that appear to be unlikely representations of the true state given the measurements that are received. 
"""

# ╔═╡ 254d3981-a8f5-41ac-9baf-81e7d2bcd4ed
md"""
To finish off, we render an animation that compares the evolution of the pendulum to that obtained using the filtering. 
"""

# ╔═╡ 708c0bef-c76b-495f-898e-927354c32cd3
anim = @animate for i in 1:length(t)
	x, y = X[i, 1:2]    	# True coordinates
	xm, ym = sol.y[i] 		# Noisy measurement
	xf, yf = Xmean[i, 1:2]  # Mean of filtered estimate
	plot([0, x], [0, y], m=:circle, markersize=[3, 15], ratio=:equal, label="True", l=7)
	plot!([0, xf], [0, yf], m=:circle, markersize=[3, 15], ratio=:equal, label="Estimate", c=:purple, l=(2))
	scatter!([xm], [ym], m=(:circle, :red, 2), label="Noisy measurements", 
	legend=:topright, xlims=(-1.1, 1.1), ylims=(-1.1, 0.2), grid=false, framestyle=:zerolines)
end; giffig = gif(anim, "anim_fps30.gif", fps = 30)

# ╔═╡ 24c5560f-23e2-46c6-a71e-37299eb735c9
giffig

# ╔═╡ 6df9215b-bdbd-4e5f-b000-a3137246327f
md"""
In the animation, it is hard to tell the difference between the true pendulum state and the estimate of the filter.
"""

# ╔═╡ faa8f242-e25c-460b-8670-ac38d8c27909
md"""
## Conclusion
We have seen how it is possible to estimate the state not only for linear systems with Gaussian noise, but also for the much richer set of systems described by differential-algebraic equations. When we are dealing with a DAE system, we must be extra careful when thinking about what disturbances are affecting the dynamics, and ensure that we employ a technique that will not lead to violations of the algebraic equations. In this post, we made use of a particle filter for the estimation, a very generally applicable estimator that can handle challenging scenarios such as nonlinearites and bimodal state distributions. We modeled the disturbances as entering through the control inputs, a common approach in practice. 
"""

# ╔═╡ 246c8328-e95c-4350-98a7-6a9ab69a26a8
md"""
## Additional detail for the curious
Internally, the filter represents the state distribution as a collection of particles. The filtering solution contains all of those particles in a matrix of size ``N \times T``.
"""

# ╔═╡ 84aff7c2-fd2d-481e-abcf-12f7f810d0a1
typeof(sol.x), size(sol.x)

# ╔═╡ e134d6f0-f08b-4887-bdcb-b1cb646f6e7c
md"""
We can verify that each particle satisfies the algebraic equation:
"""

# ╔═╡ 38efb011-8cdd-40f4-a2a9-27a1c181be94
violation = maximum(abs.(getindex.(pend.(sol.x, u', p, 0), 5)))

# ╔═╡ ab4ae38b-4d7b-424d-9e6c-f17ec81b9ee3
md"""
The largest violation of the algebraic equiation over all the particles was $(round(violation, sigdigits=3)), which is hopefully less than the solver tolerance $(discrete_dynamics.abstol)
"""

# ╔═╡ Cell order:
# ╟─624402d7-7a79-44bb-a813-32edef49db6e
# ╟─d1a4d317-8b27-4f7e-bb6c-02742cf5ab58
# ╟─24c5560f-23e2-46c6-a71e-37299eb735c9
# ╟─1573960e-89b2-4a0a-b5a3-033dc7c414d9
# ╟─9510e624-99d0-45bb-acfd-4da7810b7f9c
# ╟─0a1ac989-e739-414f-af3a-7af31bf771a6
# ╟─fe5001a3-5037-4100-8f4d-f947d49c727c
# ╟─a12d6f59-8b4a-423d-9979-37b089a23de8
# ╟─5df4068d-a4f2-4f74-9ec4-0ad921cf4242
# ╠═42056c78-903c-4dc5-a0b0-6c67ea0ce6f5
# ╠═c08f7fbc-4638-11ee-3944-434b1f71c165
# ╠═d56d8c16-ed6c-4736-85bf-0c60a8ef0826
# ╟─c1ec5b9c-2bf7-4bdf-b5ad-2990f5826069
# ╠═5e003076-c235-4355-899f-037250afa915
# ╟─ebef9c1c-c9cc-4886-9e93-71409da485e3
# ╠═924095f4-51e6-4b0a-b045-5282f2c2b433
# ╟─ff56ae63-ab60-4ce1-b4b3-23bdff078f68
# ╠═bdde4cfd-f546-4c89-9428-6478c296e238
# ╟─cffd10c8-c111-4288-aad5-4ec0e5e3d60e
# ╠═2bcf761a-a598-4dc7-af3d-e5eb32e97846
# ╟─bd6c047d-9346-46f7-b59c-0e5d8a8ba877
# ╠═6995f5c3-e878-4f54-b18f-b6ebc8de14bd
# ╟─30aee77f-0517-4019-98e4-ff3e47f5a8eb
# ╠═8ad859e3-950d-4313-9820-ed437e3e2190
# ╟─29d34dcd-b609-4c08-9182-ccfeaba6f246
# ╠═18457a70-0994-4d75-ac89-31e3d31d9fd1
# ╟─78194091-3a3d-469f-a483-412c01d393ab
# ╠═fa7a6f4a-84da-4110-ace6-ce72e3dab051
# ╟─9a17ba77-ed08-409d-86bf-9a799e8c01cb
# ╠═35d86ddb-ee2d-47f7-9f97-474635458ac2
# ╟─53f68faf-695e-4e65-9a8b-6e4dbb95eb34
# ╠═467caa21-6188-4b2a-914d-1b329ee99413
# ╟─4c8afc2f-c2a7-4a04-847d-f5f38011bebc
# ╟─254d3981-a8f5-41ac-9baf-81e7d2bcd4ed
# ╠═708c0bef-c76b-495f-898e-927354c32cd3
# ╟─6df9215b-bdbd-4e5f-b000-a3137246327f
# ╟─faa8f242-e25c-460b-8670-ac38d8c27909
# ╟─246c8328-e95c-4350-98a7-6a9ab69a26a8
# ╠═84aff7c2-fd2d-481e-abcf-12f7f810d0a1
# ╟─e134d6f0-f08b-4887-bdcb-b1cb646f6e7c
# ╠═38efb011-8cdd-40f4-a2a9-27a1c181be94
# ╟─ab4ae38b-4d7b-424d-9e6c-f17ec81b9ee3
