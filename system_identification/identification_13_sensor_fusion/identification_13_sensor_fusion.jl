### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ 6f594c8c-b616-11ef-1a5d-715a0aa6d6ce
using Pkg; Pkg.activate(@__DIR__()); using LowLevelParticleFilters, ModelingToolkit, ModelingToolkitStandardLibrary.Blocks, ModelingToolkitStandardLibrary.Mechanical.Rotational, OrdinaryDiffEqTsit5, Plots, LinearAlgebra, Random; using ModelingToolkit: renamespace

# ╔═╡ b13223a3-793d-423c-a31f-b01e00e07c63
using PlutoUI

# ╔═╡ 05ec1077-bac7-4a2e-996a-79d24ae670b6
using ModelingToolkit: generate_control_function, build_explicit_observed_function

# ╔═╡ c45f38cb-3e3e-4adf-859a-a5f1e3b5fc84
using LowLevelParticleFilters: AbstractKalmanFilter, parameters, particletype, covtype, state, covariance, KalmanFilteringSolution

# ╔═╡ f09556d1-a597-4e28-8b5b-4547d4413b54
using ControlSystemsMTK, ControlSystemsBase

# ╔═╡ ee8c2372-b65b-41c6-b168-2a027d574177
TableOfContents(depth=4)

# ╔═╡ 2298ac16-5124-4018-8449-9e11752f57cf
# ╠═╡ show_logs = false
theme(:juno); plotly()

# ╔═╡ cb4e7a18-521e-4882-8411-2733a1c271b1
md"""
# Sensor fusion

In this tutorial
- Model a system with ModelingToolkit
- Perform state estimation with multiple sensors running at different rates: __sensor fusion__.
    - Noisy accelerometer, gyroscope with drift
- Add a disturbance model to eliminate gyroscope drift
"""

# ╔═╡ 905d4165-beb3-4d3f-a72a-1979e43d6653
md"""
## Modeling a pendulum

We will use [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl/)
"""

# ╔═╡ 3060a879-2f05-4b66-bf66-9d21ef43972a
import ModelingToolkit: t_nounits as t, D_nounits as D

# ╔═╡ 08d6b16e-ab19-4014-81b8-482f2776f8f8
@mtkmodel Pendulum begin
	@components begin
		input = RealInput()
		accx = RealOutput()
		accy = RealOutput()
		vel = RealOutput()
	end
	@parameters begin
		g = 9.81
		L = 1
		m = 1
		b = 0.3
	end
	@variables begin
		θ(t) = 0
		ω(t) = 0
		τ(t)
	end
	@equations begin
		D(θ) ~ ω
    	D(ω) ~ -g/L * sin(θ) - b/m * ω + τ/(m*L^2)
		input.u ~ τ
		accx.u ~ D(ω) * L * cos(θ) / g + sin(θ)
        accy.u ~ -D(ω) * L * sin(θ) / g + cos(θ)
        vel.u ~ ω
	end
end

# ╔═╡ b73ee871-67be-4814-b455-d621f9252777
@named dynamics_model = Pendulum(); cdyn = complete(dynamics_model)

# ╔═╡ ae3bbafd-1bd1-4fe6-adb8-977006362a8c
@mtkmodel PendWithInput begin
	@components begin
		sine = Blocks.Sine(amplitude = 1, frequency = 2)
		pendulum = Pendulum()
	end
	@equations begin
		connect(sine.output, :u, pendulum.input)
	end
end

# ╔═╡ ad997d7f-9601-48a0-9e8a-5646a492e883
@named model = PendWithInput();

# ╔═╡ c95b0414-b103-4ccd-812a-bcf6e2646460
cmodel = complete(model);

# ╔═╡ d481df4f-ddf9-4e71-b4b0-b9fdef1fdc8f
ssys = structural_simplify(model)

# ╔═╡ 0f1b9932-08cc-49dd-a218-eb2e312c7171
prob = ODEProblem(ssys, [cmodel.sine.amplitude=>0, cmodel.pendulum.θ=>pi-1e-3], (0, 10));

# ╔═╡ 1ef4f684-7088-49ea-ad62-851a398af20c
sol = solve(prob, Tsit5());

# ╔═╡ 5a1c5309-8246-470b-b7d4-917d5a8bbb0c
plot(sol, title="Open-loop simulation")

# ╔═╡ ae42c333-88e3-496d-9b69-8fa4efc7a76a
measured_outputs = [
	cdyn.accx.u
	cdyn.accy.u
	cdyn.vel.u
];

# ╔═╡ 52182822-e947-4629-b1b7-5f86f21eab3b
state_outputs = [
	cdyn.θ
	cdyn.ω
];

# ╔═╡ 44949b1b-6cf3-4168-b852-cf5fdaedb7ac
outputs = [state_outputs; measured_outputs]

# ╔═╡ 9e72115d-36aa-4ad8-882d-a49fdccb0ab6
state_outputsns = renamespace.((cmodel.pendulum,), state_outputs);

# ╔═╡ d2aeac1c-e11b-4c7f-849b-bf3e66af5738
measured_outputsns = renamespace.((cmodel.pendulum,), measured_outputs);

# ╔═╡ b87ec4a5-af3d-4e7d-aa05-47917bd60b7f
outputsns = [state_outputsns; measured_outputsns]

# ╔═╡ 14242408-6d9d-407a-a640-e592fbabae5c
md"""
## Sensor fusion

What to do if sensors operate on different rates?
- Use the lowest sample rate only (simple 👍, discards data 👎)
- Interpolate slow sensor (simple 👍, cannot be done at higher rate than slow sensor 👎)
- Perform individual measurement updates at different rates (this video, simple 👍👍 👍)

Situations that can occur
- Slow rate is an integer multiple of fast rate (this video)
- No simple relationship
    - Pick a small "base rate" such that each sample rate is an approximate multiple of base rate
- Stochastic sample rate
    - Event based
    - Use generic integration between events
"""

# ╔═╡ ca9c4d3d-189e-428a-92ba-a66a8d61b3ee
md"""
### Simulate measurement data
- Input is updated at rate ``T_s = 0.01s``
- Gyroscope is sampling at rate ``T_s = 0.01s``
- Accelerometer is sampling at rate ``T_s = 0.05s``
"""

# ╔═╡ 6aee3190-49e6-41a5-8a2b-a778a57b41ed
Ts = 0.01; # Controller sample rate

# ╔═╡ 9cc73709-e2d8-4e4f-8096-39b25650433b
md"""
### Generating a dynamics function
MTK can generate a function ``\dot x = f(x,u,p,t)`` that we can discretize using RK4 into ``x^+ = f(x,u,p,t)``
"""

# ╔═╡ e7221fbb-5557-4885-8a5c-06c0f522b405
(dynamics, ip_dynamics), dvs, ps, io_sys = generate_control_function(
	cdyn,
	[cdyn.τ];
	split = false,
);

# ╔═╡ dc80e26c-2d98-4534-abe7-93824fade287
discrete_dynamics = LowLevelParticleFilters.rk4(dynamics, Ts);

# ╔═╡ 1437fb69-04fe-4fc7-93a5-44d35631079b
md"""
Linearize the system to obtain a state-permutation matrix
"""

# ╔═╡ 37ff0ed9-033d-4be7-8f1e-19aa02e5f56f
md"""
To forumulate the state covariance matrix, we perform a similarity transform with ``C`` which transforms the MTK-chosen state ``x_{MTK}`` into our desired state order, ``R_{MTK} = C^T R_1^{desired}C``. This allows us to express the covariance in the desired state order, and have it transformed to the state order chosen by MTK.

Since ``C`` is a permutaion matrix, ``C^{-1} = C^T``.
"""

# ╔═╡ a3e1b787-26b1-4091-9781-271e60152d60
op2 = Dict([cdyn.θ=>pi, cdyn.τ => 0]);

# ╔═╡ c30b6992-726d-4d77-bb6d-ff45201f6a5f
md"Obtain numeric initial condition and parameter vector from MTK:"

# ╔═╡ 3ee4bbf5-f3b2-4417-b6ea-4eb0d65b47fc
u0, p = ModelingToolkit.get_u0_p(io_sys, op2, op2);

# ╔═╡ fca67a24-ba69-4288-8cd1-fff64ff5e3f4
md"""
### Sensors with different sample rates

- Gyro is fast 👍 and has little noise 👍, but slowly drifts over time 👎
- Accelerometer is 5x slower 👎 and noisy 👎, but it has no bias 👍
- Construct one _measurement model_ for each sensor
- Fuse the sensors (and the dynamics model), get the best of both worlds 👍👍👍
"""

# ╔═╡ aa3b9c7c-ef8b-4b56-9554-104a5c6e3e92
findapprox(t, tvec) = findfirst(ti->abs(t-ti) < 1e-8, tvec)

# ╔═╡ 9e8a60d4-df29-4ec5-afef-c8f6c70fb1b9
md"""
#### Result with different sample rates
"""

# ╔═╡ fafa006d-7c6f-43ed-83f4-b79e27ac59f7
md"""
- There is a bias in the estimate of the velocity 👎
- The filter is overly confident about its estimate of the velocity 👎👎
- What can we do about this?
"""

# ╔═╡ c934fcd1-c525-407c-a83b-624bd5020fa8
md"""
## Adding a noise model
By adding a model of the measurement disturbance ``x_w`` (gyro drift), we can let the state estimator estimate the bias!

```math
\begin{aligned}
\dot x_w &= 0 + w \quad &\text{Simple integrator}\\
y_{gyro} &= ω + x_w  + e \quad &\text{Actual velocity + bias}
\end{aligned}
```
"""

# ╔═╡ e61710f6-c037-455a-98b6-42a6f823f487
@mtkmodel PendulumWithNoiseModel begin
	@components begin
		input = RealInput()
		accx = RealOutput()
		accy = RealOutput()
		vel = RealOutput()
	end
	@parameters begin
		g = 9.81
		L = 1
		m = 1
		b = 0.1
	end
	@variables begin
		θ(t) = 0
		ω(t) = 0
		τ(t)
		w(t) = 0 	# Noise input
		x_w(t) = 0 	# Noise model state
	end
	@equations begin
		D(θ) ~ ω
    	D(ω) ~ -g/L * sin(θ) - b/m * ω + τ/(m*L^2)
		input.u ~ τ
		accx.u ~ D(ω) * L * cos(θ) / g + sin(θ)
        accy.u ~ -D(ω) * L * sin(θ) / g + cos(θ)
		D(x_w) ~ w
        vel.u ~ ω + x_w
	end
end

# ╔═╡ 1e20123d-45e2-488f-b9bd-eae3f05d6240
@named augmented_dynamics_model = PendulumWithNoiseModel(); caugdyn = complete(augmented_dynamics_model)

# ╔═╡ b587d11f-e71c-4ebc-b712-f0ec62ac7f24
P3 = named_ss(augmented_dynamics_model, [caugdyn.τ, caugdyn.w], measured_outputs; op=Dict([caugdyn.θ=>pi, caugdyn.τ=>0]))

# ╔═╡ d47d1e4b-3ad8-482a-b083-e85e15592259
md"""
#### Result with added noise model
"""

# ╔═╡ 37b2fc46-117f-4b39-9266-8dd66e41d772
md"""
- The bias is mostly eliminated! 👍
- The filter is less confident about the state estimates 👍
"""

# ╔═╡ e0264246-f5a3-4b95-aa1f-a39a2b294414
md"""
Compare the errors in the estimated state trajectories:
"""

# ╔═╡ bf0902c8-c637-442d-9696-892d96c6b0af
md"""
Do we actually need the slow and noisy accelerometer?

Could we get by with just the gyro when estimating the bias?
"""

# ╔═╡ 4f8cfdac-de1a-49a6-9c68-ec63496aee56
begin
	Ts_acc = 0.05
	Ts_vel = 0.01
end;

# ╔═╡ 932e5650-9ac0-473d-af2c-cabdd3e10296
md"""
## Closing remarks
- The state estimator operates with separate steps for dynamics and measurement updates
- One can run zero, one or several measurement updates for each prediction step
- One may use both linear and nonlinear measurement models
- A [complementary filter](https://vanhunteradams.com/Pico/ReactionWheel/Complementary_Filters.html) is often used to fuse accelerometer and gyro, but how would it handle the different rates?
"""

# ╔═╡ 345cac0b-116b-443b-86b5-57089e15ef6d
md"""
## Control design and simulation

The code below linearizes the system in the upward equilibrium and designs an LQR controller.
"""

# ╔═╡ 70fc2059-87d5-475d-aa0f-87b53764236d
op = Dict([cmodel.pendulum.θ=>pi]);

# ╔═╡ 14360fc6-6b0b-4a3d-afb0-2605d55264e8
sys = named_ss(model, :u, outputsns; op) # This linearizes the model

# ╔═╡ a658e5a5-8876-4907-95de-fd62c5057343
C = sys[Symbol.(state_outputsns), :].C # Useful for state permutation

# ╔═╡ d7a38468-32ed-421c-8904-836bc78be662
R1 = C'*Diagonal([0.0001,0.01])*C

# ╔═╡ 1973db54-f220-4a28-9550-41e116e83c11
d0 = LowLevelParticleFilters.SimpleMvNormal(u0, 10R1);

# ╔═╡ 7c02d4e9-8eef-4845-861e-0a542d3cee3e
P = sys[Symbol.(measured_outputsns), :]

# ╔═╡ 7965a7c6-3cc6-4a98-9690-552e74f4f92a
begin
	measurement_acc = build_explicit_observed_function(io_sys, [cdyn.accx.u, cdyn.accy.u], inputs=[cdyn.τ])
	
	measurement_vel = build_explicit_observed_function(io_sys, [cdyn.vel.u], inputs=[cdyn.τ])

	R2_acc = 0.1^2*I(2)
	R2_vel = 0.01I(1)

	# Construct two different measurement models
	measurement_model_acc = UKFMeasurementModel{Float64, false, false}(measurement_acc, R2_acc; P.nx, ny=2)
	
	measurement_model_vel = UKFMeasurementModel{Float64, false, false}(measurement_vel, R2_vel; P.nx, ny=1)

	composite_measurement_model = CompositeMeasurementModel(measurement_model_acc, measurement_model_vel)

	kf2 = UnscentedKalmanFilter{false,false,false,false}(discrete_dynamics, composite_measurement_model, R1, d0; P.nu, p, Ts)
end;

# ╔═╡ 971c860f-fef5-4b9f-b9a9-2612b4d44bb3
begin
	R13 = cat(R1, [0.001;;], dims=(1,2)) # Covariance matrix for augmented state
	
	(dynamics3, ip_dynamics3), dvs3, ps3, io_sys3 = ModelingToolkit.generate_control_function(caugdyn, [caugdyn.τ], [caugdyn.w]; split=false) # Tell MTK that caugdyn.w is a disturbance input
	
	measurement_acc3 = ModelingToolkit.build_explicit_observed_function(io_sys3, [caugdyn.accx.u, caugdyn.accy.u], inputs=[caugdyn.τ])
	
	measurement_vel3 = ModelingToolkit.build_explicit_observed_function(io_sys3, [caugdyn.vel.u], inputs=[caugdyn.τ])
	
	u03, p3 = ModelingToolkit.get_u0_p(io_sys3, op2, op2)
	discrete_dynamics3 = LowLevelParticleFilters.rk4(dynamics3, Ts)
	
	d03 = LowLevelParticleFilters.SimpleMvNormal(u03, 10R13)

	measurement_model_acc3 = UKFMeasurementModel{Float64, false, false}(measurement_acc3, R2_acc; P3.nx, ny=2)
	
	measurement_model_vel3 = UKFMeasurementModel{Float64, false, false}(measurement_vel3, R2_vel; P3.nx, ny=1)

	composite_measurement_model3 = LowLevelParticleFilters.CompositeMeasurementModel(measurement_model_acc3, measurement_model_vel3)

	kf3 = UnscentedKalmanFilter{false,false,false,false}(discrete_dynamics3, composite_measurement_model3, R13, d03; P3.nu, p=p3, Ts)
end

# ╔═╡ c666df41-83a9-4902-8f8e-ca25066a453d
L = lqr(sys, C'*Diagonal([10,0.1])*C, 0.001I(sys.nu)) / C # C permutes to output order

# ╔═╡ 51a6a532-bc49-4fc6-8251-790884e6e8a5
md"Create a model of the closed-loop system and simulate some data to be used for the filtering in this video"

# ╔═╡ 5605e1d0-d4ef-4269-8420-44cd220c4ed1
@mtkmodel ClosedLoop begin
	@components begin
		gain = Blocks.MatrixGain(K = L)
		ref = Blocks.Square(amplitude=0.5, frequency=1/2, offset=pi)
		pendulum = Pendulum()
	end
	@equations begin
		connect(gain.output, pendulum.input)
		gain.input.u[1] ~ ref.output.u - pendulum.θ
		gain.input.u[2] ~ -pendulum.ω
	end
end

# ╔═╡ a863dab2-6daf-42e7-b8bd-973c2352af21
@named cl = ClosedLoop();

# ╔═╡ 8dccdb59-9cd1-4d55-9c87-d532b466f48c
ccl = complete(cl);

# ╔═╡ de75fe0d-e00b-4630-86a8-79515f2510b4
ssys_cl = structural_simplify(cl)

# ╔═╡ 8b9349e4-419c-4f66-88cb-c0606843c64d
prob_cl = ODEProblem(ssys_cl, [ccl.pendulum.θ=>pi-1e-3], (0, 6));

# ╔═╡ 8a0fc8e4-a73d-4900-ae80-915a5ac41125
sol_cl = solve(prob_cl, Tsit5());

# ╔═╡ a18bd605-2b83-4277-aa6d-5c5068ca2323
plot(sol_cl, idxs=[state_outputsns; ccl.ref.output.u; 0.01ccl.pendulum.τ], title="Simulated data with closed loop")

# ╔═╡ 7d61b536-e031-4ef8-bd05-3affe3271e44
t_vec = 0:Ts:sol_cl.t[end];

# ╔═╡ da7eef1e-feac-4bdf-9331-0f8740b5e694
u = collect.(eachcol(Matrix(sol_cl(t_vec, idxs=[cmodel.pendulum.τ]))));

# ╔═╡ edb870ea-27f7-42b2-8949-4d8a234c79b5
begin
	# Obtain simulated measurement data for each sensor
	Random.seed!(0)
	# Accelerometer
	t_acc = 0:Ts_acc:sol_cl.t[end]
	y_acc = [y + sqrt(R2_acc)*randn(2) for y in sol_cl(t_acc, idxs=[ccl.pendulum.accx.u, ccl.pendulum.accy.u])]

	# Gyroscope
	t_vel = 0:Ts_vel:sol_cl.t[end]
	y_vel = [y + sqrt(R2_vel)*randn(1) for y in sol_cl(t_vel, idxs=[ccl.pendulum.vel.u + 0.2t])] # Add measurement noise and the slow drift (linear drifte over time with t)

	# All outputs at high sample rate for comparison
	Y = Matrix(sol_cl(t_vec, idxs=measured_outputsns .+ [0,0,0.2t]))
	y = collect.(eachcol(Y))
end;

# ╔═╡ be5ac935-8ed9-4f7a-afce-b59f55771a91
plot(t_acc, reduce(hcat, y_acc)', label="acc"); plot!(t_vel, reduce(hcat, y_vel)', label="gyro", title="Measured data")

# ╔═╡ 962a48c6-8498-49ce-aa42-d543822005f9
function sensor_fusion(kf::AbstractKalmanFilter, measurement_model_acc, measurement_model_vel, u::AbstractVector, y_acc, y_vel, p=parameters(kf))
    reset!(kf)
    T    = length(y_vel)
    x    = Array{particletype(kf)}(undef,T)
    xt   = Array{particletype(kf)}(undef,T)
    R    = Array{covtype(kf)}(undef,T)
    Rt   = Array{covtype(kf)}(undef,T)
    e    = similar(y_vel)
    ll   = zero(eltype(particletype(kf)))
    for t = 1:T
        ti = (t-1)*kf.Ts
        x[t]  = state(kf)      |> copy
        R[t]  = covariance(kf) |> copy
		lli = 0.0
		ei = fill(0.0, 3)
		# Perform measurement updates only when data is available
		if (i_sens = findapprox(ti, t_acc)) !== nothing
        	lli0, ei0 = correct!(kf, measurement_model_acc, u[t], y_acc[i_sens], p, ti)
			lli += lli0
			ei[1:2] .= ei0
		end
		if (i_sens = findapprox(ti, t_vel)) !== nothing
        	lli0, ei0 = correct!(kf, measurement_model_vel, u[t], y_vel[i_sens], p, ti)
			lli += lli0
			ei[3] = ei0[]
		end
        ll += lli
        e[t] = ei
        xt[t] = state(kf)      |> copy
        Rt[t] = covariance(kf) |> copy
        predict!(kf, u[t], p, ti)
    end
    KalmanFilteringSolution(kf,u,y,x,xt,R,Rt,ll,e)
end

# ╔═╡ 1eb13d0e-5818-49c5-9594-4e64424b2fee
sol2 = sensor_fusion(kf2, measurement_model_acc, measurement_model_vel, u, y_acc, y_vel)

# ╔═╡ b6f91acb-dadb-4fc2-ac42-0f63a7dac168
sol3 = sensor_fusion(kf3, measurement_model_acc3, measurement_model_vel3, u, y_acc, y_vel)

# ╔═╡ bf06c3cd-8328-4ed7-abf9-991becd97eef
begin
	true_state = Matrix(sol_cl(t_vec))'
	plot(sol2, plotRt=true, plotu=false, ploty=false, plotyh=false, layout=(2,1), size=(650,500*2/3))
	plot!(t_vec, true_state, label="True x")
	vline!(t_acc', l=(:white, :dash), primary=false, alpha=0.2)
	scatter!(t_vel, reduce(vcat, y_vel), sp=2, alpha=0.5, label="y gyro")
end

# ╔═╡ 0cf00595-6b64-43dd-8fa9-56c9a1f43080
begin
	plot(sol3, plotRt=true, plotu=false, ploty=false, plotyh=false, layout=(3,1), size=(650,500))
	plot!(t_vec, true_state, label="True x")
	vline!(ones(1,3) .* t_acc, l=(:white, :dash), primary=false, alpha=0.2)
	scatter!(t_vel, reduce(vcat, y_vel), sp=2, alpha=0.5, label="y gyro")
	plot!(t_vel, 0.2t_vel, sp=3, lab="Gyro drift", l=(:white, :dash))
end

# ╔═╡ 001ee9da-029e-4b10-9578-cdff50b8b4e4
begin
	errors2 = reduce(hcat, sol2.xt)' - true_state
	errors3 = reduce(hcat, sol3.xt)'[:, 1:2] - true_state
	
	@info "Without noise model: $(sum(abs2, errors2))\nwith noise model: $(sum(abs2, errors3))"
end

# ╔═╡ 01dc8894-eed6-4dbd-9528-3832b24350bd
gr(); anim = @animate for i in 1:1:length(t_vec)
	ti = t_vec[i]
	θ, ω = true_state[i, 1:2]    	# True coordinates
	y,x = sincos(θ-pi/2)
	θm = sol3.xt[i][1]
	ym, xm = sincos(θm-pi/2) 		# Noisy measurement

	plot([0, x], [0, y], m=:circle, markersize=[3, 15], ratio=:equal, label="True", l=7, c=3)
	plot!([0, xm], [0, ym], m=:circle, markersize=[2, 12], ratio=:equal, label="Estimate", c=2, l=(2),legend=:topright, xlims=(-1.1, 1.1), ylims=(0, 1.1), grid=false, framestyle=:zerolines, xaxis=false, yaxis=false)
end; plotly(); giffig = gif(anim, "anim_fps30.gif", fps = 30); giffig

# ╔═╡ 5be074be-4168-494a-bc7d-834b238df241
giffig

# ╔═╡ Cell order:
# ╠═6f594c8c-b616-11ef-1a5d-715a0aa6d6ce
# ╠═b13223a3-793d-423c-a31f-b01e00e07c63
# ╠═ee8c2372-b65b-41c6-b168-2a027d574177
# ╠═2298ac16-5124-4018-8449-9e11752f57cf
# ╟─cb4e7a18-521e-4882-8411-2733a1c271b1
# ╟─5be074be-4168-494a-bc7d-834b238df241
# ╟─905d4165-beb3-4d3f-a72a-1979e43d6653
# ╠═3060a879-2f05-4b66-bf66-9d21ef43972a
# ╠═08d6b16e-ab19-4014-81b8-482f2776f8f8
# ╠═b73ee871-67be-4814-b455-d621f9252777
# ╠═ae3bbafd-1bd1-4fe6-adb8-977006362a8c
# ╠═ad997d7f-9601-48a0-9e8a-5646a492e883
# ╠═c95b0414-b103-4ccd-812a-bcf6e2646460
# ╠═d481df4f-ddf9-4e71-b4b0-b9fdef1fdc8f
# ╠═0f1b9932-08cc-49dd-a218-eb2e312c7171
# ╠═1ef4f684-7088-49ea-ad62-851a398af20c
# ╟─5a1c5309-8246-470b-b7d4-917d5a8bbb0c
# ╠═ae42c333-88e3-496d-9b69-8fa4efc7a76a
# ╠═52182822-e947-4629-b1b7-5f86f21eab3b
# ╠═44949b1b-6cf3-4168-b852-cf5fdaedb7ac
# ╠═9e72115d-36aa-4ad8-882d-a49fdccb0ab6
# ╠═d2aeac1c-e11b-4c7f-849b-bf3e66af5738
# ╠═b87ec4a5-af3d-4e7d-aa05-47917bd60b7f
# ╠═a18bd605-2b83-4277-aa6d-5c5068ca2323
# ╟─14242408-6d9d-407a-a640-e592fbabae5c
# ╟─ca9c4d3d-189e-428a-92ba-a66a8d61b3ee
# ╠═6aee3190-49e6-41a5-8a2b-a778a57b41ed
# ╠═7d61b536-e031-4ef8-bd05-3affe3271e44
# ╠═da7eef1e-feac-4bdf-9331-0f8740b5e694
# ╠═edb870ea-27f7-42b2-8949-4d8a234c79b5
# ╟─be5ac935-8ed9-4f7a-afce-b59f55771a91
# ╟─9cc73709-e2d8-4e4f-8096-39b25650433b
# ╠═05ec1077-bac7-4a2e-996a-79d24ae670b6
# ╠═e7221fbb-5557-4885-8a5c-06c0f522b405
# ╠═dc80e26c-2d98-4534-abe7-93824fade287
# ╟─1437fb69-04fe-4fc7-93a5-44d35631079b
# ╠═a658e5a5-8876-4907-95de-fd62c5057343
# ╟─37ff0ed9-033d-4be7-8f1e-19aa02e5f56f
# ╠═a3e1b787-26b1-4091-9781-271e60152d60
# ╟─c30b6992-726d-4d77-bb6d-ff45201f6a5f
# ╠═3ee4bbf5-f3b2-4417-b6ea-4eb0d65b47fc
# ╠═1973db54-f220-4a28-9550-41e116e83c11
# ╟─fca67a24-ba69-4288-8cd1-fff64ff5e3f4
# ╠═d7a38468-32ed-421c-8904-836bc78be662
# ╠═7965a7c6-3cc6-4a98-9690-552e74f4f92a
# ╠═c45f38cb-3e3e-4adf-859a-a5f1e3b5fc84
# ╠═aa3b9c7c-ef8b-4b56-9554-104a5c6e3e92
# ╠═962a48c6-8498-49ce-aa42-d543822005f9
# ╠═1eb13d0e-5818-49c5-9594-4e64424b2fee
# ╟─9e8a60d4-df29-4ec5-afef-c8f6c70fb1b9
# ╟─bf06c3cd-8328-4ed7-abf9-991becd97eef
# ╟─fafa006d-7c6f-43ed-83f4-b79e27ac59f7
# ╟─c934fcd1-c525-407c-a83b-624bd5020fa8
# ╠═e61710f6-c037-455a-98b6-42a6f823f487
# ╠═1e20123d-45e2-488f-b9bd-eae3f05d6240
# ╠═b587d11f-e71c-4ebc-b712-f0ec62ac7f24
# ╠═971c860f-fef5-4b9f-b9a9-2612b4d44bb3
# ╟─d47d1e4b-3ad8-482a-b083-e85e15592259
# ╠═b6f91acb-dadb-4fc2-ac42-0f63a7dac168
# ╠═0cf00595-6b64-43dd-8fa9-56c9a1f43080
# ╟─37b2fc46-117f-4b39-9266-8dd66e41d772
# ╟─e0264246-f5a3-4b95-aa1f-a39a2b294414
# ╠═001ee9da-029e-4b10-9578-cdff50b8b4e4
# ╟─bf0902c8-c637-442d-9696-892d96c6b0af
# ╠═4f8cfdac-de1a-49a6-9c68-ec63496aee56
# ╟─932e5650-9ac0-473d-af2c-cabdd3e10296
# ╟─345cac0b-116b-443b-86b5-57089e15ef6d
# ╠═f09556d1-a597-4e28-8b5b-4547d4413b54
# ╠═70fc2059-87d5-475d-aa0f-87b53764236d
# ╠═14360fc6-6b0b-4a3d-afb0-2605d55264e8
# ╠═7c02d4e9-8eef-4845-861e-0a542d3cee3e
# ╠═c666df41-83a9-4902-8f8e-ca25066a453d
# ╟─51a6a532-bc49-4fc6-8251-790884e6e8a5
# ╠═5605e1d0-d4ef-4269-8420-44cd220c4ed1
# ╠═a863dab2-6daf-42e7-b8bd-973c2352af21
# ╠═8dccdb59-9cd1-4d55-9c87-d532b466f48c
# ╠═de75fe0d-e00b-4630-86a8-79515f2510b4
# ╠═8b9349e4-419c-4f66-88cb-c0606843c64d
# ╠═8a0fc8e4-a73d-4900-ae80-915a5ac41125
# ╠═01dc8894-eed6-4dbd-9528-3832b24350bd
