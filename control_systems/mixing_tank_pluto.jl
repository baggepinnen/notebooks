### A Pluto.jl notebook ###
# v0.19.30

using Markdown
using InteractiveUtils

# ╔═╡ db3be3eb-d8aa-4750-bfa5-9e0aeb806c64
import Pkg;Pkg.activate(;temp=true)

# ╔═╡ d6845b2a-ed77-4058-b503-de043ce8c107
# ╠═╡ show_logs = false
using JuliaSimControl

# ╔═╡ 01dd9e50-8476-4bdd-8eed-ca390844a22a
using ModelingToolkit, ModelingToolkitStandardLibrary, OrdinaryDiffEq, Plots; theme(:dracula); gr(dpi=300)

# ╔═╡ 607955ec-c529-4d0d-8e01-385585c9d7b5
using ModelingToolkitStandardLibrary.Blocks

# ╔═╡ 101b6e8a-6649-4b9a-a610-758fecaccca4
using Random

# ╔═╡ 9754bc74-531b-496d-98f9-608224ce5d5d
using PlutoUI

# ╔═╡ f14003e2-d9a7-49ba-af25-1085fef8a2f9
md"""
# Acausal Modeling for Nonlinear Control Design and Closed-Loop Analysis:
## Application to Chemical Reaction Tanks

"""

# ╔═╡ f95ea338-3093-4cbf-a846-4731ca7e39c4
rc = 0.25 # Reference concentration 

# ╔═╡ 068ec563-0802-421a-ad49-855157f1e2a0
md"""
# Tank modeling
"""

# ╔═╡ 3cb3bca3-c219-414d-b075-afa1ea8741ca
md"""
# Open loop simulation with constant input
"""

# ╔═╡ efae2c84-46d0-4c6c-88d4-1ca5bc19875f
tspan = (0.0, 1000.0)

# ╔═╡ 56ab17b7-7c98-4f1c-bcd6-60b4261db8ae
md"""
Simulate open loop and look at frequency-domain properties
"""

# ╔═╡ 89fed73e-3eea-4341-9738-d6c2ccd07b13
md"""
The open-loop system is poorly damped in this operating point.
"""

# ╔═╡ 3126f474-4e01-4081-a084-44789101123e
md"""
# Closed loop with P controller

With temperature feedback
"""

# ╔═╡ 824658b8-153c-488d-961d-ba3cba2a22dd
md"""
# Closed loop with PI controller

With temperature feedback.

No steady-state error in temperature, but what about concentration?
"""

# ╔═╡ 8c872dbb-d499-4b03-88b6-5c94a88bd9f1
md"""
# Filter design

Design the ``F`` such that ``F P^{-1}`` is strictly proper
"""

# ╔═╡ 246173b4-1618-4ae3-a30c-2f20e790acbe
begin
	Ftf = tf(1, [(100), 1])^3
	Fss = ss(Ftf)

	"Compute initial state that yields y0 as output"
	function init_filter(y0)
	    (; A,B,C,D) = Fss
	    Fx0 = -A\B*y0
	    @assert C*Fx0 ≈ [y0] "C*Fx0*y0 ≈ y0 failed, got $(C*Fx0*y0) ≈ $(y0)]" 
	    Fx0
	end

	# Create an MTK-compatible constructor 
	RefFilter(; y0, name) = ODESystem(Fss; name, x0=init_filter(y0))
end

# ╔═╡ 6dcfdd57-989b-4394-bef4-6a2c1bb170ea
md"""
# Feedforward using inverse model
"""

# ╔═╡ 4595cfcd-0231-4456-aa98-bcf80a3f2fd4
md"""
Simulate with 50% error in the plant model parameters
"""

# ╔═╡ 68ae9371-f3a2-42a9-9b50-2a2717d67ab4
md"""
# Closed-loop analysis
"""

# ╔═╡ 9ba5eaca-2be0-4db4-93bd-6bafb07aee52
md"""
Investigate controller noise gain
"""

# ╔═╡ 2e64ab45-366e-4e9a-a8f1-2bb682a42868
md"""
---
#### Notebook setup
"""

# ╔═╡ bc75dfd0-dcd8-4c16-87a4-736ea1acb5d7
linearize_named_ss = named_ss;

# ╔═╡ 99ebe7c4-2aa6-4c30-b065-b2d40c545168
connect = ModelingToolkit.connect;

# ╔═╡ 517032af-5b7a-4c1f-b6d9-5d5a0b5c9bad
@parameters t; D = Differential(t);

# ╔═╡ 72d9ec45-78c8-4009-b3ef-77b6d7e5420a
@mtkmodel MixingTank begin
    @parameters begin
        c0  = 0.8,   [description = "Nominal concentration"]
        T0  = 308.5, [description = "Nominal temperature"]
        a1  = 0.2674
        a21 = 1.815
        a22 = 0.4682
        b   = 1.5476
        k0  = 1.05e14
        ϵ   = 34.2894
    end

    @variables begin
        γ(t),         [description = "Reaction speed"]
        xc(t) = c0,   [description = "Concentration"]
        xT(t) = T0,   [description = "Temperature"]
        xT_c(t) = T0, [description = "Cooling temperature"]
    end

    @components begin
        T_c = RealInput()
        c   = RealOutput()
        T   = RealOutput()
    end

    begin
        τ0   = 60
        wk0  = k0/c0
        wϵ   = ϵ*T0
        wa11 = a1/τ0
        wa12 = c0/τ0
        wa13 = c0*a1/τ0
        wa21 = a21/τ0
        wa22 = a22*T0/τ0
        wa23 = T0*(a21 - b)/τ0
        wb   = b/τ0
    end
    @equations begin
        γ ~ xc*wk0*exp( -wϵ/xT)
        D(xc) ~ -wa11*xc - wa12*γ + wa13
        D(xT) ~ -wa21*xT + wa22*γ + wa23 + wb*xT_c

        xc ~ c.u
        xT ~ T.u
        xT_c ~ T_c.u
    end

end;

# ╔═╡ 1868e073-0483-411b-ae2e-c5c863946398
@mtkmodel TankWithInput begin
	@components begin
		input = Constant(k=306.25) # Constant cooling temperature input
		tank = MixingTank()
	end
	@equations begin
		connect(input.output, :u, tank.T_c)
	end
end;

# ╔═╡ 79d08299-8814-4823-bfad-161c63210082
# ╠═╡ show_logs = false
let
	@named model = TankWithInput()
	cmodel = complete(model)
	
	ssys = structural_simplify(model)
	prob = ODEProblem(ssys, [], tspan)
	
	sol = solve(prob, Rodas5P())
	f1 = plot(sol, layout=2, title="Open-loop")
	linmod = linearize_named_ss(model, :u, [cmodel.tank.xc, cmodel.tank.xT])
	w = exp10.(-4:0.01:2)
	plot(f1,
		nyquistplot(linmod, w, legend=:outerright, unit_circle=true, ylims=(-5, 1), xlims=(-3, 1), ratio=1, title="", ylabel=""),
		pzmap(linmod, ratio=1),
		bodeplot(linmod[2,1], w),
		margin=1Plots.mm,
		size=(800,800),
	)
end

# ╔═╡ 53b5adc6-eab1-4e51-8ae5-ca88a5ee0c4a
@mtkmodel PControlledTank begin
	@components begin
		ref = Constant(k=325) # Temperature reference
		controller = Gain(k=20)
		tank = MixingTank()
		feedback = Feedback()
	end
	@equations begin
		connect(ref.output, feedback.input1)
		connect(tank.T, feedback.input2)
		connect(feedback.output, controller.input)
		connect(controller.output, tank.T_c)
	end
end;

# ╔═╡ 3bd15fcb-d990-41f1-8784-199d8edc2d30
let	
	@named model = PControlledTank()
	ssys = structural_simplify(model)
	prob = ODEProblem(ssys, [], tspan)
	sol = solve(prob, Rodas5P())
	plot(sol, layout=2)
	hline!([rc prob[complete(model).ref.k]], label="ref")
end

# ╔═╡ 798627ae-27e1-47bb-ba6e-51e2bd9361cd
@mtkmodel PIControlledTank begin
	@components begin
		ref = Constant(k=325)
		controller = PI(k=20, T=50)
		tank = MixingTank()
		feedback = Feedback()
	end
	@equations begin
		connect(ref.output, feedback.input1)
		connect(tank.T, feedback.input2)
		connect(feedback.output, controller.err_input)
		connect(controller.ctr_output, tank.T_c)
	end
end;

# ╔═╡ d4da467b-a761-4f0f-b902-2cd7b5bc8241
let
	@named model = PIControlledTank()
	ssys = structural_simplify(model)
	prob = ODEProblem(ssys, [], tspan)
	sol = solve(prob, Rodas5P())
	plot(sol, idxs=[model.tank.xc, model.tank.xT, model.controller.ctr_output.u], layout=3, sp=[1 2 3])
	hline!([rc prob[complete(model).ref.k]], label="ref")
end

# ╔═╡ 60f87548-4edb-40e4-9b38-aa61d9e39356
@mtkmodel InverseControlledTank begin
	begin
		c0 = 0.8    #  "Nominal concentration
		T0 = 308.5 	#  "Nominal temperature
		x10 = 0.42 	
		x20 = 0.01 
		u0 = -0.0224 

		c_start = c0*(1-x10) 		# Initial concentration
		T_start = T0*(1+x20) 		# Initial temperature
		c_high_start = c0*(1-0.72) 	# Reference concentration
		T_c_start = T0*(1+u0) 		# Initial cooling temperature
	end
	@components begin
		ref 			= Constant(k=0.25) # Concentration reference
		ff_gain 		= Gain(k=1) # To allow turning ff off
		controller 		= PI(k=10, T=500)
		tank 			= MixingTank(xc=c_start, xT = T_start, c0=c0, T0=T0)
		inverse_tank 	= MixingTank(xc=c_start, xT = T_start, c0=c0, T0=T0)
		feedback 		= Feedback()
		add 			= Add()
		filter 			= RefFilter(y0=c_start) # Initialize filter states to the initial concentration
		noise_filter 	= FirstOrder(k=1, T=1, x=T_start)
		# limiter = Gain(k=1)
		limiter 		= Limiter(y_max=370, y_min=250) # Saturate the control input 
	end
	@equations begin
		connect(ref.output, :r, filter.input)
		connect(filter.output, inverse_tank.c)

		connect(inverse_tank.T_c, ff_gain.input)
		connect(ff_gain.output, :uff, limiter.input)
		connect(limiter.output, add.input1)

		connect(controller.ctr_output, :u, add.input2)

		#connect(add.output, :u_tot, limiter.input)
		#connect(limiter.output, :v, tank.T_c)

		connect(add.output, :u_tot, tank.T_c)


		connect(inverse_tank.T, feedback.input1)

		connect(tank.T, :y, noise_filter.input)

		connect(noise_filter.output, feedback.input2)
		connect(feedback.output, :e, controller.err_input)
	end
end;

# ╔═╡ 6aabc0e9-3a08-45b7-9e7c-937c772b859b
begin
	@named model = InverseControlledTank()
	ssys = structural_simplify(model)
	cm = complete(model)
	
	op = Dict{Num, Real}(
		D(cm.inverse_tank.xT) => 1.0,
	)

	op_50 = Dict{Num, Real}(
		D(cm.inverse_tank.xT) => 1.0,
		cm.tank.a1 => 	1.5 * cm.inverse_tank.a1, # 50% error in plant model parameter
		cm.tank.a21 => 	1.5 * cm.inverse_tank.a21,
		cm.tank.a22 => 	1.5 * cm.inverse_tank.a22,
		cm.tank.b => 	1.5 * cm.inverse_tank.b,
		cm.tank.k0 => 	1.5 * cm.inverse_tank.k0,
	)
end;

# ╔═╡ eb09d92e-df2a-4290-915e-89864787cf19
begin
	prob = ODEProblem(ssys, op_50, tspan)
	
	sol = solve(prob, Rodas5P(), abstol=1e-8, reltol=1e-8)
	
	@assert SciMLBase.successful_retcode(sol)
	
	plot(sol, idxs=[cm.tank.xc, cm.tank.xT, cm.controller.ctr_output.u, cm.add.output.u, cm.controller.err_input.u], layout=4, sp=[1 2 4 3 4])
	plot!(sol, idxs=[cm.inverse_tank.xc, cm.inverse_tank.xT], sp=[1 2], l=:dash, legend=true)
	
	hline!([prob[cm.ref.k]], label="ref", sp=1, dpi=300)
end

# ╔═╡ 187f792d-38fc-47f7-95f9-51e5dbc46534
begin
	matrices_S = get_sensitivity(model, :y; op)[1]
	matrices_T = get_comp_sensitivity(model, :y; op)[1]
	
	S 	= ss(matrices_S...) |> sminreal
	T 	= ss(matrices_T...) |> sminreal
	f1 	= bodeplot(S, label="S", plotphase=false)
	bodeplot!(T, label="T", plotphase=false)
	
	matrices_L, Lsys = get_looptransfer(model, :u; op)
	L = -ss(matrices_L...) |> sminreal
	ω = exp10.(LinRange(-4, 1, 600))
	
	f2 	= plot(diskmargin(L, 0, ω))
	f3 	= plot(diskmargin(L), xlims=(0, 7))
	wny = exp10.(-1.3:0.1:2)
	f4 	= nyquistplot(L, wny, unit_circle=true, ratio=1, legend=false)
	plot(f1, f2, f3, f4, size=(1000, 1000))
end

# ╔═╡ afa53b32-5946-4ce0-839e-28bbf59d1158
begin
	CS = linearize_named_ss(model, :y, :u; op) |> sminreal
	bodeplot(CS) # Oops, high gain for high frequencies -> noise amplification	
end

# ╔═╡ Cell order:
# ╟─f14003e2-d9a7-49ba-af25-1085fef8a2f9
# ╠═d6845b2a-ed77-4058-b503-de043ce8c107
# ╠═db3be3eb-d8aa-4750-bfa5-9e0aeb806c64
# ╠═01dd9e50-8476-4bdd-8eed-ca390844a22a
# ╠═607955ec-c529-4d0d-8e01-385585c9d7b5
# ╠═f95ea338-3093-4cbf-a846-4731ca7e39c4
# ╟─068ec563-0802-421a-ad49-855157f1e2a0
# ╠═72d9ec45-78c8-4009-b3ef-77b6d7e5420a
# ╟─3cb3bca3-c219-414d-b075-afa1ea8741ca
# ╠═efae2c84-46d0-4c6c-88d4-1ca5bc19875f
# ╠═1868e073-0483-411b-ae2e-c5c863946398
# ╟─56ab17b7-7c98-4f1c-bcd6-60b4261db8ae
# ╠═79d08299-8814-4823-bfad-161c63210082
# ╟─89fed73e-3eea-4341-9738-d6c2ccd07b13
# ╟─3126f474-4e01-4081-a084-44789101123e
# ╠═53b5adc6-eab1-4e51-8ae5-ca88a5ee0c4a
# ╠═3bd15fcb-d990-41f1-8784-199d8edc2d30
# ╟─824658b8-153c-488d-961d-ba3cba2a22dd
# ╠═798627ae-27e1-47bb-ba6e-51e2bd9361cd
# ╠═d4da467b-a761-4f0f-b902-2cd7b5bc8241
# ╟─8c872dbb-d499-4b03-88b6-5c94a88bd9f1
# ╠═246173b4-1618-4ae3-a30c-2f20e790acbe
# ╟─6dcfdd57-989b-4394-bef4-6a2c1bb170ea
# ╠═60f87548-4edb-40e4-9b38-aa61d9e39356
# ╟─4595cfcd-0231-4456-aa98-bcf80a3f2fd4
# ╠═6aabc0e9-3a08-45b7-9e7c-937c772b859b
# ╠═eb09d92e-df2a-4290-915e-89864787cf19
# ╟─68ae9371-f3a2-42a9-9b50-2a2717d67ab4
# ╠═187f792d-38fc-47f7-95f9-51e5dbc46534
# ╟─9ba5eaca-2be0-4db4-93bd-6bafb07aee52
# ╠═afa53b32-5946-4ce0-839e-28bbf59d1158
# ╟─2e64ab45-366e-4e9a-a8f1-2bb682a42868
# ╟─101b6e8a-6649-4b9a-a610-758fecaccca4
# ╟─bc75dfd0-dcd8-4c16-87a4-736ea1acb5d7
# ╟─9754bc74-531b-496d-98f9-608224ce5d5d
# ╟─99ebe7c4-2aa6-4c30-b065-b2d40c545168
# ╟─517032af-5b7a-4c1f-b6d9-5d5a0b5c9bad
