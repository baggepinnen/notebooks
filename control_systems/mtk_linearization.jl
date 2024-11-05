### A Pluto.jl notebook ###
# v0.19.39

using Markdown
using InteractiveUtils

# ╔═╡ fdaba736-dae9-11ee-33de-35e987fbe98a
import Pkg

# ╔═╡ fdaba786-dae9-11ee-0c82-d5319c896496
# This cell disables Pluto's package manager and activates the global environment. Click on ? inside the bubble next to Pkg.activate to learn more.
# (added automatically because a sysimage is used)
Pkg.activate()

# ╔═╡ fdaba790-dae9-11ee-1932-af6255c16750
using ModelingToolkit, JuliaSimControl

# ╔═╡ 335a2fb6-bcaa-4816-88d0-efe6a1224c5d
using ControlSystemsBase, ControlSystemsMTK, Plots; theme(:juno)

# ╔═╡ 57cdc467-3e73-40dc-830e-5ab36ffc6aa6
md"""
# Linearization of a ModelingToolkit model

This notebook performs linearization of an MTK model using 
- `linearize`
- `named_ss` (the same as `linearize` but returns a state space system with signal names)
- Batch linearization 
"""

# ╔═╡ 00a676fe-b874-4ff4-8eb9-ec9dbae75362
rcam = JuliaSimControl.ControlDemoSystems.rcam();

# ╔═╡ e404c1ac-2344-48a4-bd5e-43f7be501351
rcam.rcam_model

# ╔═╡ 0e4ea8ec-5586-4849-895e-613894dc5707
operating_point = merge(rcam.x0, rcam.u0)

# ╔═╡ 670cdcaa-dfc8-439d-80e9-5cba4e438ca6
matrices, simplified_sys = linearize(
	rcam.rcam_model,
	rcam.inputs,
	states(rcam.iosys);
	op = operating_point
);

# ╔═╡ b41fcb6e-b2f1-4c40-9e72-b15e636e040c
P = ss(matrices...)

# ╔═╡ 0dab62c1-d17c-401b-9511-c213eea0b52f
P_named = named_ss(
	rcam.rcam_model,
	rcam.inputs,
	states(rcam.iosys);
	op = operating_point
)

# ╔═╡ 3a8b9794-2a88-4d36-aaa0-8773495c7aed
bodeplot(P_named[[:u, :ϕ], :uA], size=(600, 800), margin=5Plots.mm)

# ╔═╡ 5555c7f6-9563-4f99-b9e7-bfa1f49965d6
md"""
# Benchmarking
"""

# ╔═╡ 06b9c150-e854-4ed5-b81f-d1341465eeca
@time linearize(
	rcam.rcam_model,
	rcam.inputs,
	states(rcam.iosys);
	op = operating_point
);

# ╔═╡ 7f01554d-7e3c-4009-bbba-8680fefad652
cm = complete(rcam.rcam_model);

# ╔═╡ 9bbf9b43-ed57-44c2-bb84-afc5145a531e
ops = map(range(-1, 1, 1000)) do i
	opi = copy(operating_point)
	opi[cm.ρ] = 0.1*i*rcam.rcam_constants[:ρ]
	opi
end

# ╔═╡ da35db06-1ccc-4693-a8eb-fff9de2e09eb
md"""
# Batch linearization

Linearize in 1 000 different operating points
"""

# ╔═╡ 1958b588-fc57-46e6-9d84-a42385b2e233
@time Ps, ssys = batch_ss(
	rcam.rcam_model,
	rcam.inputs,
	states(rcam.iosys)[1:1],
	ops
);

# ╔═╡ c07626cb-c988-4508-999f-5d8ac04f2de7
bodeplot([P[1,2] for P in Ps], legend=false, alpha=0.5, line_z = eachindex(Ps)')

# ╔═╡ 8d7badba-1860-499b-b38e-3544181cc9d3
function ControlSystemsMTK.batch_linearize(sys, inputs, outputs, ops::AbstractVector{<:AbstractDict}; t = 0.0,
        allow_input_derivatives = false,
        kwargs...)
    lin_fun, ssys = ModelingToolkit.linearization_function(sys, inputs, outputs; op=ops[1], kwargs...)
    lins = map(ops) do op
        linearize(ssys, lin_fun; op, t, allow_input_derivatives)
    end
    lins, ssys
end

# ╔═╡ Cell order:
# ╟─57cdc467-3e73-40dc-830e-5ab36ffc6aa6
# ╠═fdaba736-dae9-11ee-33de-35e987fbe98a
# ╠═fdaba786-dae9-11ee-0c82-d5319c896496
# ╠═fdaba790-dae9-11ee-1932-af6255c16750
# ╠═00a676fe-b874-4ff4-8eb9-ec9dbae75362
# ╠═e404c1ac-2344-48a4-bd5e-43f7be501351
# ╠═0e4ea8ec-5586-4849-895e-613894dc5707
# ╠═670cdcaa-dfc8-439d-80e9-5cba4e438ca6
# ╠═335a2fb6-bcaa-4816-88d0-efe6a1224c5d
# ╠═b41fcb6e-b2f1-4c40-9e72-b15e636e040c
# ╠═0dab62c1-d17c-401b-9511-c213eea0b52f
# ╠═3a8b9794-2a88-4d36-aaa0-8773495c7aed
# ╟─5555c7f6-9563-4f99-b9e7-bfa1f49965d6
# ╠═06b9c150-e854-4ed5-b81f-d1341465eeca
# ╠═7f01554d-7e3c-4009-bbba-8680fefad652
# ╠═9bbf9b43-ed57-44c2-bb84-afc5145a531e
# ╟─da35db06-1ccc-4693-a8eb-fff9de2e09eb
# ╠═1958b588-fc57-46e6-9d84-a42385b2e233
# ╠═c07626cb-c988-4508-999f-5d8ac04f2de7
# ╟─8d7badba-1860-499b-b38e-3544181cc9d3
