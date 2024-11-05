### A Pluto.jl notebook ###
# v0.19.16

#> [frontmatter]
#> title = "How to tune a Kalman filter"
#> date = "2023-03-03"
#> tags = ["kalmanfilter", " modeling", "controlsystems", " stateestimation"]
#> description = "This blog post attempts to bring intuition to the otherwise opaque choise of the covariance-matrix parameters of the Kalman filter."

using Markdown
using InteractiveUtils

# ╔═╡ c4b71bc3-0033-4e18-8bbf-f7f040bf4671
begin
	using ControlSystemsBase, RobustAndOptimalControl, LowLevelParticleFilters, Plots, LinearAlgebra, Distributions, PlutoUI
	
	Ts = 0.01 # Sample time
	m1 = 1    # Mass 1
	m2 = 1    # Mass 2
	k = 100   # Spring constant
	c = 1     # Damping constant
	A = [0 1 0 0;
	     -k/m1 -c/m1 k/m1 c/m1;
	     0 0 0 1;
	     k/m2 c/m1 -k/m2 -c/m2]
	B = [0; 1/m1; 0; 0]
	C = [1 0 0 0]
	
	Cwf = [0, 1, 0, 0]
	Bwf = [0; 0; 0; 0; 1/m1]
	Bws = [0; 1/m1; 0; -1/m2; 0]
	Awf = 0
	Aa = [A Cwf;
	      zeros(1, 4) Awf]
	Ba = [B; 0]
	Ca = [C 0]
	P  = ss(A, B, C, 0)    # Continuous-time system model
	Pa = ss(Aa, Ba, Ca, 0) # Augmented system model
end

# ╔═╡ f057fa96-b996-11ed-15f1-219d9c8f6959
md"""
# How to tune a Kalman filter

The celebrated Kalman filter finds applications in many fields of engineering and economics. While many are familiar with the basic concepts of the Kalman filter, almost equally many find the "tuning parameters" associated with a Kalman filter non intuitive and difficult to choose. While there are several parameters that can be tuned in a real-world application of a Kalman filter, we will focus on the most important ones: the process and measurement noise covariance matrices.

## The Kalman filter
The Kalman filter is a form of Bayesian estimation algorithm that estimates the state ``x`` of a linear dynamical system, subject to Gaussian noise ``e`` acting on the measurements as well as the dynamics, ``w``. More precisely, let the dynamics of a discrete-time linear dynamical system be given by
```math
\begin{aligned}
x_{k+1} &= Ax_k + Bu_k + w_k\\
y_k &= Cx_k + Du_k + e_k
\end{aligned}
```
where ``x_k \in \mathbb{R}^{n_x}`` is the state of the system at time ``k``, ``u_k \in \mathbb{R}^{n_u}`` is an external input, ``A`` and ``B`` are the state transition and input matrices respectively, ``C`` an output matrix and ``w_k \sim N(0, R_1)`` and ``e_k \sim N(0, R_1)``, are normally distributed process noise and measurement noise terms respectively. A state estimator like the Kalman filter allows us to estimate ``x`` given only noisy measurements ``y \in \mathbb{R}^{n_y}``, i.e., without necessarily having measurements of all the components of ``x`` available.[^obs] For this reason, state estimators are sometimes referred to as *virtual sensors*, i.e., they allows use to *estimate what we cannot measure*.



The Kalman filter is popular for several important reasons, for one, it is the *optimal estimator in the mean-square sense* if the system dynamics is linear (can be time varying) and the noise acting on the system is Gaussian. In most practical applications, neither of these conditions hold exactly, but they often hold sufficiently well for the Kalman filter to remain useful.[^nonlin] A perhaps even more useful property of the Kalman filter is that the posterior probability distribution over the state remains Gaussian throughout the operation of the filter, making it efficient to compute and store. 

[^obs]: Under a technical condition on the [*observability*](https://en.wikipedia.org/wiki/Observability) of the system dynamics.
[^nonlin]: Several nonlinear state estimators exist as well.

## What does "tuning the Kalman filter" mean?

To make use of a Kalman filter, we obviously need the dynamical model of the system given by the four matrices ``A, B, C`` and ``D``. We furthermore require a choice of the covariance matrices ``R_1`` and ``R_2``, and it is here a lot of aspiring Kalman-filter users get stuck. The covariance matrix of the measurement noise is often rather straightforward to estimate, just collect some measurement data when the system is at rest and compute the sample covariance, but we often lack any and all feeling for what the process noise covariance, ``R_1``, should be.

In this blog post, we will try to give some intuition for how to choose the process noise covariance matrix ``R_1``. We will come at this problem from a *disturbance-modeling* perspective, i.e., trying to reason about what disturbances act on the system and how, and what those imply for the structure and value of the covariance matrix ``R_1``. 
"""

# ╔═╡ 946ffc65-f180-4783-910b-41ff76045646
md"""
## Disturbance modeling

Intuitively, a disturbance acting on a dynamical system is some form of *unwanted* input. If you are trying to control the temperature in a room, it may be someone opening a window, or the sun shining on your roof. If you are trying to keep the rate of inflation at 2%, the disturbance may be a pandemic.

The linear dynamics assumed by the Kalman filter, here on discrete-time form
```math
x_{k+1} = Ax_k + Bu_k + w_k
```
make it look like we have very little control over the shape and form of the disturbance ``w``, but there are a lof of possibilities hiding behind this equation.

In the equation above, the disturbance ``w`` has the same dimension as the state ``x``. Implying that the covariance matrix ``R_1 \in \mathbb{R}^{n_x \times n_x}`` has ``n_x^2`` parameters. We start by noting that covariance matrices are symmetric and positive semi-definite, this means that only an upper or lower triangle of ``R_1`` contains free parameters. We further note that we can restrict the influence of ``w`` to a subset of the equations by introducing an input matrix ``B_w`` such that
```math
x_{k+1} = Ax_k + Bu_k + B_w \tilde{w}_k
```
where ``w = B_w \tilde{w}`` and ``\tilde{w}`` may have a smaller dimension than ``w``. To give a feeling for why this might be relevant, we consider a very basic system, the *double integrator*.

A double integrator appears whenever Newton's second law appears
```math
f = ma
```
This law states that the acceleration ``a`` of a system is proportional to the force ``f`` acting on it, on continuous-time statespace form,[^tf] this looks like
```math
\begin{aligned}
\dot p(t) &= v(t)\\
m\dot v(t) &= f(t)
\end{aligned}
```
or on the familiar matrix form:
```math
\dot x = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} x + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} f 
```
where ``x = [p, v]^T``. 


Now, what disturbances could possibly act on this system? The relation between velocity ``v`` and position ``p`` is certainly deterministic, and we cannot disturb the position of a system other than by continuously changing the velocity first (otherwise an infinite force would be required). This means that any disturbance acting on this system must take the form of a *disturbance force*, i.e., ``w = B_w w_f`` where ``B_w = \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix}``. A disturbance force ``w_f`` may be something like friction, air resistance or someone applying an unknown external force etc. This means that the disturbance has a single degree of freedom only, and we can write the dynamics of the system as
```math
\dot x = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} x + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} f + \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} w_f
```
where ``w_f`` is a scalar. This further means that the covariance matrix ``R_1`` has a *single free parameter only*, and we can write it as
```math
R_1 = \sigma_w^2 B_w B_w^{T} = \dfrac{\sigma_w^2}{m^2} \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}
```
where ``\sigma_w^2`` is the variance of the disturbance ``w_f``. This is now our tuning parameter that we use to trade off the filter response time vs. the noise in the estimate.

What may initially have appeared as a tuning parameter ``R_1`` with three parameters to tune, has now been reduced to a single parameter by reasoning about how a possible disturbance acts on the system dynamics! The double integrator is a very simple example, but it illustrates the idea that the structure of the disturbance covariance matrix ``R_1`` is determined by the structure of the system dynamics and the form of the disturbance.

[^tf]: As a transfer function in the Laplace domain, the double integrator looks like ``P(s)/F(s) = \frac{1}{s^2}`` where ``P`` is the Laplace-transform of the position and ``F`` that of the force.
"""

# ╔═╡ fcc1ce42-f1ca-4766-a688-8a14d4216cc2
md"""
## But white noise, really?

Having had a look at the structural properties of the dynamics noise, let's now consider its *spectrum*. With noise like ``w_k \sim N(0, R_1)``, where ``w_k`` is uncorrelated with ``w_j`` for ``j \neq k``, is called *white noise* in analogy with white light, i.e., "containing all frequencies", or, "has a flat spectrum". White noise can often be a reasonable assumption for measurement noise, but much less so for dynamics noise. If we come back to the example of the temperature controlled room, the disturbance implied by the sun shining on the roof is likely dominated by low frequencies. The sun goes up in the morning and down in the evening, and clouds that may block the sun for a while do not move infinitely fast etc. For a disturbance like this, modeling it as white noise may not be the best choice.

Fear not, we can easily give color to our noise and still write the resulting model on the form 
```math
\dot x = Ax + Bu + B_w w
```


Let's say that our linear system ``P`` can be depicted in block-diagram form as follows:
```
   │w
   ▼
┌─────┐
│  W  │
└──┬──┘
   │w̃
   │  ┌─────┐
 u ▼  │     │ y
 ──+─►│  P  ├───►
      │     │
      └─────┘
```
Here, ``w`` is filtered through another linear system ``W`` to produce ``\tilde{w}``. If ``w`` has a flat white spectrum, the spectrum of ``\tilde{w}`` will be colored by the frequency response of ``W``. Thus, if we want to model that the system is affected by low-frequency noise ``w̃``, we can choose ``W`` as some form of low-pass filter. If we write ``W`` on statespace form as 
```math
\begin{aligned}
\dot x_w &= A_w x_w + B_w w \\
w̃ &= C_w x_w
\end{aligned}
```
we can form an *augmented system model* ``P_a`` as follows:
```math
\begin{aligned}
\dot x &= A_a x_a + B_a u + B_{aw} w \\
y &= C_a x_a
\end{aligned}
```
where
```math
A_a = \begin{bmatrix} A & C_w \\ 0 & A_w \end{bmatrix}, \quad B_a = \begin{bmatrix} B \\ 0 \end{bmatrix}, \quad B_{aw} = \begin{bmatrix} 0 \\ B_w \end{bmatrix}, \quad C_a = \begin{bmatrix} C & 0 \end{bmatrix}
```
and
```math
x_a = \begin{bmatrix} x \\ x_w \end{bmatrix}
```
the augmented model has a state vector that is comprised of both the state vector of the original system ``P``, as well as the state vector ``x_w`` of the *disturbance model* ``W``. If we run a Kalman filter with this augmented model, the filter will estimate both the state of the original system ``P`` as well as the state of the disturbance model ``W`` for us! We have built what is called a *disturbance observer*.

It may at this point be instructive to reflect upon why we performed this additional step of modeling the disturbance? By including the disturbance model ``W``, we tell the Kalman filter what frequency-domain properties the disturbance has, and the filter can use these properties to make better predictions of the state of the system. This brings us to another key point of making use of a state estimator, it can perform *sensor fusion*.
"""

# ╔═╡ 99b91a40-5493-45c0-a8c7-8392c4cc6a95
md"""
## Sensor fusion
By making use of models of the dynamics, disturbances and measurement noise, the state estimator performs something often referred to as "sensor fusion". As the name suggests, sensor fusion is the process of combining information from multiple sensors to produce a more accurate estimate of the state of the system. In the case of the Kalman filter, the state estimator combines information from the dynamics model, the measurement model and the disturbance models to produce a more accurate estimate of the state of the system. We will contrast this approach to two common state-estimation heuristics
- Differentiation for velocity estimation
- Complementary filtering for orientation estimation

Velocity is notoriously difficult to directly measure in most applications, but measuring position is often easy. A naive approach to estimating velocity is to differentiate the position measurements. However, there are a couple of problems associated with this approach. First, the noise in the position measurements will be amplified by the differentiation, second, only differentiating the measured position ignores any information of the input to the system. Intuitively, if we know how the system behaves in response to an input, we should be able to use this knowledge to form a better estimate of both the position and the velocity?

Indeed, a Kalman filter allows you to estimate the velocity, taking into account both the input to the system and the noisy position measurement. If the model of the system is perfect, we do not even need a measurement, the model and the input is sufficient to compute what the velocity will be. In practice, models are never perfect, and we thus make use of the "fusion aspect" of the state estimator to incorporate the two different sources of information, the model and the measurement, to produce a better estimate of the velocity.

A slightly more complicated example is the complimentary filter. This filter is often used with inertial measurement units (IMUs), containing accelerometers and gyroscopes to estimate the orientation of a system. Accelerometers are often very noisy, but they measure the correct orientation on average. Gyroscopes are often very accurate, but their reading slowly drifts over time. The complimentary filter combines the information from the accelerometer and the gyroscope to produce a more accurate estimate of the orientation. This is done by low-pass filtering the accelerometer measurement to get rid of the high-frequency measurement noise, and high-pass filtering the gyroscope measurement to get rid of the low-frequency drift.

We can arrive at a filter with these same properties, together with a dynamical model that indicates the system's response to control inputs, by using a Kalman filter. In this case, we would include two different disturbance models, one acting on the accelerometer output ``y_a`` and one the gyroscope output ``y_g`` like this
```
            │wa
            ▼
         ┌─────┐
         │  Wa │
         └──┬──┘
            │
    ┌─────┐ ▼
u   │     ├─+─► ya
───►│  P  │
    │     ├─+─► yg
    └─────┘ ▲
            │
         ┌──┴──┐
         │  Wg │
         └─────┘
            ▲
            │wg
```
``W_a`` would here be chosen as some form of a high-pass filter to indicate that the acceleration is corrupted by high-frequency noise. Inversely, ``W_g`` would be chosen as a low-pass filter since the drift over time can be modeled as a low-frequency disturbance acting on the measurement. The models of the disturbances are thus the complements of the filters we would have applied if we had performed the filtering manually.

The complimentary filter makes the "complimentary assumption" ``W_g = 1 - W_a``, i.e., ``W_a`` and ``W_g`` sum to one. This is a simple and often effective heuristic, but the naive complementary filter does not make any use of the input signal ``u`` to form the estimate, and will thus suffer from phase loss, sometimes called *lag*, in response to inputs. This is particularly problematic when there are communication *delays* present between the sensor and the state estimator. During the delay time, the sensor measurements contain no information at all about any system response to inputs. 
"""

# ╔═╡ a74dd703-cdb7-4d69-80ee-6d72510c191c
md"""

## Discretization
So far, I have switched between writing dynamics in continuous time, i.e., on the form
```math
\dot x(t) = A x(t) + B u(t)
```
and in discrete time
```math
x_{k+1} = A x_k + B u_k
```
Physical systems are often best modeled in continuous time, while some systems, notably those living inside a computer, are inherently discrete time. Kalman filters are thus most often implemented in discrete time, and any continuous-time model must be discretized before it can be used in a Kalman filter. For control purposes, models are often discretized using a zero-order-hold assumption, i.e., input signals are assumed to be constant between sample intervals. This is often a valid assumption for control inputs, but not always for disturbance inputs. If the sample rate is fast in relation to the time constants of the system, the discretization method used does not matter all too much. For the purposes of this tutorial, we will use the zero-order-hold (ZoH) assumption for all inputs, including disturbances.

To learn the details on ZoH discretization consult [Discretization of linear state space models (wiki)](https://en.wikipedia.org/wiki/Discretization#discrete_function). Here, we will simply state a convenient way of computing this discretization, using the matrix exponential. Let ``A_c`` and ``B_c`` be the continuous-time dynamics and input matrices, respectively. Then, the discrete-time dynamics and input matrices are given by
```math
\begin{bmatrix}
A_d & B_d \\
0 & I
\end{bmatrix}
=
\exp\left(\begin{bmatrix}
A_c & B_c \\
0 & 0
\end{bmatrix} T_s\right)
```
where ``A_d`` and ``B_d`` are the discrete-time dynamics and input matrices, respectively, and ``T_s`` is the sample interval. The ``I`` in the bottom right corner is the identity matrix. To discretize the input matrix for a disturbance model, we simply replace ``B`` with ``B_w``, or put all the ``B`` matrices together by horizontal concatenation and discretize them all at once.

Discretizing the continuous time model of the double integrator with a disturbance force, we get
```math
\begin{aligned}
x_{k+1} &= \begin{bmatrix} 1 & T_s \\ 0 & 1 \end{bmatrix} x_k + \dfrac{1}{m}\begin{bmatrix} T_s^2/2 \\ T_s \end{bmatrix} f + \dfrac{1}{m}\begin{bmatrix} T_s^2/2 \\ T_s \end{bmatrix} w_f \\
y_k &= \begin{bmatrix} 1 & 0 \end{bmatrix} x_k + e_k
\end{aligned}
```
with the corresponding covariance matrix
```math
R_1 = \dfrac{\sigma^2_{w_f}}{m^2} \begin{bmatrix}
\frac{T_s^4}{4} & \frac{T_s^3}{2} \\
\frac{T_s^3}{2} & T_s^2
\end{bmatrix}
```
This may look complicated, but it still has a single tuning parameter only, ``\sigma_{w_f}``.


## Putting it all together
We will now try to put the learnings from above together, applied to a slightly more complicated example. This time, we will consider a double-mass model, where two masses are connected by a spring and a damper, and an input force can be applied to one of the masses.
![double-mass system](https://user-images.githubusercontent.com/3797491/222670124-ebbf9df1-1099-4571-bfa0-dcf4a92b0bec.png)
This is a common model of transmission systems, where the first mass represents the inertia of a motor, while the second mass represents the inertia of the load. The spring and damper represent the dynamics of the transmission, e.g., the gearbox and transmission shafts etc.

The model, without disturbances, is given by
```math
\begin{aligned}
\dot x &= \begin{bmatrix}
0 & 1 & 0 & 0 \\
-k/m_1 & -c/m_1 & k/m_1 & c/m_1 \\
0 & 0 & 0 & 1 \\
k/m_2 & c/m_1 & -k/m_2 & -c/m_2 \end{bmatrix} x + \begin{bmatrix} 0 \\ 1/m_1 \\ 0 \\ 0 \end{bmatrix} u\\
y &= \begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix} x + e
\end{aligned}
```
where ``x = [p_1, v_1, p_2, v_2]`` is the state vector, ``u = f`` is the input force and ``y = p_1`` is the measured position of the first mass.[^vel] The parameters ``m_1``, ``m_2``, ``k``, and ``c`` are the masses, spring constant, and damping constant, respectively.

What disturbances could act on such a system? One could imagine a friction force acting on the masses, indeed, most systems with moving parts are subject to friction. Friction is often modeled as a low-frequency disturbance, in particular Coulomb friction. The Coulomb friction is constant as long as the velocity does not cross zero, at which point it changes sign. We thus adopt an integrating model of this disturbance.

To model the fact that we are slightly uncertain about the dynamics of the flexible transmission, we could model a disturbance force acting on the spring. This could account for an uncertainty in a linear spring constant, but also model the fact that the transmission is not perfectly linear, i.e., it might be a *stiffening spring* or contain some *backlash* etc. The question is, what frequency properties should we attribute to this disturbance? Backlash is typically a low-frequency disturbance, but uncertainties in the stiffness properties of the spring would likely affect higher frequencies as well. We thus let this disturbance have a flat spectrum and omit a model of its frequency properties.

With the friction disturbance ``w_f`` and the spring disturbance ``w_s``, we can write the model as


```math
\begin{aligned}
\dot x &= A_a x_a + B_a u + B_{w_f} w_f + B_{w_s} w_s\\
y &= C_a x_a
\end{aligned}
```
where
```math
A_a = \begin{bmatrix} A & C_{w_f} \\ 0 & A_{w_f} \end{bmatrix},
\quad B_a = \begin{bmatrix} B \\ 0 \end{bmatrix},
\quad B_{w_f} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 1/m_1 \end{bmatrix},
\quad B_{w_s} = \begin{bmatrix} 0 \\ 1/m_1 \\ 0 \\ -1/m_2 \\ 0 \end{bmatrix}
\quad C_a = \begin{bmatrix} C & 0 \end{bmatrix}
```

and ``x_a = \begin{bmatrix} x \\ x_{w_f} \end{bmatrix}``. When modeling ``w_f`` as an integrating disturbance ``W_f = 1/(m_1s)``, we get the dynamics
```math
A_{w_f} = 0, \quad B_{w_f} = 1/m_1, \quad C_{w_f} = 1
```

[^vel]: It's unfortunately common that a single position only is measurable, and oftentimes the sensor is located at the moror output. We are thus unable to directly measure what we often ultimately care about, the position and velocity of the load side.

### In code
To demonstrate how constructing a Kalman filter for the double-mass system would look using JuliaSim, we start by defining the dynamics in the form of a `StateSpace` model, as well as all the input matrices.
"""

# ╔═╡ fdae3793-1f91-46b7-84fc-d953d98ea57e
md"""
In practice, models are never perfect, we thus create another version of the model where the spring constant ``k`` is 10% larger. We will use this perturbed model to test how well the Kalman filter can estimate the state of the system in the presence of model mismatch.
"""

# ╔═╡ 45a31f9e-8e47-4f0f-8cd0-1f985a1aff05
begin
	k_actual = 1.1*k
	A_actual = [0 1 0 0;
	            -k_actual/m1 -c/m1 k_actual/m1 c/m1;
	            0 0 0 1;
	            k_actual/m2 c/m1 -k_actual/m2 -c/m2]
	P_actual = ss(A_actual, B, C, 0) # Actual system model
end;

# ╔═╡ fba3c116-3671-4c44-9c97-30fad150a1d1
md"""We then define the covariance matrices according to the above reasoning, and discretize them using the `c2d` function. Finally, we construct a `KalmanFilter` object."""

# ╔═╡ 90026e82-326a-47d9-90ca-bf4dc2353ed2
begin
	σf  = 100  # Standard deviation of friction disturbance
	σs  = 10   # Standard deviation of spring disturbance
	σy  = 0.01 # Standard deviation of measurement noise
	R1  = σf^2 * Bwf * Bwf' + σs^2 * Bws * Bws' # Covariance of continuous-time disturbance
	R2  = σy^2 * I(1)     # Covariance of measurement noise
	
	Pad = c2d(ss(Aa, [Ba Bwf Bws], Ca, 0), Ts) # Discrete-time augmented system model
	Pd = Pad[:, 1]      # Discrete-time system model
	Bwfd = Pad.B[:, 2]  # Discrete-time friction disturbance input matrix
	Bwsd = Pad.B[:, 3]  # Discrete-time spring disturbance input matrix
	R1d = σf^2 * Bwfd * Bwfd' + σs^2 * Bwsd * Bwsd' + 1e-8I |> Symmetric # Covariance of discrete-time disturbance
	
	kf  = KalmanFilter(ssdata(Pd)..., R1d, R2)
end;

# ╔═╡ c8407825-bb01-4c12-baf1-c1f1c21407b8
md"""
Here, we added a small amount of diagonal covariance to ``R_1`` (``10^{-8}``) to make it strictly positive definite. This improves the numerics as well as acts like a "catch all" covariance that prevents the estimator from malfunctioning completely in case we have forgotten to model some disturbance affecting the system.

When running a Kalman filter in a real-world application, we often receive and process measurements in real time. However, for the purposes of this blog post, we generate some inputs and outputs by simulating the system. We construct a PID controller `Cfb` and close the loop around this controller to to simulate how the system behaves under feedback. We let the reference position ``r`` be an integrated square signal. To obtain both control input and measured output from the simulation, we form the feedback interconnection using the function `feedback_control`. 
"""

# ╔═╡ 3a9f0d6d-d89b-4dce-9904-7e8f59c0b2d6
begin
	Cfb = pid(100, 1, 0.2, Tf=2Ts)
	G = feedback_control(P_actual, Cfb)*tf(1, [0.1, 1]) # Closed-loop system
	r = 0.1cumsum(sign.(sin.(0.1.*(0:0.02:10).^2)))     # Reference position
	timevec = range(0, length=length(r), step=Ts)
	res = lsim(G, r', timevec) # Perform linear simulation
	inputs = res.y[2,:]        # Extract the control input
	outputs = res.y[1,:]       # Extract the output
	plot(res, ylabel=["Position mass 1 \$y = x_1\$" "Input signal \$u\$"])
end

# ╔═╡ 813bb92e-7963-4309-92d2-61adb7388bbc
md"""
To simulate a Coulomb friction disturbance, we extract the velocity of the first mass, ``v_1 = x_2``, and compute the friction force as ``- k_f \operatorname{sign}(v)``. We then apply this disturbance to the input as seen by the Kalman filter in order to simulate the effect of the disturbance on the estimator. We then call `forward_trajectory` to perform Kalman filtering along the entire recorded trajectory, and plot the result alongside the true state trajectory as obtained by the function `lsim`.
"""

# ╔═╡ 0b215ecc-6331-4c72-ac1a-c031cf52bf12
begin
	wf = map(eachindex(res.t)) do i
	    ui = inputs[i]
	    v = res.x[2,i]
	    -20*sign(v) # Coulomb friction disturbance
	end
	
	u = map(eachindex(res.t)) do i
		# The input seen by the state estimator is the true input + the disturbance
	    ui = inputs[i] + wf[i]
	    [ui]
	end
	y = [[yi + σy*randn()] for yi in outputs] # Outputs with added measurement noise
	sol = forward_trajectory(kf, u, y)
	true_states = lsim(P_actual, inputs', timevec).x
	plot(sol, layout=(7,1), size=(800, 1299))
	plot!(true_states', sp=(1:4)', lab="True")
	plot!(-wf, sp=5, lab="Friction force")
end

# ╔═╡ 30aef3c4-1ac0-4f96-b40d-c76ac731e4a1
md"""
This figure shows the 5 state variables ``x_a = [p_1, v_1, p_2, v_2, x_{w_f}]`` of the augmented system model ``P_a``, as well as the input signal ``u`` and the measured output ``y``. The notation ``x(t | t-1)`` denotes the estimate of ``x(t)`` given data available at time ``t-1``. In the plot of the estimated disturbance state ``x_5 = x_{w_f}``, we plot also the true friction force from the simulation. Analyzing the results, we see that the state estimator needs a couple of samples to catch up to the change in the friction disturbance, but it does correctly converge to a noisy estimate of the friction force in steady state. 

We have thus built and simulated a disturbance observer that is capable of estimating the friction force as well as the full state of the dynamical system, based on a dynamical model and a measurement of the position of one of the masses only!
"""

# ╔═╡ a6bce974-45a3-4737-b68e-a3c8a4126fdb
md"""
## Concluding remarks

In this blog post, we have discussed how we could give meaning to the all-so-important covariance matrix of the dynamics noise appearing in a Kalman filter. By reasoning about what disturbances we expect to act on the system, both due to disturbance inputs and due to model mismatch, we can construct a covariance matrix that is consistent with our expectations. Here, we modeled disturbances in order to improve the estimation of the state of the system, however, in some applications the disturbance estimate itself is the primary quantity of interest. Examples of disturbance estimation include estimation of external contact forces acting on a robot, and room occupancy estimation (a person contributes approximately 100W of heat power to their surrounding). With all these examples in mind, no wonder people sometimes call state estimators for "virtual sensors"!

### Potential improvements
Although we managed to estimate the friction force acting on the double-mass system above, there is still plenty of room for improvement. If we have an accurate, nonlinear model of the friction, we could likely cancel out the friction disturbance much faster than what our simple integrating model did.[^int] Even if we did not have an accurate model, making use of the fact that the friction force has a complicated behavior when the velocity crosses zero would likely improve the performance of the estimator. This could easily be done in a Kalman-filtering framework by giving the covariance a little bump when the velocity crosses or approaches zero. An example of such an adaptive covariance approach is given in the [Adaptive Kalman-filter tutorial](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/adaptive_kalmanfilter/).

[^int]: Using a Kalman filter with an integrating disturbance model in a state-feedback controller results in a controller with integral action. See [Integral action](https://help.juliahub.com/juliasimcontrol/dev/integral_action/) for more details.

### Further reading
To learn more about state estimation in JuliaSim, see the following resources:
- [State estimation with ModelingToolkit](https://help.juliahub.com/juliasimcontrol/dev/examples/state_estimation/)
- [Noise adaptive Kalman filter tutorial](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/adaptive_kalmanfilter/)
- [Parameter estimation for state estimators](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/parameter_estimation/)
- [Estimation of Kalman filters directly from data using statespace model estimation](https://baggepinnen.github.io/ControlSystemIdentification.jl/dev/ss/)
- [Disturbance modeling with ModelingToolkit](https://help.juliahub.com/juliasimcontrol/dev/examples/mtk_disturbance_modeling/)
- [Disturbance modeling for Model-Predictive Control (MPC)](https://help.juliahub.com/juliasimcontrol/dev/examples/disturbance_rejection_mpc/)
- In this blog post, we considered state estimation with a linear dynamics model. State estimation can be used also with [nonlinear models, even with DAE models](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/)!
- In the example above, we treated ``\sigma_f^2`` and ``\sigma_s^2`` as tuning parameters to be manually selected. However, it is possible to use data to estimate these parameters using, e.g., maximum likelihood estimation. This blog post is long enough as it is, so that's a story for another time.[^est]

[^est]: Here's part of that story if you are really curious: [Parameter estimation for state estimators](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/parameter_estimation/).
"""

# ╔═╡ 21c55254-a494-467b-8dcc-06a098f3e9e0
TableOfContents()

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ControlSystemsBase = "aaaaaaaa-a6ca-5380-bf3e-84a91bcd477e"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LowLevelParticleFilters = "d9d29d28-c116-5dba-9239-57a5fe23875b"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RobustAndOptimalControl = "21fd56a4-db03-40ee-82ee-a87907bee541"

[compat]
ControlSystemsBase = "~1.3.6"
Distributions = "~0.25.85"
LowLevelParticleFilters = "~3.3.2"
Plots = "~1.38.6"
PlutoUI = "~0.7.50"
RobustAndOptimalControl = "~0.4.20"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cc37d689f599e8df4f464b2fa3870ff7db7492ef"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ec9c36854b569323551a6faf2f31fda15e3459a7"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.2.0"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "Static"]
git-tree-sha1 = "2c144ddb46b552f72d7eafe7cc2f50746e41ea21"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSolve]]
git-tree-sha1 = "9441451ee712d1aec22edad62db1a9af3dc8d852"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.ComponentArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "LinearAlgebra", "Requires", "StaticArrayInterface"]
git-tree-sha1 = "2736dee49260e412a352b2d0a37fb863f9a5b559"
uuid = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
version = "0.13.8"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "89a9db8d28102b094992472d333674bd1a83ce2a"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.1"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.ControlSystemsBase]]
deps = ["DSP", "ForwardDiff", "IterTools", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MatrixEquations", "MatrixPencils", "Polyester", "Polynomials", "Printf", "Random", "RecipesBase", "SparseArrays", "StaticArrays", "UUIDs"]
git-tree-sha1 = "fc4b9660f5a9e6ef1bbd40f707418ab4960f037a"
uuid = "aaaaaaaa-a6ca-5380-bf3e-84a91bcd477e"
version = "1.3.6"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "da8b06f89fce9996443010ef92572b193f8dca1f"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.8"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DescriptorSystems]]
deps = ["LinearAlgebra", "MatrixEquations", "MatrixPencils", "Polynomials", "Random"]
git-tree-sha1 = "defeb5ca1195c44f7b1933d367aac1ac473a2f08"
uuid = "a81e2ce2-54d1-11eb-2c75-db236b00f339"
version = "1.3.7"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ChainRulesCore", "DataStructures", "Distributions", "DocStringExtensions", "EnumX", "FastBroadcast", "ForwardDiff", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "Parameters", "PreallocationTools", "Printf", "RecursiveArrayTools", "Reexport", "Requires", "SciMLBase", "Setfield", "SparseArrays", "Static", "StaticArraysCore", "Statistics", "Tricks", "TruncatedStacktraces", "ZygoteRules"]
git-tree-sha1 = "a057a5fe2a6a05f28ef1092d5974a0c2986be23c"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.121.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "fb372fc76a20edda014dfc2cdb33f23ef80feda6"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.85"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "d1248fceea0b26493fd33e8e9e8c553270da03bd"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.2.5"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "ed1b56934a2f7a65035976985da71b6a65b4f2cf"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.18.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "1cd7f0af1aa58abc02ea1d872953a97359cb87fa"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.4"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "660b2ea2ec2b010bb02823c6d0ff6afd9bdc5c16"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.71.7"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d5e1fd17ac7f3aa4c5287a61ee28d4f8b8e98873"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.71.7+0"

[[deps.GenericSchur]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "fb69b2a645fa69ba5f474af09221b9308b160ce6"
uuid = "c145ed77-6b09-5dd9-b285-bf645a82121e"
version = "0.5.3"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "d3b3624125c1474292d0d8ed0f65554ac37ddb23"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "37e4657cd56b11abe3d10cd4a1ec5fbdb4180263"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.7.4"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "734fd90dd2f920a2f1921d5388dcebe805b262dc"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.14"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6f2675ef130a300a112286de91973805fcc5ffbc"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.91+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "2422f47b34d4b127720a18f86fa7b1aa2e141f29"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.18"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["ChainRulesCore", "LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "42970dad6b0d2515571613010bd32ba37e07f874"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.9.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "ChainRulesCore", "CloseOpenIntervals", "DocStringExtensions", "ForwardDiff", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "SIMDTypes", "SLEEFPirates", "SnoopPrecompile", "SpecialFunctions", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "2acf6874142d05d5d1ad49e8d3786b8cd800936d"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.152"

[[deps.LowLevelParticleFilters]]
deps = ["Distributions", "ForwardDiff", "Lazy", "LinearAlgebra", "LoopVectorization", "PDMats", "Parameters", "Polyester", "Printf", "Random", "RecipesBase", "Requires", "SciMLBase", "SimpleNonlinearSolve", "StaticArrays", "Statistics", "StatsAPI", "StatsBase", "SymbolicIndexingInterface"]
git-tree-sha1 = "e712ab060f2e75198a15de190d65e9ad32a1ad8d"
uuid = "d9d29d28-c116-5dba-9239-57a5fe23875b"
version = "3.3.2"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "2ce8695e1e699b68702c03402672a69f54b8aca9"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "2c3fc86d52dfbada1a2e5e150e50f06c30ef149c"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.6.2"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "b7e8016e06c4b2da1281f0f6801ca32a023b34c9"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.2.7"

[[deps.MatrixPencils]]
deps = ["LinearAlgebra", "Polynomials", "Random"]
git-tree-sha1 = "c14a030f3614ee9486da70be2e091cca6d4b02e1"
uuid = "48965c70-4690-11ea-1f13-43a2532b2fa8"
version = "1.7.6"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MonteCarloMeasurements]]
deps = ["Distributed", "Distributions", "ForwardDiff", "GenericSchur", "LinearAlgebra", "MacroTools", "Random", "RecipesBase", "Requires", "SLEEFPirates", "StaticArrays", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "3be0d822973d4edfe432bea8704d2570d0522db7"
uuid = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
version = "1.1.1"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Observables]]
git-tree-sha1 = "6862738f9796b3edc1c09d0890afce4eca9e7e93"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.4"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "6503b77492fd7fcb9379bf73cd31035670e3c509"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9ff31d101d987eb9d66bd8b176ac7c277beccd09"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.20+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "1903afc76b7d01719d9c30d3c7d501b61db96721"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.4"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6f4fbcd1ad45905a5dee3f4256fabb49aa2110c6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.7"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "c95373e73290cf50a8a22c3375e4625ded5c5280"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "da1d3fb7183e38603fcdd2061c47979d91202c97"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.38.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "0fe4e7c4d8ff4c70bfa507f0dd96fa161b115777"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.3"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "MakieCore", "RecipesBase"]
git-tree-sha1 = "a10bf14e9dc2d0897da7ba8119acc7efdb91ca80"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.5"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "Requires"]
git-tree-sha1 = "f739b1b3cc7b9949af3b35089931f2b58c289163"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.12"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "261dddd3b862bd2c940cf6ca4d1c8fe593e457c8"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.3"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "e974477be88cb5e3040009f3767611bc6357846f"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.11"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "ZygoteRules"]
git-tree-sha1 = "3dcb2a98436389c0aac964428a5fa099118944de"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.38.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RobustAndOptimalControl]]
deps = ["ChainRulesCore", "ComponentArrays", "ControlSystemsBase", "DescriptorSystems", "Distributions", "GenericSchur", "LinearAlgebra", "MatrixEquations", "MatrixPencils", "MonteCarloMeasurements", "Optim", "Printf", "Random", "RecipesBase", "Statistics", "UUIDs", "UnPack"]
git-tree-sha1 = "557fc563711d0db8006ac2054b76efb5920b7ed6"
uuid = "21fd56a4-db03-40ee-82ee-a87907bee541"
version = "0.4.20"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "50314d2ef65fce648975a8e80ae6d8409ebbf835"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.5"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "cda0aece8080e992f6370491b08ef3909d1c04e7"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.38"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "Preferences", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SnoopPrecompile", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
git-tree-sha1 = "fe55d9f9d73fec26f64881ba8d120607c22a54b0"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.88.0"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "8419114acbba861ac49e1ab2750bae5c5eda35c4"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.1.22"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleNonlinearSolve]]
deps = ["ArrayInterface", "DiffEqBase", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "Reexport", "Requires", "SciMLBase", "SnoopPrecompile", "StaticArraysCore"]
git-tree-sha1 = "326789bbaa1b65b809bd4596b74e4fc3be5af6ac"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "0.1.13"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "d0435ba43ab5ad1cbb5f0d286ca4ba67029ed3ee"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.4"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "5589ab073f8a244d2530b36478f53806f9106002"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.2.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2d7d9e1ddadc8407ffd460e24218e37ef52dd9a3"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.16"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "2842f1dbd12d59f2728ba79f4002cd6b61808f8b"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.4.8"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SymbolicIndexingInterface]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "f8ab052bfcbdb9b48fad2c80c873aa0d0344dfe5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.2.2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "c97f60dd4f2331e1a495527f80d242501d2f9865"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.1"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils"]
git-tree-sha1 = "47b1c66a0a4f98312e667992fb9cf7611eaf1c97"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.0.1"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "952ba509a61d1ebb26381ac459c5c6e838ed43c4"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.60"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "ed8d92d9774b077c53e1da50fd81a36af3744c1c"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6edfe154ad7b313c01aceca188c05c835c67360"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.4+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─f057fa96-b996-11ed-15f1-219d9c8f6959
# ╟─946ffc65-f180-4783-910b-41ff76045646
# ╟─fcc1ce42-f1ca-4766-a688-8a14d4216cc2
# ╟─99b91a40-5493-45c0-a8c7-8392c4cc6a95
# ╟─a74dd703-cdb7-4d69-80ee-6d72510c191c
# ╠═c4b71bc3-0033-4e18-8bbf-f7f040bf4671
# ╟─fdae3793-1f91-46b7-84fc-d953d98ea57e
# ╠═45a31f9e-8e47-4f0f-8cd0-1f985a1aff05
# ╟─fba3c116-3671-4c44-9c97-30fad150a1d1
# ╠═90026e82-326a-47d9-90ca-bf4dc2353ed2
# ╟─c8407825-bb01-4c12-baf1-c1f1c21407b8
# ╠═3a9f0d6d-d89b-4dce-9904-7e8f59c0b2d6
# ╟─813bb92e-7963-4309-92d2-61adb7388bbc
# ╠═0b215ecc-6331-4c72-ac1a-c031cf52bf12
# ╟─30aef3c4-1ac0-4f96-b40d-c76ac731e4a1
# ╟─a6bce974-45a3-4737-b68e-a3c8a4126fdb
# ╟─21c55254-a494-467b-8dcc-06a098f3e9e0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
