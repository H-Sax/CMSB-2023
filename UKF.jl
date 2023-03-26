using DifferentialEquations, Plots, LinearAlgebra, Distributions, OffsetArrays, Random, LaTeXStrings

function Valve(R, deltaP)
    q = 0.0
    if (-deltaP) < 0.0 
        q =  deltaP/R
    else
        q = 0.0
    end
    return q

end

function ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    τₑₛ = τₑₛ*τ
    τₑₚ = τₑₚ*τ
    #τ = 4/3(τₑₛ+τₑₚ)
    tᵢ = rem(t + (1 - Eshift) * τ, τ)

    Eₚ = (tᵢ <= τₑₛ) * (1 - cos(tᵢ / τₑₛ * pi)) / 2 +
         (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * (1 + cos((tᵢ - τₑₛ) / (τₑₚ - τₑₛ) * pi)) / 2 +
         (tᵢ <= τₑₚ) * 0

    E = Eₘᵢₙ + (Eₘₐₓ - Eₘᵢₙ) * Eₚ

    return E
end



function DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)

    τₑₛ = τₑₛ*τ
    τₑₚ = τₑₚ*τ
    #τ = 4/3(τₑₛ+τₑₚ)
    tᵢ = rem(t + (1 - Eshift) * τ, τ)

    DEₚ = (tᵢ <= τₑₛ) * pi / τₑₛ * sin(tᵢ / τₑₛ * pi) / 2 +
          (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * pi / (τₑₚ - τₑₛ) * sin((τₑₛ - tᵢ) / (τₑₚ - τₑₛ) * pi) / 2
    (tᵢ <= τₑₚ) * 0
    DE = (Eₘₐₓ - Eₘᵢₙ) * DEₚ

    return DE
end

# Model parameter values
Eshift = 0.0
Eₘᵢₙ = 0.03
τₑₛ = 0.3
τₑₚ = 0.45 
Eₘₐₓ = 1.5
Rmv = 0.06
τ = 1.0
 
function NIK!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u 
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = p

    # 1) Left Ventricle
    du[1] = (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) + pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # 2) Systemic arteries 
    du[2] = (Qav - Qs ) / Csa     
    # 3) Venous
    du[3] = (Qs - Qmv) / Csv 
    # 4) Left Ventricular Volume
    du[4] = Qmv - Qav 
    # 5) Aortic Valve flow
    du[5]    = Valve.(Zao, (pLV - psa)) - Qav
    # 6) Mitral Valve flow
    du[6]   = Valve(Rmv, (psv - pLV)) - Qmv 
    # 7) Systemic flow
    du[7]     = (du[2] - du[3]) / Rs
    nothing 
end
##
M = [1.  0  0  0  0  0  0
     0  1.  0  0  0  0  0
     0  0  1.  0  0  0  0
     0  0  0  1.  0  0  0
     0  0  0  0  0  0  0
     0  0  0  0  0  0  0 
     0  0  0  0  0  0  1. ]
     
Nik_ODE = ODEFunction(NIK!,mass_matrix=M)

u0 = [8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0]

p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

tspan = (0, τ * 30)

prob = ODEProblem(Nik_ODE, u0, tspan, p)

@time sol = solve(prob, Rodas5P(autodiff = false), adaptive = false, dt = 0.00225, reltol = 1e-12, abstol = 1e-12)

# Model solution plots
p1 = plot(sol, idxs=[1, 2], tspan=(6 * τ, 7 * τ), xlabel = L"Time [s]", ylabel = L"Pressure [mmHg]",  hidexaxis = nothing, labels=[L"P_{LV}" L"P_{SA}"])# Make a line plot
p2 = plot(sol, idxs=[4], tspan=(6 * τ, 7 * τ),xlabel = L"Time [s]", ylabel = L"Volume [ml]",  linkaxes = :all, labels="V_{LV}")
p3 = plot(sol, idxs=[5,6], tspan=(6 * τ, 7 * τ),xlabel = L"Time [s]", ylabel = L"Flow rate [ml/s]", linkaxes = :all, labels=[L"q_{1}" L"q_{3}"])
y = sol[1,:]
x = sol[4,:]
p4 =plot(x[1500:2000],y[1500:2000], label = false, xlabel = L"Volume-(ml)", ylabel = L"Pressure-(mmHg)", title = L"PV-Loop")
a = plot(p1, p4, p2, p3, layout=@layout([a b; c d]), legend = true)

N = 13335
# Create the observations with errors Presure and Volume
Obs = [sol[1,:] sol[2,:] sol[4,:]]
noise = zeros(N,3)
Nobs = zeros(N,3)
ϵ = Normal(0.0,0.025)
for i in 1:3
    for k in 1:N
    noise[k,i] = rand(ϵ,1)[1]
    Nobs[k,i]=Obs[k,i]*(1+noise[k,i])
    end
end


Nobs = Array(transpose(Nobs))
t = LinRange(0.0,30.0*τ,N)

plot(t,sol[1,:])
plot!(t, Nobs[1,:], label = "LV - P")

plot(t,sol[2,:])
plot!(t, Nobs[2,:], label = "LV - P")

plot(t,sol[4,:])
plot(t, Nobs[3,:], label = "LV - V")


## Constants needed for unsented method 
global const L  = 16 #dimension of state vector 
global const m = 3 # number of measurements
global const k = 0.0 # 
global const α = 1*10^(-3) # determines spread of sigma points 
global const β = 2.0
global const λ = α^2*(L+k) - L

## Define an augmented state vector 

# Define measurement Covariance matrix 
σ =5.0
R=(σ^2)*Matrix{Float64}(I,m,m)

χ = zeros(L,2L +1,N+1)
χ = OffsetArray(χ, 1:L, 1:2L+1, 0:N)
χf =  zeros(L,2L +1,N)
Wm = zeros(2L+1)
Wc = zeros(2L+1)
Xμ = zeros(L,N)
Xa = zeros(L,N)
PX = zeros(L,L,N)
PXA = zeros(L,L,N)
Y = zeros(m,2L +1,N)
Yμ = zeros(m,N)
PY = zeros(m,m,N)
PχY = zeros(L,m,N)
K = zeros(L,m,N)

# inital covariance matrix
PXA[:,:,1] = diagm([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.3, 0.3, 0.3, 0.3, 0.01]) #first cov matrix 
# Inital augmented state vector
Xa[:,1] = [sol[1]; rand(Normal(0.3, 0.01), 1)[1]; rand(Normal(0.45, 0.01), 1)[1]; rand(Normal(0.06, 0.01), 1)[1];  rand(Normal(0.033, 0.01), 1)[1]; rand(Normal(1.11, 0.3), 1)[1]; rand(Normal(1.13, 0.3), 1)[1]; rand(Normal(11.0, 0.3), 1)[1];rand(Normal(1.5, 0.3), 1)[1] ; rand(Normal(0.03, 0.01), 1)[1]]

PX[:,:,1] = PXA[:,:,1]
PY[:,:,1] = diagm([0.1, 0.1, 0.1])
# Weights 
Wm[1] = λ/(L+λ)
Wc[1] = λ/(L+λ) + (1 - α^2 + β)

@inbounds for i in 2:2L+1
    Wm[i] = 0.5/(L+λ)
    Wc[i] = 0.5/(L+λ)
end 


χ[:,1, 0] = Xa[:,1]

Lmatrix = LinearAlgebra.cholesky((PXA[:,:,1]) |> LinearAlgebra.Hermitian).L

@inbounds for (t, j) in zip(2:L+1, 1:L)
    χ[:, t, 0] = Xa[:,1] .+ sqrt((L + λ)) * Lmatrix[:, j]  
end

@inbounds for (t, j) in zip(L+2:2L+1, 1:L)
    χ[:, t, 0] = Xa[:,1] .- sqrt((L + λ)) * Lmatrix[:, j] 
end



condition2(u,t,integrator) = integrator.iter % 1 == 0 
 
function affect2!(integrator)
    i  = integrator.iter
    u_ = integrator.u
    p_ = integrator.p
    ## Start unsented algorithm
    Lmatrix = LinearAlgebra.cholesky((PXA[:,:,i]) |> LinearAlgebra.Hermitian).L

    Xa[:,i] = [u_;p_]
    χ[:,1, i] = Xa[:,i]

    @inbounds for (t, j) in zip(2:L+1, 1:L)
        χ[:, t, i] = Xa[:,i] .+ sqrt((L + λ)) * Lmatrix[:, j]  
    end

    @inbounds for (t, j) in zip(L+2:2L+1, 1:L)
        χ[:, t, i] = Xa[:,i] .- sqrt((L + λ)) * Lmatrix[:, j] 
    end

    @inbounds  for j in 1:2L+1
        a = ODEProblem(Nik_ODE, χ[1:7,j,i-1], (integrator.t - integrator.dt, integrator.t),χ[8:end,j,i-1]);
        res = DifferentialEquations.solve(a,Rodas5P(autodiff = false), adaptive = false,dt = 0.00225, reltol = 1e-12, abstol = 1e-12, saveat = [integrator.t]) 
        χf[:,j,i+1] = [res.u[1];χ[8:end,j,i]]
    end 

    a = zeros(L,2L+1)
    @inbounds  for j in 1:2L+1
        a[:,j] = Wm[j]*χf[:,j,i+1]
        Xμ[:,i+1] = sum(eachcol(a))
    end

    b = zeros(L,L,2L+1)
    @inbounds for j in 1:2L+1
        b[:,:,j] = Wc[j]*(χf[:,j,i+1] - Xμ[:,i+1])*transpose(χf[:,j,i+1] - Xμ[:,i+1])
        PX[:,:,i+1] = sum(b, dims = 3) + (1e-8)*Matrix{Float64}(I,L,L)
    end

    @inbounds for j in 1:2L+1
        Y[:,j,i+1] = χf[[1,2,4],j,i+1]
    end

    d = zeros(m,2L+1)
    @inbounds for j in 1:2L+1
        d[:,j] = Wm[j]*Y[:,j,i+1]
        Yμ[:,i+1] = sum(eachcol(d))
    end

    e = zeros(m,m,2L+1)
    @inbounds for j in 1:2L+1
        e[:,:,j] = Wc[j]*(Y[:,j,i+1] - Yμ[:,i+1])*transpose(Y[:,j,i+1] - Yμ[:,i+1]) 
        PY[:,:,i+1] = sum(e, dims = 3) + R
    end
 
    f = zeros(L,m,2L+1)
    @inbounds for j in 1:2L+1
        f[:,:,j] = Wc[j]*(χf[:,j,i+1] - Xμ[:,i+1])*transpose(Y[:,j,i+1] - Yμ[:,i+1])
        PχY[:,:,i+1] = sum(f, dims = 3)
    end

    # Now assimilate the prediction

    K[:,:,i+1] = PχY[:,:,i+1]*inv(PY[:,:,i+1])

    Xa[:,i+1] = Xμ[:,i+1] + K[:,:,i+1]*(Nobs[:,i+1] - Yμ[:,i+1])

    PXA[:,:,i+1] =  PX[:,:,i+1] - K[:,:,i+1]*PY[:,:,i+1]*transpose(K[:,:,i+1])

    integrator.u = Xa[1:7,i+1]
    integrator.p = Xa[8:end,i+1]

end
save_positions = (false,false)

cb2 = DiscreteCallback(condition2, affect2!, save_positions=save_positions)
tspan = (0, τ * 30)
p = 0.9*[0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

prob = ODEProblem(Nik_ODE,u0,tspan,p)
@time sol = solve(prob, Rodas5P(autodiff = false), adaptive = false, dt = 0.00225, reltol = 1e-12, abstol = 1e-12, callback = cb2)

### Covariance Plot 
using CairoMakie
begin

    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98));
    gabc = f[1:3,1]
    ga = gabc[1, 1] = GridLayout()
    gb = gabc[2, 1] = GridLayout()
    gc = gabc[3, 1] = GridLayout()

    gdef = f[1:3, 2] = GridLayout()
    gd = gdef[1, 1] = GridLayout()
    ge = gdef[2, 1] = GridLayout()
    gf = gdef[3, 1] = GridLayout()

    gghi = f[1:3, 3] = GridLayout()
    gg = gghi[1, 1] = GridLayout()
    gh = gghi[2, 1] = GridLayout()
    gi = gghi[3, 1] = GridLayout()

    x = sol.t
    ax = Axis(ga[1,1], xgridstyle = :dash, ygridstyle = :dash,  yticksize = 5, yscale = log10)
    lines!(x, PXA[8,8,:]; label = L"τ_{es}")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));


    ax = Axis(gb[1,1], xgridstyle = :dash, ygridstyle = :dash,  yticksize = 5, yscale = log10)
    lines!(x, PXA[9,9,:]; label = L"τ_{ep}")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gc[1,1], xgridstyle = :dash, ygridstyle = :dash,  yticksize = 5, xlabel = L"t(s)", yscale = log10)
    lines!(x, PXA[10,10,:]; label = L"Rmv")
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gd[1,1], xgridstyle = :dash, ygridstyle = :dash, yticksize = 5, yscale = log10)
    lines!(x, PXA[11,11,:]; label = L"Zao")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(ge[1,1], xgridstyle = :dash, ygridstyle = :dash,   yticksize = 5, yscale = log10)
    lines!(x, PXA[12,12,:]; label = L"Rs")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gf[1,1], xgridstyle = :dash, ygridstyle = :dash,  yticksize = 5, xlabel = L"t(s)", yscale = log10)
    lines!(x, PXA[13,13,:]; label = L"Csa")
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gg[1,1], xgridstyle = :dash, ygridstyle = :dash,  yticksize = 5, yscale = log10)
    lines!(x, PXA[14,14,:]; label = L"Csv")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gh[1,1], xgridstyle = :dash, ygridstyle = :dash,  yticksize = 5, yscale = log10)
    lines!(x, PXA[15,15,:]; label = L"E_{max}")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gi[1,1], xgridstyle = :dash, ygridstyle = :dash, yticksize = 5, xlabel = L"t(s)", yscale = log10)
    lines!(x, PXA[16,16,:]; label = L"E_{min}")
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    for (label, layout) in zip(["A", "B", "C", "D", "E", "F", "G", "H", "I"], [ga, gb, gc, gd, ge, gf, gg, gh, gi])
        Label(layout[1, 1, TopRight()], label,fontsize = 18,font = :bold,halign = :right)
    end
    
    resize_to_layout!(f)

    f
end  

# UKF estimate 
begin 
    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98));
    gabc = f[1:3,1]
    ga = gabc[1, 1] = GridLayout()
    gb = gabc[2, 1] = GridLayout()
    gc = gabc[3, 1] = GridLayout()

    gdef = f[1:3, 2] = GridLayout()
    gd = gdef[1, 1] = GridLayout()
    ge = gdef[2, 1] = GridLayout()
    gf = gdef[3, 1] = GridLayout()

    gghi = f[1:3, 3] = GridLayout()
    gg = gghi[1, 1] = GridLayout()
    gh = gghi[2, 1] = GridLayout()
    gi = gghi[3, 1] = GridLayout()

    x = sol.t
    ax = Axis(ga[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, Xa[8,:]; label = L"τ_{es}")
    lines!(x, [0.3])
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rb, bgcolor = (:grey90, 0.25));


    x = sol.t
    ax = Axis(gb[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, Xa[9,:]; label = L"τ_{ep}")
    lines!(x, [0.45])
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rb, bgcolor = (:grey90, 0.25));

    x = sol.t
    ax = Axis(gc[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)")
    lines!(x, Xa[10,:]; label = L"Rmv")
    lines!(x, [0.06])
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    x = sol.t
    ax = Axis(gd[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, Xa[11,:]; label = L"Zao")
    lines!(x, [0.033])
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    x = sol.t
    ax = Axis(ge[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, Xa[12,:]; label = L"Rs")
    lines!(x, [1.11])
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    x = sol.t
    ax = Axis(gf[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)")
    lines!(x, Xa[13,:]; label = L"Csa")
    lines!(x, [1.13])
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    x = sol.t
    ax = Axis(gg[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, Xa[14,:]; label = L"Csv")
    lines!(x, [11.])
    hidexdecorations!(ax, grid = false)
    axislegend(position = :lb, bgcolor = (:grey90, 0.25));

    x = sol.t
    ax = Axis(gh[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, Xa[15,:]; label = L"E_{max}")
    lines!(x, [1.5])
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    x = sol.t
    ax = Axis(gi[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)")
    lines!(x, Xa[16,:]; label = L"E_{min}")
    lines!(x, [0.03])
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    for (label, layout) in zip(["A", "B", "C", "D", "E", "F", "G", "H", "I"], [ga, gb, gc, gd, ge, gf, gg, gh, gi])
        Label(layout[1, 1, TopRight()], label,fontsize = 18,font = :bold,halign = :right)
    end

    resize_to_layout!(f)

    f
end    

## RMSE for UKF estimate

param = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]
R = zeros(9,length(sol.t))
for j in 1:9
    for i in 1:length(sol.t)
        R[j,i] =  sqrt(sum(Xa[(j+7),i] - param[j])^2/length(sol.t))
    end 
end 


begin 
    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98));
    gabc = f[1:3,1]
    ga = gabc[1, 1] = GridLayout()
    gb = gabc[2, 1] = GridLayout()
    gc = gabc[3, 1] = GridLayout()

    gdef = f[1:3, 2] = GridLayout()
    gd = gdef[1, 1] = GridLayout()
    ge = gdef[2, 1] = GridLayout()
    gf = gdef[3, 1] = GridLayout()

    gghi = f[1:3, 3] = GridLayout()
    gg = gghi[1, 1] = GridLayout()
    gh = gghi[2, 1] = GridLayout()
    gi = gghi[3, 1] = GridLayout()

    x = sol.t
    ax = Axis(ga[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, R[1,:]; label = L"τ_{es}")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gb[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, R[2,:]; label = L"τ_{ep}")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gc[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)")
    lines!(x, R[3,:]; label = L"Rmv")
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gd[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, R[4,:]; label = L"Zao")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(ge[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, R[5,:]; label = L"Rs")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gf[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)")
    lines!(x, R[6,:]; label = L"Csa")
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gg[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, R[7,:]; label = L"Csv")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :lt, bgcolor = (:grey90, 0.25));

    ax = Axis(gh[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5)
    lines!(x, R[8,:]; label = L"E_{max}")
    hidexdecorations!(ax, grid = false)
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    ax = Axis(gi[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)")
    lines!(x, R[9,:]; label = L"E_{min}")
    axislegend(position = :rt, bgcolor = (:grey90, 0.25));

    for (label, layout) in zip(["A", "B", "C", "D", "E", "F", "G", "H", "I"], [ga, gb, gc, gd, ge, gf, gg, gh, gi])
        Label(layout[1, 1, TopRight()], label,fontsize = 18,font = :bold,halign = :right)
    end

    resize_to_layout!(f)

    f
end   
