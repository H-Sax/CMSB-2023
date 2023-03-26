using DifferentialEquations, CairoMakie, LinearAlgebra, Distributions, OffsetArrays, Random, LaTeXStrings, GlobalSensitivity, QuasiMonteCarlo, Statistics

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

## GSA ##
x = range(start = 15,stop = 16,step = 0.00225)

f1 = function (p) # Para 
    prob_func(prob,i,repeat) = remake(prob;p=p[:,i])
    ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
    sol = solve(ensemble_prob,Rodas5P(autodiff = false), adaptive = false, dt = 0.00225, reltol = 1e-12, abstol = 1e-12, EnsembleThreads();saveat=x,trajectories=size(p,2))
    # Now sol[i] is the solution for the ith set of parameters
    out = zeros(1335,size(p,2))
    for i in 1:size(p,2)
      out[1:445,i] = Array(sol[i][1,:]')
      out[446:890,i] = Array(sol[i][2,:]')
      out[891:1335,i] = Array(sol[i][4,:]')
    end
    out
end

N = 3000
lb = [0.21, 0.36, 0.042, 0.0231, 0.777, 0.791, 7.7, 1.05, 0.021]
ub = [0.34, 0.585, 0.078, 0.0429, 1.443, 1.469, 14.3, 1.95, 0.039]

bounds = tuple.(lb,ub)
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(N, lb, ub, sampler)
@time sobol_result_time = gsa(f1,Sobol(),A,B, batch=true)

## Plots for first order Sobol indices

# LV pressure 
s11 = sobol_result_time.S1[1:445,:]

# Systemic Artery pressure
s21 = sobol_result_time.S1[446:890,:]

# LV Volume 
s31 = sobol_result_time.S1[891:1335,:]

#### Time Averaged Sobol indices #####
sol = solve(prob,Rodas5P(autodiff = false), adaptive = false, dt = 0.00225, reltol = 1e-12, abstol = 1e-12, saveat=x)

### LV.P #####
S1_LVP = s11
S11_LVP = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:length(x)
    S11_LVP[i] = (sum(S1_LVP[1:j,i]*var(sol[1,1:j])))/(sum(var(sol[1,1:j])))/length(x)
    end 
end 

### SA.p #####
S1_SAP = s21
S11_SAP = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:length(x)
    S11_SAP[i] = (sum(S1_SAP[1:j,i]*var(sol[2,1:j])))/(sum(var(sol[2,1:j])))/length(x)
    end 
end 

### LV.V#####
S1_LVV = s31
S11_LVV = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:length(x)
    S11_LVV[i] = (sum(S1_LVV[1:j,i]*var(sol[4,1:j])))/(sum(var(sol[4,1:j])))/length(x)
    end 
end 

### Whole plots 
S1 = [S11_LVP S11_SAP S11_LVV]'

## PCA calculation 
F = transpose(S1)*S1

e_deomp=eigen(F)

λ = abs.(e_deomp.values)
Q = abs.(e_deomp.vectors)

e_value_sum = sum(λ)

e = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:9
    e[i] = sum(λ[j]*Q[i,j])/e_value_sum
    end 
end 
p=sortperm(e,rev=true)

## Orthogonality plot
Orth_heat1 = Matrix{Float64}(undef,9,9)
for j in 1:9
    for i in 1:9
        if i==j
            Orth_heat1[i,j] = 0
        else 

        Orth_heat1[i,j] = sin(acos(((transpose(S1[:,i])*S1[:,j]))/(norm(S1[:,i])*norm(S1[:,j]))-1e-15))  #Slight numerical rounding error without the additional add on 
        end 
    end 
end 

# Histogram calculation 
a1 = Orth_heat1[2:end,1]
a2 = Orth_heat1[3:end,2]
a3 = Orth_heat1[4:end,3]
a4 = Orth_heat1[5:end,4]
a5 = Orth_heat1[6:end,5]
a6 = Orth_heat1[7:end,6]
a7 = Orth_heat1[8:end,7]
a8 = Orth_heat1[9:end,8]
a = reduce(vcat, (a1,a2,a3,a4,a5,a6,a7,a8))

begin
    f = Figure(resolution = (900, 600),backgroundcolor = RGBf(0.98, 0.98, 0.98));

    ax = Axis(f[1,1], xticklabelrotation = π / 3, xticklabelalign = (:right, :center), xticks = (1:3, [L"LV.P", L"SA.P", L"LV.V"]), yticks = (1:9, [L"τ_{es}", L"τ_{ep}", L"Rmv", L"Zao", L"Rs", L"Csa", L"Csv", L"E_{max}", L"E_{min}"]), title = L"Sobol - First~Order", xlabel = L"Measurements", ylabel = L"Parameters")
    hm = CairoMakie.heatmap!(ax,S1, colormap=:plasma)
    for i in 1:3, j in 1:9
        txtcolor = S1[i, j] < -1000.0 ? :white : :black
        text!(ax, "$(round(S1[i,j], digits = 2))", position = (i, j),
            color = txtcolor, align = (:center, :center), fontsize = 15)
    end
    CairoMakie.Colorbar(f[1,2],hm);

    ax = Axis(f[1,3],title=L"Parameter~Importance", xticks = (1:9, [L"τ_{es}", L"τ_{ep}", L"Rmv", L"Zao", L"Rs", L"Csa", L"Csv", L"E_{max}", L"E_{min}"][p]), xlabel = L"Parameters", ylabel = L"Importance")
    CairoMakie.scatter!( e[p], label=L"Condition~Number")



    ax1 = Axis(f[2,1], xticks = (1:9, [L"τ_{es}", L"τ_{ep}", L"Rmv", L"Zao", L"Rs", L"Csa", L"Csv", L"E_{max}", L"E_{min}"]), yticks = (1:9, [L"τ_{es}", L"τ_{ep}", L"Rmv", L"Zao", L"Rs", L"Csa", L"Csv", L"E_{max}", L"E_{min}"]), title = L"S1-Orthogonality~Matrix")
    hm1 = CairoMakie.heatmap!(ax1,Orth_heat1, colormap=:plasma)
    for i in 1:9, j in 1:9
        txtcolor = Orth_heat1[i, j] < -0.0 ? :white : :black
        text!(ax1, "$(round(Orth_heat1[i,j], digits = 2))", position = (i, j),
            color = txtcolor, align = (:center, :center), fontsize = 15)
    end
    CairoMakie.Colorbar(f[2,2],hm1, label = L"Orthogonality~Score", ticks = 0.0:0.2:1.0)

    ax = Axis(f[2,3], xticks = 0.0:0.1:1.0, title = L"Orthogonality~Spread", xlabel = L"Orthogonality~Score", ylabel = L"Density")
    hist!(ax, a, color = :values, bins = 0.0:0.1:1.0, colormap= :plasma, strokewidth = 1, strokecolor = :black)

    for (label, layout) in zip(["A", "B", "C", "D"], [f[1,1], f[1,3], f[2,1], f[2,3]])
        Label(layout[1, 1, TopRight()], label,fontsize = 18,font = :bold,halign = :right)
    end

    f
end 
