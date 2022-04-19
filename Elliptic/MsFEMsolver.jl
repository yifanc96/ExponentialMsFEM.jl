include("PDEstruct.jl")
include("MsFEMutility.jl")
include("FEMutility.jl")
using Logging
using ForwardDiff
# using BenchmarkTools
using PyPlot

## PDE parameters
function afun(t,s)
    # epsilon1 = 1/5;
    # epsilon2 = 1/13;
    # epsilon3 = 1/17;
    # epsilon4 = 1/31;
    # epsilon5 = 1/65;
    # return 1/6*((1.1+sin(2*pi*t/epsilon1))/(1.1+sin(2*pi*s/epsilon1))+
    #         (1.1+sin(2*pi*s/epsilon2))/(1.1+cos(2*pi*t/epsilon2))+
    #         (1.1+cos(2*pi*t/epsilon3))/(1.1+sin(2*pi*s/epsilon3))+
    #         (1.1+sin(2*pi*s/epsilon4))/(1.1+cos(2*pi*t/epsilon4))+
    #         (1.1+cos(2*pi*t/epsilon5))/(1.1+sin(2*pi*s/epsilon5))+
    #         sin(4*s^2*t^2)+1);
    return 1.0
end

function u(x)
    return sin(0.5*π*x[1])*sin(π*x[2])*exp(x[1]+2*x[2])
end

function rhs(t,s)
    x = [t,s]
    result = -sum(ForwardDiff.gradient(x -> afun(x[1],x[2]), x).*ForwardDiff.gradient(u, x)) - afun(x[1],x[2])*tr(ForwardDiff.hessian(u,x))

    return result
    # return t^4-s^3+1;
end

function bdy_Diri(t,s)
    return u([t,s])
end

function bdy_Neum(t,s)
    return ForwardDiff.derivative(t->u([t,s]),t)
end

function bdy_Robin(t,s)
    function bdy_Robin1(t,s)
        return 1
    end
    return bdy_Robin1(t,s), -ForwardDiff.derivative(t->u([t,s]),t) + bdy_Robin1(t,s)*u([t,s])
end

function bdy_type(t,s)
    Type = nothing
    # Neuman and Robin boundary condition should be on open intervals
    if t == 0 && 0<s<1
        Type = 1
    elseif t == 1 && 0<s<1
        Type = 1
    else
        Type = 1
    end
    return Type
end


none() = nothing

PDEparam = VarElliptic(afun,rhs,bdy_type, bdy_Diri, bdy_Neum, bdy_Robin)



## MsFEM parameters
Nce = 2^6 # Ne elements each dimension
Nfe = 2^5
MsFEMparam = MsFEM_2d2ScaleUnifQuadMesh(Nce,Nfe)


## Solver:
@time coarse_sol, sol_bubble, MsFEMstore = MsFEM_CoarseSolver(MsFEMparam,PDEparam)
# subsequent linear solve; rhs and bdy can be different 
@time coarse_sol, sol_bubble = MsFEM_SubsequentSolve(MsFEMparam,PDEparam,MsFEMstore)

## error calculation: can use when truth exists
# coarse accuracy
Nce = MsFEMparam.Nce
x = MsFEMparam.CGrid_x
y = MsFEMparam.CGrid_y

FEMparam = FEM_2dUnifQuadMesh(Nce)
FEMstore = FEM_StiffnMassAssembly(FEMparam,PDEparam)

truth = [u([x[i],y[j]]) for j in 1:Nce+1 for i in 1:Nce+1]
Linf = maximum(abs.(truth - coarse_sol)) / maximum(abs.(truth))
L2 = sqrt((truth - coarse_sol)'*FEMstore.M*(truth - coarse_sol) / (truth'*FEMstore.M*truth))
energy = sqrt((truth - coarse_sol)'*FEMstore.A*(truth - coarse_sol) / (truth'*FEMstore.A*truth))
@info "[Error] Coarse solution: Relative energy err $energy, Relative L2 $L2, Linf $Linf"


# fine accuracy
@time fine_sol = MsFEM_FineConstruct(coarse_sol, sol_bubble, MsFEMstore)

Ne = Nce*Nfe
FEMparam = FEM_2dUnifQuadMesh(Ne)
@time FEMstore = FEM_StiffnMassAssembly(FEMparam,PDEparam)
x = FEMparam.Grid_x
y = FEMparam.Grid_y

truth = [u([x[i],y[j]]) for j in 1:Ne+1 for i in 1:Ne+1]
Linf = maximum(abs.(truth - fine_sol)) / maximum(abs.(truth))
L2 = sqrt((truth - fine_sol)'*FEMstore.M*(truth - fine_sol) / (truth'*FEMstore.M*truth))
energy = sqrt((truth - fine_sol)'*FEMstore.A*(truth - fine_sol) / (truth'*FEMstore.A*truth))
@info "[Error] Fine solution: Relative energy err $energy, Relative L2 $L2, Linf $Linf"



# ## plot
function meshgrid(x,y)
    xx = repeat(x',length(y),1)
    yy = repeat(y,1,length(x))
    return xx,yy
end
xx, yy = meshgrid(x, y)

result=reshape(truth - fine_sol,Ne+1,Ne+1)'
figure()
plt.contourf(xx,yy,abs.(result))
colorbar()
display(gcf())