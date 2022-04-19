include("PDEstruct.jl")
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
        Type = 3
    elseif t == 1 && 0<s<1
        Type = 2
    else
        Type = 1
    end
    return Type
end


none() = nothing

PDEparam = VarElliptic(afun,rhs,bdy_type, bdy_Diri, bdy_Neum, bdy_Robin)



## FEM parameters
Ne = 2^5 # Ne elements each dimension
FEMparam = FEM_2dUnifQuadMesh(Ne)


## Solver: get nodal values and mass matrix
# assembly + linear solve
@time FEMsol, FEMstore = FEM_Solver(FEMparam,PDEparam)
# subsequent linear solve; rhs and bdy can be different 
@time FEMsol = FEM_SubsequentSolve(FEMparam,PDEparam,FEMstore)


## error: can use when truth exists
x = FEMparam.Grid_x
y = FEMparam.Grid_y
truth = [u([x[i],y[j]]) for j in 1:Ne+1 for i in 1:Ne+1]
Linf = maximum(abs.(truth - FEMsol)) / maximum(abs.(truth))
L2 = sqrt((truth - FEMsol)'*FEMstore.M*(truth - FEMsol) / (truth'*FEMstore.M*truth))
energy = sqrt((truth - FEMsol)'*FEMstore.A*(truth - FEMsol) / (truth'*FEMstore.A*truth))
@info "[Error] Relative energy err $energy, Relative L2 $L2, Linf $Linf"



# ## plot
function meshgrid(x,y)
    xx = repeat(x',length(y),1)
    yy = repeat(y,1,length(x))
    return xx,yy
end
xx, yy = meshgrid(x, y)

result=reshape(truth - FEMsol,Ne+1,Ne+1)'
figure()
plt.contourf(xx,yy,abs.(result))
colorbar()
display(gcf())