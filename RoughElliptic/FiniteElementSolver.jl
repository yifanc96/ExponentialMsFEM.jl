include("PDEstruct.jl")
include("FEMutility.jl")
using Logging
using ForwardDiff
using BenchmarkTools

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

function bdy(t,s)
    return u([t,s])
end
bdy_type = "Dirichlet"
PDEparam = VarElliptic(afun,rhs,bdy,bdy_type)



## FEM parameters
Nf = 2^9 # Nf elements each dimension
FEMparam = FEM_UnifQuadMesh(Nf)


## Solver: get nodal values and mass matrix
# assembly + linear solve
@time FEMsol, A, M = FEM_Solver(FEMparam,PDEparam)
# subsequent linear solve; rhs and bdy can be different 
@time FEMsol = FEM_SubsequentSolve(FEMparam,A,M,rhs,bdy)


## error: can use when truth exists
x = FEMparam.grid_x
y = FEMparam.grid_y
truth = [u([x[i],y[j]]) for j in 1:Nf+1 for i in 1:Nf+1]
Linf = maximum(abs.(truth - FEMsol)) / maximum(abs.(truth))
L2 = sqrt((truth - FEMsol)'*M*(truth - FEMsol) / (truth'*M*truth))
@info "[Error] Relative L2 $L2, Linf $Linf"



# ## plot
# function meshgrid(x,y)
#     xx = repeat(x',length(y),1)
#     yy = repeat(y,1,length(x))
#     return xx,yy
# end
# xx, yy = meshgrid(x, y)
