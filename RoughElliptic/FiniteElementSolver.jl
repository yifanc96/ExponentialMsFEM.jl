using SparseArrays
using LinearAlgebra
using Logging
# PDEs
abstract type AbstractPDEs end

struct VarElliptic <: AbstractPDEs
    # domain is fixed [0,1]*[0,1]
    a::Function
    rhs::Function
    bdy::Function
end

struct FEM_UnifQuadMesh{Ti,Tf}
    Nf::Ti # number of D.O.F in each dimension
    grid_x::Vector{Tf}
    grid_y::Vector{Tf}
    loc2glo::Function
    # all but the first attribute are automatically generated
end

# constructor
function FEM_UnifQuadMesh(Nf)
    xx = collect(LinRange(0, 1, Nf+1))
    yy = copy(xx)
    function loc2glo(N, m, n, i)
        # N: number of elements in one direction
        if i <= 2 # bottom
            idx = (N+1)*(n-1) + m + i - 1;
        else # top 
            idx = (N+1)*n + m + 4-i;
        end    
    end        
    @info "[Mesh generation] mesh generated, $(Nf+1) nodes in each dimension"
    return FEM_UnifQuadMesh(Nf,xx,yy,loc2glo)
end

# domain [0,1]*[0,1], Nf is the number of interior DOF in each dimension
function FEM_solver(FEMparam,PDEparam)
    # FEM for VarElliptic, uniform quadrilaternal mesh, so we only need one parameter N_f that is the number of interior DOF in each dimension, to fully characterize the mesh
    A, M, F = FEM_GlobalAssembly(FEMparam,PDEparam)
    sol = A\F
    @info "[Linear solver] linear system solved"
    N_f = FEMparam.Nf
    result=reshape(sol,N_f+1,N_f+1)'
    return result, M
end

function FEM_GlobalAssembly(FEMparam,PDEparam)

    # number of codes, including nodes on the boundary
    N_f = FEMparam.Nf
    nNodes = (N_f+1)*(N_f+1)
    grid_x = FEMparam.grid_x
    grid_y = FEMparam.grid_y

    # for sparse stiffness mtx and load matrix A, M construction
    Icol= zeros(16*N_f^2)
    Jrow= copy(Icol)
    Aval = copy(Icol)
    Mval = copy(Icol)
    # for right hand side construction
    F = zeros(nNodes);
    
    # run over all elements
    for i = 1:N_f
        for j = 1:N_f  
            local_K, local_f = FEM_LocalAssembly(PDEparam, grid_x, grid_y, i, j)
            for p = 1:4
                global_p = FEMparam.loc2glo(N_f, i, j, p);
                for q = 1:4
                    index = 16*N_f*(i-1)+16*(j-1)+4*(p-1)+q
                    global_q = FEMparam.loc2glo(N_f, i, j, q);
                    Icol[index] = global_p
                    Jrow[index] = global_q
                    Aval[index] = local_K[p, q]
                    if p == q
                        Mval[index] = 1/9/N_f^2
                    elseif p==q+2 || p==q-2
                        Mval[index] = 1/36/N_f^2
                    else 
                        Mval[index] = 1/18/N_f^2
                    end
                end
                F[global_p] = F[global_p] + local_f[p];
            end
        end
    end

    # boundary location
    b = reduce(vcat,collect.([1:N_f+1,N_f+2:N_f+1:(N_f+1)*(N_f+1),2*(N_f+1):N_f+1:(N_f+1)*(N_f+1),N_f*N_f+N_f+2:(N_f+1)*(N_f+1)-1]))

    A = sparse(Icol,Jrow,Aval,nNodes,nNodes)
    M = sparse(Icol,Jrow,Mval,nNodes,nNodes)
    A[b,:] .= 0; A[:,b] .= 0; F[b] .= 0; 
    A[b,b] .= sparse(I, length(b),length(b));
    @info "[Assembly] finish assembly of stiffness matrice"
    return A, M, F
end

function FEM_LocalAssembly(PDEparam, grid_x, grid_y, i, j)
    # the stiffness matrix required locally to compute local nodal basis 
    # for the bilinear boundary value with 1 at node 
    xlow = grid_x[i];
    xhigh = grid_x[i+1];
    ylow = grid_y[j];
    yhigh = grid_y[j+1];
    
    x = (xlow+xhigh)/2;
    y = (ylow+yhigh)/2;
    
    local_K = zeros(4,4)
    local_f = zeros(4);

    for i = 1:4 
        for j = 1:4
            if i==j
                local_K[i,j] = 2/3*PDEparam.a(x,y);
            elseif i==j+2 || i==j-2
                local_K[i,j] = -1/3*PDEparam.a(x,y);
            else 
                local_K[i,j] = -1/6*PDEparam.a(x,y);
            end
        end
    end
    
    for i=1:4
        local_f[i] = PDEparam.rhs(x,y)*(xhigh-xlow)^2/4;
    end
    
    return local_K, local_f
end



function afun(t,s)
    epsilon1 = 1/5;
    epsilon2 = 1/13;
    epsilon3 = 1/17;
    epsilon4 = 1/31;
    epsilon5 = 1/65;
    return 1/6*((1.1+sin(2*pi*t/epsilon1))/(1.1+sin(2*pi*s/epsilon1))+
            (1.1+sin(2*pi*s/epsilon2))/(1.1+cos(2*pi*t/epsilon2))+
            (1.1+cos(2*pi*t/epsilon3))/(1.1+sin(2*pi*s/epsilon3))+
            (1.1+sin(2*pi*s/epsilon4))/(1.1+cos(2*pi*t/epsilon4))+
            (1.1+cos(2*pi*t/epsilon5))/(1.1+sin(2*pi*s/epsilon5))+
            sin(4*s^2*t^2)+1);
end

function rhs(t,s)
    return t^4-s^3+1;
end


function bdy(t,s)
    return 0
end

# grid generation
Nf = 2^8
FEMparam = FEM_UnifQuadMesh(Nf)
PDEparam = VarElliptic(afun,rhs,bdy)
@time FEMsol, M = FEM_solver(FEMparam,PDEparam)
@info "[Error]"

