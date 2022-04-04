using SparseArrays
using LinearAlgebra
using Logging
import Base.Threads: nthreads, @threads


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


function FEM_GlobalAssembly(FEMparam,PDEparam)

    # number of codes, including nodes on the boundary
    N_f = FEMparam.Nf
    nNodes = (N_f+1)*(N_f+1)

    # for sparse stiffness mtx and load matrix A, M construction
    Icol= zeros(16*N_f^2)
    Jrow= copy(Icol)
    Aval = copy(Icol)
    Mval = copy(Icol)

    # for right hand side construction
    F = zeros(nNodes);
    
    # run over all elements
    println("[multithreading] using ", Threads.nthreads(), " threads")
    for i = 1:N_f
        @threads for j = 1:N_f 

            local_K, local_f = FEM_LocalAssembly(FEMparam,PDEparam, i, j)
            # compute inner product in each elements
            for p = 1:4
                global_p = FEMparam.loc2glo(N_f, i, j, p);
                for q = 1:4
                    index = 16*N_f*(i-1)+16*(j-1)+4*(p-1)+q
                    global_q = FEMparam.loc2glo(N_f, i, j, q);
                    Icol[index] = global_p
                    Jrow[index] = global_q
                    Aval[index] = local_K[p, q]

                    # mid-point quadrature
                    if p == q # same
                        Mval[index] = 1/9/N_f^2
                    elseif p==q+2 || p==q-2 # diagonal
                        Mval[index] = 1/36/N_f^2
                    else # adjacent
                        Mval[index] = 1/18/N_f^2
                    end
                end
                F[global_p] = F[global_p] + local_f[p];
            end
        end
    end

    # boundary location: recall that our ordering is x first y second
    b = reduce(vcat,collect.([1:N_f+1,N_f+2:N_f+1:(N_f+1)*(N_f+1),2*(N_f+1):N_f+1:(N_f+1)*(N_f+1),N_f*N_f+N_f+2:(N_f+1)*(N_f+1)-1]))


    A = sparse(Icol,Jrow,Aval,nNodes,nNodes)
    M = sparse(Icol,Jrow,Mval,nNodes,nNodes)

    # Dirichlet boundary condition
    if PDEparam.bdy_type == "Dirichlet"
        A[b,:] .= 0; 

        # for Dirichlet zero bondary condition
        # A[:,b] .= 0; F[b] .= 0;  

        # for general Dirichlet boundary condition
        F[1:N_f+1] .= PDEparam.bdy.(FEMparam.grid_x,0.0)
        F[N_f+2:N_f+1:(N_f+1)*(N_f+1)] .= PDEparam.bdy.(0.0, FEMparam.grid_y[2:end])
        F[2*(N_f+1):N_f+1:(N_f+1)*(N_f+1)] .= PDEparam.bdy.(1.0, FEMparam.grid_y[2:end])
        F[N_f*N_f+N_f+2:(N_f+1)*(N_f+1)-1] .= PDEparam.bdy.(FEMparam.grid_x[2:end-1],1.0)
        
        A[b,b] .= sparse(I, length(b),length(b));
    else
        @info "other boundary condition not supported now"
    end

    @info "[Assembly] finish assembly of stiffness matrice"
    return A, M, F
end

function FEM_LocalAssembly(FEMparam, PDEparam, i, j)
    # the stiffness matrix required locally to compute local nodal basis 
    # for the bilinear boundary value with 1 at node 
    xlow = FEMparam.grid_x[i];
    xhigh = FEMparam.grid_x[i+1];
    ylow = FEMparam.grid_y[j];
    yhigh = FEMparam.grid_y[j+1];
    
    x = (xlow+xhigh)/2;
    y = (ylow+yhigh)/2;
    
    local_K = zeros(4,4)
    local_f = zeros(4);

    # mid-point quadrature for stiffness matrix
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
    
    # mid-point quadrature for rhs
    for i = 1:4
        local_f[i] = PDEparam.rhs(x,y)*(xhigh-xlow)^2/4;
    end
    
    return local_K, local_f
end


# domain [0,1]*[0,1], Nf is the number of interior DOF in each dimension
function FEM_Solver(FEMparam,PDEparam)
    # FEM for VarElliptic, uniform quadrilaternal mesh, so we only need one parameter N_f that is the number of interior DOF in each dimension, to fully characterize the mesh
    A, M, F = FEM_GlobalAssembly(FEMparam,PDEparam)
    sol = A\F
    @info "[Linear solver] linear system solved"
    
    # N_f = FEMparam.Nf
    # result=reshape(sol,N_f+1,N_f+1)'

    # the ordering is that x moves first y second
    return sol, A, M
end

## subsequent solve
function FEM_SubsequentSolve(FEMparam,A,M,rhs::Function,bdy::Function)
    N_f = FEMparam.Nf
    x = FEMparam.grid_x
    y = FEMparam.grid_y
    F = M*[rhs(x[i],y[j]) for j in 1:Nf+1 for i in 1:Nf+1]
    F[1:N_f+1] .= bdy.(FEMparam.grid_x,0.0)
    F[N_f+2:N_f+1:(N_f+1)*(N_f+1)] .= bdy.(0.0, FEMparam.grid_y[2:end])
    F[2*(N_f+1):N_f+1:(N_f+1)*(N_f+1)] .= bdy.(1.0, FEMparam.grid_y[2:end])
    F[N_f*N_f+N_f+2:(N_f+1)*(N_f+1)-1] .= bdy.(FEMparam.grid_x[2:end-1],1.0)

    sol = A\F
    return sol
end