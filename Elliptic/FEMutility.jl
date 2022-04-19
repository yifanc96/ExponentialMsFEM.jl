using SparseArrays
using LinearAlgebra
using Logging
import Base.Threads: nthreads, @threads


# some lesson: exact calculation of integral of tent functions leads to best accuracy!
# can either pull out some functions (e.g. a, f, or u) out of the integral and then use exact formula for tent functions, or can write a, f, u as linear combination of tent functions and obtain exact integral for the interactions between tent functions
# the latter approach is safer. The former approach needs to pay attention to boundaries, where basis functions change
# thus, combine accuracy and convenience: use stiffness and mass matrices, but for the boundary condition, can use the latter approach (do not need to introduce boundary element; this is why we always implement neuman and robin boundary condition in an open interval, in which case there is no boundary of boundary issues)

struct FEM_2dUnifQuadMesh{Ti,Tf}
    Ne::Ti # number of elements in each dimension
    Grid_x::Vector{Tf} # uniform grid in x axis, boundary included
    Grid_y::Vector{Tf} # uniform grid in y axis, boundary included
    Bdy_coordinates::Matrix{Tf} # bdy coordinates, used for getting boundary data
    Bdy_indices::Vector{Ti} # bdy index among all the nodes; ordering: one runs over x first then y
    ElemNode_loc2glo::Function # local index (node of an element) to global index (global indexed node); one runs over i first then j
end

# store the stiffness and mass matrix
struct FEM_2dMatrices{Ti,Tf}
    A::SparseMatrixCSC{Tf,Ti}
    M::SparseMatrixCSC{Tf,Ti}
end

# constructor
function FEM_2dUnifQuadMesh(Ne)
    x = collect(LinRange(0, 1, Ne+1))
    y = copy(x)

    Bdy_coordinates = vcat(
        [x[1:end] y[1]*ones(Ne+1)], 
        [x[1]*ones(Ne) y[2:end]], 
        [x[end]*ones(Ne) y[2:end]], 
        [x[2:end-1] y[end]*ones(Ne-1)]
    )

    Bdy_indices = reduce(
        vcat,collect.(
            [1:Ne+1,
            Ne+2:Ne+1:(Ne+1)*(Ne+1),
            2*(Ne+1):Ne+1:(Ne+1)*(Ne+1),
            Ne*Ne+Ne+2:(Ne+1)*(Ne+1)-1]
        )
    )

    function ElementNode_loc2glo(N, i, j, ind_node)
        # N: number of elements in one direction
        # i,j is the location of the element
        # edge marks the index of the edge (from 1 to 4, from bottom to top)
        if ind_node <= 2 # bottom
            global_idx = (N+1)*(j-1) + i + ind_node - 1;
        else # top 
            global_idx = (N+1)*j + i + 4 - ind_node;
        end 
        return global_idx  
    end

    @info "[Mesh generation] mesh generated, $(Ne+1) nodes in each dimension"
    return FEM_2dUnifQuadMesh(Ne,x,y,Bdy_coordinates, Bdy_indices, ElementNode_loc2glo)
end


function FEM_StiffnMassAseembly(FEMparam,PDEparam)
    d = 2
    Ne = FEMparam.Ne

    # for sparse stiffness mtx A, and mass matrix M construction
    Icol= zeros(16*Ne^2) # 16*Ne^2: Ne^2 elements, each one leads to 4^2 edge interactions
    Jrow= copy(Icol)
    Aval = copy(Icol)
    Mval = copy(Icol)
    
    println("[multithreading] using ", Threads.nthreads(), " threads")
    # run over all elements
    for i = 1:Ne
        @threads for j = 1:Ne 
            # local stiffness and local mass matrix
            local_A, local_M = FEM_LocalAssembly(FEMparam,PDEparam, i, j)

            for p = 1:4
                global_p = FEMparam.ElemNode_loc2glo(Ne, i, j, p);
                for q = 1:4
                    index = 16*Ne*(i-1)+16*(j-1)+4*(p-1)+q
                    global_q = FEMparam.ElemNode_loc2glo(Ne, i, j, q);
                    Icol[index] = global_p
                    Jrow[index] = global_q
                    Aval[index] = local_A[p, q]
                    Mval[index] = local_M[p, q]
                end
            end
        end
    end
    A = sparse(Icol,Jrow,Aval,(Ne+1)^d,(Ne+1)^d)
    M = sparse(Icol,Jrow,Mval,(Ne+1)^d,(Ne+1)^d)
    @info "[Assembly] finish assembly of stiffness and mass matrice"
    return FEM_2dMatrices(A,M)
end

function FEM_BdyRhsAssembly(FEMparam,PDEparam,FEMmtx)
    
    A = copy(FEMmtx.A) # stiffness matrix for solving the systen
    M = FEMmtx.M

    # right hand side
    Ne = FEMparam.Ne
    x = FEMparam.Grid_x
    y = FEMparam.Grid_y
    F = [PDEparam.rhs(x[i],y[j]) for j in 1:Ne+1 for i in 1:Ne+1]
    F = M * F

    # boundary location
    bdy_loc = FEMparam.Bdy_indices
    bdy_points = FEMparam.Bdy_coordinates
    bdy_type = PDEparam.bdy_type.(bdy_points[:,1],bdy_points[:,2])
    
    # Dirichlet boundary
    Diri_loc = findall(x->x==1,bdy_type)
    if length(Diri_loc) > 0
        Diri_bdy = PDEparam.bdy_Diri.(bdy_points[Diri_loc,1],bdy_points[Diri_loc,2])
        Diri_loc = bdy_loc[Diri_loc] # global index
        A[Diri_loc,:] .= 0
        F[Diri_loc] .= Diri_bdy
        A[Diri_loc,Diri_loc] .= sparse(I, length(Diri_loc),length(Diri_loc));
    end

    Neum_loc = findall(x->x==2,bdy_type)
    if length(Neum_loc) > 0
        Neum_bdy = PDEparam.bdy_Neum.(bdy_points[Neum_loc,1],bdy_points[Neum_loc,2])
        Neum_loc = bdy_loc[Neum_loc] # global index
        F[Neum_loc] += 1/(Ne)*Neum_bdy 
    end

    Robin_loc = findall(x->x==3,bdy_type)
    if length(Robin_loc) > 0
        Robin_bdy = [PDEparam.bdy_Robin.(bdy_points[Robin_loc[i],1],bdy_points[Robin_loc[i],2]) for i in 1:length(Robin_loc)]
        Robin_bdy = reduce(vcat, Robin_bdy)
        
        Robin_bdy1 = vcat([Robin_bdy[i][1] for i in 1:length(Robin_loc)])
        Robin_bdy2 = vcat([Robin_bdy[i][2] for i in 1:length(Robin_loc)])
        Robin_loc = bdy_loc[Robin_loc] # global index
        A[Robin_loc,Robin_loc] = A[Robin_loc,Robin_loc] + diagm(Robin_bdy1)*1/(Ne)
        F[Robin_loc] += 1/(Ne)*Robin_bdy2
    end

    @info "[Assembly] finish incorporating boundary data"
    return A, F
end

function FEM_LocalAssembly(FEMparam, PDEparam, i, j)
    # the stiffness matrix required locally to compute local nodal basis 
    # for the bilinear boundary value with 1 at node 
    Ne = FEMparam.Ne
    xlow = FEMparam.Grid_x[i];
    xhigh = FEMparam.Grid_x[i+1];
    ylow = FEMparam.Grid_y[j];
    yhigh = FEMparam.Grid_y[j+1];
    
    mid_x = (xlow+xhigh)/2;
    mid_y = (ylow+yhigh)/2;
    
    local_A = zeros(4,4)
    local_M = copy(local_A)

    # mid-point of a, then exact quadrature for tent functions
    # f is written as sum of nodal values * tent functions
    for i = 1:4 
        for j = 1:4
            if i==j
                local_A[i,j] = 2/3*PDEparam.a(mid_x,mid_y);
                local_M[i,j] = 1/9/Ne^2 # exact
            elseif i==j+2 || i==j-2
                local_A[i,j] = -1/3*PDEparam.a(mid_x,mid_y);
                local_M[i,j] = 1/36/Ne^2
            else 
                local_A[i,j] = -1/6*PDEparam.a(mid_x,mid_y);
                local_M[i,j] = 1/18/Ne^2
            end
        end
    end

    return local_A, local_M
end


# domain [0,1]*[0,1], Ne is the number of interior DOF in each dimension
function FEM_Solver(FEMparam,PDEparam)
    # FEM for VarElliptic, uniform quadrilaternal mesh, so we only need one parameter Ne that is the number of interior DOF in each dimension, to fully characterize the mesh
    FEMmtx = FEM_StiffnMassAseembly(FEMparam,PDEparam)
    A, F = FEM_BdyRhsAssembly(FEMparam,PDEparam,FEMmtx)
    sol = A\F
    @info "[Linear solver] linear system solved"
    
    # result=reshape(sol,Ne+1,Ne+1)'
    # the ordering is that x moves first y second
    return sol, FEMmtx
end

## subsequent solve
function FEM_SubsequentSolve(FEMparam,PDEparam,FEMmtx)
    A, F = FEM_BdyRhsAssembly(FEMparam,PDEparam,FEMmtx)
    sol = A\F
    @info "[Linear solver] linear system solved"
    return sol
end