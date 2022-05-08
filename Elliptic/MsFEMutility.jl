# The target is to develop an efficient MsFEM for quadrilateral mesh, for research purposes. Here, the quadrilaternal mesh is axis parallel, so it is easy to describe its elements, boundary and neighboring geometry

using SparseArrays
using LinearAlgebra
using Logging
import Base.Threads: nthreads, @threads

# 2d 2scale mesh
# element easily described by x and y coordinates [Nce*Nce elements]
# element boundary easily described by a vector of boundary location (ordering: x increases first y second) [write a function/attribute]

### simple boundary treatment: just run over interior and boundaries separately (only for the boundary part use the if-else to select the correct b.c.). Each time run over it, construct the RHS and boundary data on the fly (not costly, as only once)

# basis function change when it is near to the boudary
# write a MsFEM that can support subsequent solve for the SAME boundary type!
# the key is the get a fine scale solver (surrogate model) for general problem; provide a coarse/fine scale solver for multiscale problems

struct MsFEM_2d2ScaleUnifQuadMesh{Ti,Tf}
    Nce::Ti # number of coarse elements in each dimension
    Nfe::Ti # number of fine elements in each coarse element
    Ne::Ti # number of total fine elements in each dimension
    CGrid_x::Vector{Tf} # coarse uniform grid in x axis, boundary included
    CGrid_y::Vector{Tf} # coarse uniform grid in y axis, boundary included
    ElemNode_loc2glo::Function # local index (node of an element) to global index (global indexed node); one runs over x first then y
    # all but the first attribute are automatically generated
    LocalBdyIndice::Vector{Ti}
    LocalBdyCondition::Matrix{Tf}
end

struct MsFEM_store{Ti,Tf}
    BasisFuns::Array{Tf, 4} # basisfuns in each element
    Fine_localAs::Matrix{SparseMatrixCSC{Tf,Ti}}
    Fine_localMs::Matrix{SparseMatrixCSC{Tf,Ti}}
    A::SparseMatrixCSC{Tf,Ti}
end

# constructor
function MsFEM_2d2ScaleUnifQuadMesh(Nce, Nfe)
    x = collect(LinRange(0, 1, Nce+1))
    y = copy(x)
    function ElemNode_loc2glo(N, i, j, ind_node)
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

    # local indices for local element
    LocalBdyIndice = reduce(vcat,collect.(
        [1:Nfe+1,
        Nfe+2:Nfe+1:(Nfe+1)*(Nfe+1),
        2*(Nfe+1):Nfe+1:(Nfe+1)*(Nfe+1),
        Nfe*Nfe+Nfe+2:(Nfe+1)*(Nfe+1)-1]
        )
    )

    # linear basis functions on the boundary (4 basis functions, equal 1 at one node and 0 at other nodes)
    LocalBdyCondition = reduce(vcat, collect.(
        [LinRange(1, 0, Nfe+1), LinRange(1-1/Nfe, 0, Nfe), zeros(Nfe), zeros(Nfe-1), 
        LinRange(0, 1, Nfe+1), zeros(Nfe), LinRange(1-1/Nfe, 0, Nfe), zeros(Nfe-1),
        zeros(Nfe+1), zeros(Nfe), LinRange(1/Nfe, 1, Nfe), LinRange(1/Nfe, 1-1/Nfe, Nfe-1),
        zeros(Nfe+1), LinRange(1/Nfe, 1, Nfe), zeros(Nfe), LinRange(1-1/Nfe, 1/Nfe, Nfe-1)]
        )
    )
    LocalBdyCondition = reshape(LocalBdyCondition, 4*Nfe, 4)

    @info "[Mesh generation] mesh generated, $(Nce+1) coarse nodes a in each dimension"
    @info "[Mesh generation] in each coarse element, $(Nfe+1) fine nodes; in total $(Nce*Nfe+1) nodes in each dimension"

    return MsFEM_2d2ScaleUnifQuadMesh(Nce,Nfe,Nce*Nfe,x,y,ElemNode_loc2glo, LocalBdyIndice, LocalBdyCondition)
end

function MsFEM_StiffnMassAssembly(MsFEMparam, PDEparam)
    # Nce num of coarse elements each dimension
    # Nfe num of fine elements in each coarse element
    Nce = MsFEMparam.Nce
    
    # sparse assembling
    Irow = zeros(16*(Nce)^2)
    Jcol = copy(Irow)
    Aval = copy(Irow)
    count = 4;
    LocalBasisFuns = zeros((Nfe+1)^2,count,Nce,Nce);
    Fine_localAs = Matrix{SparseMatrixCSC{Float64,Int}}(undef,Nce,Nce)
    Fine_localMs = copy(Fine_localAs)
    # there might be inner patches, patches on the edge, and nodal patches
    # run over coarse patches
    println("[multithreading] using ", Threads.nthreads(), " threads")
    @threads for ci = 1:Nce
        for cj = 1:Nce 
            coarse_localbasis, coarse_localA, fine_localA, fine_localM = MsFEM_LocalBasis(MsFEMparam, PDEparam, ci, cj);
            LocalBasisFuns[:,:,ci,cj] = coarse_localbasis; # store basis functions
            Fine_localAs[ci,cj] = fine_localA
            Fine_localMs[ci,cj] = fine_localM

            for p = 1:count
                global_p = MsFEMparam.ElemNode_loc2glo(Nce, ci, cj, p);
                for q = 1:count
                    global_q = MsFEMparam.ElemNode_loc2glo(Nce, ci, cj, q);
                    index = 16*(Nce)*(ci-1)+16*(cj-1)+(4)*(p-1)+q;
                    Irow[index] = global_p;
                    Jcol[index] = global_q;
                    Aval[index] = coarse_localA[p, q];
                end
            end
        end
    end
    A = sparse(Irow,Jcol,Aval,(Nce+1)^2,(Nce+1)^2)
    @info "[Assembly] finish assembly of stiffness matrice"
    
    return MsFEM_store(LocalBasisFuns,Fine_localAs,Fine_localMs,A)
end

function MsFEM_LocalBasis(MsFEMparam, PDEparam, ci, cj) 

    Nfe = MsFEMparam.Nfe

    # boundary location
    b = MsFEMparam.LocalBdyIndice
    # b = reduce(vcat,collect.(
    #     [1:Nfe+1,
    #     Nfe+2:Nfe+1:(Nfe+1)*(Nfe+1),
    #     2*(Nfe+1):Nfe+1:(Nfe+1)*(Nfe+1),
    #     Nfe*Nfe+Nfe+2:(Nfe+1)*(Nfe+1)-1]
    #     )
    # )

    # linear boundary conditions
    # vec_bd_f = reduce(vcat, collect.(
    #     [LinRange(1, 0, Nfe+1), LinRange(1-1/Nfe, 0, Nfe), zeros(Nfe), zeros(Nfe-1), 
    #     LinRange(0, 1, Nfe+1), zeros(Nfe), LinRange(1-1/Nfe, 0, Nfe), zeros(Nfe-1),
    #     zeros(Nfe+1), zeros(Nfe), LinRange(1/Nfe, 1, Nfe), LinRange(1/Nfe, 1-1/Nfe, Nfe-1),
    #     zeros(Nfe+1), LinRange(1/Nfe, 1, Nfe), zeros(Nfe), LinRange(1-1/Nfe, 1/Nfe, Nfe-1)]
    #     )
    # )
    # bd_f = reshape(vec_bd_f, 4*Nfe, 4) # 4 edges
    bd_f = MsFEMparam.LocalBdyCondition

    fine_localA, fine_localM = MsFEM_LocalHarmExt(MsFEMparam, PDEparam, ci, cj); # harmonic extension matrix
    Diri_A = copy(fine_localA); # local inner product A
    Diri_A[b,:] .= 0; 
    F = zeros((Nfe+1)^2, 4)
    F[b,:] = bd_f
    Diri_A[b,b] .= sparse(I, length(b),length(b));

    coarse_localbasis = Diri_A\F
    coarse_localA = coarse_localbasis'*fine_localA*coarse_localbasis; # energy inner product 

    return coarse_localbasis, coarse_localA, fine_localA, fine_localM
end

function MsFEM_LocalHarmExt(MsFEMparam, PDEparam, ci, cj)

    Nfe = MsFEMparam.Nfe

    xlow = MsFEMparam.CGrid_x[ci];
    xhigh = MsFEMparam.CGrid_x[ci+1];
    ylow = MsFEMparam.CGrid_y[cj];
    yhigh = MsFEMparam.CGrid_y[cj+1];
    x = collect(LinRange(xlow, xhigh, Nfe+1))
    y = collect(LinRange(ylow, yhigh, Nfe+1))

    # sparse assembling
    Irow = zeros(16*Nfe^2);
    Jcol = copy(Irow)
    Aval = copy(Irow)
    Mval = copy(Irow)
    
    for fi = 1:Nfe
        for fj = 1:Nfe
            local_A, local_M = MsFEM_LocalHarmExt_MatrixAssemby(MsFEMparam, PDEparam, x, y, fi, fj);
            for p = 1:4
                global_p = MsFEMparam.ElemNode_loc2glo(Nfe, fi, fj, p);
                for q = 1:4
                    index=16*Nfe*(fi-1)+16*(fj-1)+4*(p-1)+q;
                    global_q = MsFEMparam.ElemNode_loc2glo(Nfe, fi, fj, q);
                    Irow[index] = global_p
                    Jcol[index] = global_q
                    Aval[index] = local_A[p, q]
                    Mval[index] = local_M[p, q]
                end
            end
        end
    end
    A = sparse(Irow,Jcol,Aval,(Nfe+1)^2,(Nfe+1)^2)
    M = sparse(Irow,Jcol,Mval,(Nfe+1)^2,(Nfe+1)^2)
    return A, M
end
    
function MsFEM_LocalHarmExt_MatrixAssemby(MsFEMparam, PDEparam, x, y, i, j)
    
    Ne = MsFEMparam.Ne
    local_A = zeros(4,4)
    local_M = copy(local_A)

    xlow = x[i];
    xhigh = x[i+1];
    ylow = y[j];
    yhigh = y[j+1];
    xmid = (xlow+xhigh)/2;
    ymid = (ylow+yhigh)/2;
    
    for i = 1:4 
        for j = 1:4
            if i==j
                local_A[i,j] = 2/3*PDEparam.a(xmid,ymid);
                local_M[i,j] = 1/9/Ne^2 # exact
            elseif i==j+2 || i==j-2
                local_A[i,j] = -1/3*PDEparam.a(xmid,ymid);
                local_M[i,j] = 1/36/Ne^2
            else 
                local_A[i,j] = -1/6*PDEparam.a(xmid,ymid);
                local_M[i,j] = 1/18/Ne^2
            end
        end
    end
    return local_A, local_M
end

function MsFEM_BdyRhsAssembly(MsFEMparam, PDEparam, MsFEMstore)
    A = copy(MsFEMstore.A)
    
    BasisFuns = MsFEMstore.BasisFuns
    Fine_localAs = MsFEMstore.Fine_localAs
    Fine_localMs = MsFEMstore.Fine_localMs
    
    # bubble part
    Ne = MsFEMparam.Ne
    sol_bubble = zeros(Ne+1,Ne+1)


    ### right hand side
    Nce = MsFEMparam.Nce
    F = zeros((Nce+1)^2)
    count = 4

    # boundary location for bubble part
    b = MsFEMparam.LocalBdyIndice
    # b = reduce(vcat,collect.(
    #             [1:Nfe+1,
    #             Nfe+2:Nfe+1:(Nfe+1)*(Nfe+1),
    #             2*(Nfe+1):Nfe+1:(Nfe+1)*(Nfe+1),
    #             Nfe*Nfe+Nfe+2:(Nfe+1)*(Nfe+1)-1]
    #     )
    # )

    @threads for ci = 1:Nce
        for cj = 1:Nce 
            xlow = MsFEMparam.CGrid_x[ci];
            xhigh = MsFEMparam.CGrid_x[ci+1];
            ylow = MsFEMparam.CGrid_y[cj];
            yhigh = MsFEMparam.CGrid_y[cj+1];
            x = collect(LinRange(xlow, xhigh, Nfe+1))
            y = collect(LinRange(ylow, yhigh, Nfe+1))
            f = [PDEparam.rhs(x[i],y[j]) for j in 1:Nfe+1 for i in 1:Nfe+1]

            # assembly global rhs
            @views val = f'*Fine_localMs[ci,cj]*BasisFuns[:,:,ci,cj]
            for p = 1:count
                global_p = MsFEMparam.ElemNode_loc2glo(Nce, ci, cj, p);
                F[global_p] += val[p]
            end

            # assembly local rhs for bubble part
            Diri_A = copy(Fine_localAs[ci, cj]); 
            Diri_A[b,:] .= 0;
            Diri_A[b,b] .= sparse(I, length(b),length(b));
            local_F = Fine_localMs[ci,cj]*f
            local_F[b] .= 0

            @views sol_bubble[(ci-1)*Nfe+1:(ci)*Nfe+1, (cj-1)*Nfe+1:(cj)*Nfe+1] = Diri_A\(local_F)
        end
    end


    ### global boundary part
    # location
    b = reduce(vcat,collect.([
        1:Nce+1,Nce+2:Nce+1:(Nce+1)*(Nce+1),2*(Nce+1):Nce+1:(Nce+1)*(Nce+1),
        Nce*Nce+Nce+2:(Nce+1)*(Nce+1)-1,(Nce+1)^2+1:(Nce+1)^2
        ]));

    # assemby
    A[b,:] .= 0; 

    # for Dirichlet boundary condition
    F[1:Nce+1] .= PDEparam.bdy_Diri.(MsFEMparam.CGrid_x,0.0)
    F[Nce+2:Nce+1:(Nce+1)*(Nce+1)] .= PDEparam.bdy_Diri.(0.0, MsFEMparam.CGrid_y[2:end])
    F[2*(Nce+1):Nce+1:(Nce+1)*(Nce+1)] .= PDEparam.bdy_Diri.(1.0, MsFEMparam.CGrid_y[2:end])
    F[Nce*Nce+Nce+2:(Nce+1)*(Nce+1)-1] .= PDEparam.bdy_Diri.(MsFEMparam.CGrid_x[2:end-1],1.0)
    
    A[b,b] .= sparse(I, length(b),length(b));
    return A, F, sol_bubble

end

function MsFEM_CoarseSolver(MsFEMparam,PDEparam)
    MsFEMstore = MsFEM_StiffnMassAssembly(MsFEMparam,PDEparam)
    A, F, sol_bubble = MsFEM_BdyRhsAssembly(MsFEMparam,PDEparam,MsFEMstore)
    coarse_sol = A\F
    @info "[Linear solver] linear system solved, coarse sol obtained"

    return coarse_sol, sol_bubble, MsFEMstore
end

function MsFEM_FineConstruct(coarse_sol, sol_bubble, MsFEMstore)
    Nce = MsFEMparam.Nce
    Nfe = MsFEMparam.Nfe
    BasisFuns = MsFEMstore.BasisFuns
    fine_sol = zeros(Nce*Nfe+1,Nce*Nfe+1)
    count = 4

    @threads for ci in 1:Nce
        for cj in 1:Nce
            val = zeros(Nfe+1,Nfe+1)

            for p = 1:count
                global_p = MsFEMparam.ElemNode_loc2glo(Nce, ci, cj, p);
                val += coarse_sol[global_p]*reshape(BasisFuns[:,p,ci,cj],Nfe+1,Nfe+1)
            end
            # for p = 1:count
            #     global_p = MsFEMparam.ElemNode_loc2glo(Nce, ci, cj, p);
            #     fine_sol[(ci-1)*Nfe+1:(ci)*Nfe+1, (cj-1)*Nfe+1:(cj)*Nfe+1] += coarse_sol[global_p]*reshape(BasisFuns[:,p,ci,cj],Nfe+1,Nfe+1)
            # end
            fine_sol[(ci-1)*Nfe+1:(ci)*Nfe+1, (cj-1)*Nfe+1:(cj)*Nfe+1] = val
        end
    end

    # repeated counting
    # edges
    # fine_sol[1:end,Nfe+1:Nfe:end-Nfe] /= 2
    # fine_sol[Nfe+1:Nfe:end-Nfe,1:end] /= 2

    @info "[Reconstruction] fine scale solution reconstructed"
    return reshape(fine_sol+sol_bubble,(Nce*Nfe+1)^2)
    # return reshape(fine_sol,(Nce*Nfe+1)^2)
end

function MsFEM_SubsequentSolve(MsFEMparam,PDEparam,MsFEMstore)
    A, F, sol_bubble = MsFEM_BdyRhsAssembly(MsFEMparam,PDEparam,MsFEMstore)
    coarse_sol = A\F
    @info "[Linear solver] linear system solved, coarse sol obtained"
    return coarse_sol, sol_bubble, MsFEMstore
end