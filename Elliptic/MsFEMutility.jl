struct MsFEM_2d2ScaleUnifQuadMesh{Ti,Tf}
    Nce::Ti # number of coarse elements in each dimension
    Nfe::Ti # number of intervals in each coarse element
    CGrid_x::Vector{Tf} # coarse uniform grid in x axis, boundary included
    CGrid_y::Vector{Tf} # coarse uniform grid in y axis, boundary included
    ElemNode_loc2glo::Function # local index (node of an element) to global index (global indexed node); one runs over i first then j
    # all but the first attribute are automatically generated
end

# constructor
function MsFEM_2ScaleUnifQuadMesh(Nc, Nf)
    x = collect(LinRange(0, 1, Nc+1))
    y = copy(x)
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

    @info "[Mesh generation] mesh generated, $(Nce+1) coarsde nodes in each dimension"
    return MsFEM_2d2ScaleUnifQuadMesh(Nce,Nfe,x,y,ElementNode_loc2glo)
end


function MsFEM_GlobalAssembly(MsFEMparam, PDEparam)
    # Nc num of coarse elements each dimension
    # Nf num of fine elements in each coarse element
    d = 2
    Nc = MsFEMparam.Nc
    Nf = MsFEMparam.Nf

    nNodes = (Nc+1)^d
    
    # sparse assembling
    numb = (4)^2*(Nc)^2; # number of edges (repeated counting)
    Icol = zeros(numb)
    Jrow = copy(I)
    Aval = copy(I)
    F = zeros(nNodes);
    count = 4;
    val = zeros((Nf+1)^2,count,Nc^2);
    
    # there might be inner patches, patches on the edge, and nodal patches
    # run over coarse patches
    for i = 1:Nc
        for j = 1:Nc  
            [value,k, f] = MsFEM_LocalAssembly(MsFEMparam, PDEparam, i, j);
            val[:,:,(i-1)*Nc+j] = value; # store basis functions
            for p = 1:count
                global_p = MsFEMparam.ElementNode_loc2glo(Nc, i, j, p);
                for q = 1:count
                    global_q = MsFEMparam.ElementNode_loc2glo(Nc, i, j, q);
                    index = 16*(Nc)*(i-1)+16*(j-1)+(4)*(p-1)+q;
                    Icol[index] = global_p;
                    Jrow[index] = global_q;
                    Aval[index] = k[p, q];
                end
                F[global_p] = F[global_p] + f[p];
            end
        end
    end
    
    # location
    b = reduce(vcat,collect.([
        1:Nc+1,Nc+2:Nc+1:(Nc+1)*(Nc+1),2*(Nc+1):Nc+1:(Nc+1)*(Nc+1),
        Nc*Nc+Nc+2:(Nc+1)*(Nc+1)-1,(Nc+1)^2+1:(Nc+1)^2
        ]));

    # assemby
    if PDEparam.bdy_type == "Dirichlet"
        A[b,:] .= 0; 

        # for general Dirichlet boundary condition
        F[1:Nc+1] .= PDEparam.bdy.(FEMparam.coarse_x,0.0)
        F[Nc+2:Nc+1:(Nc+1)*(Nc+1)] .= PDEparam.bdy.(0.0, FEMparam.coarse_y[2:end])
        F[2*(Nc+1):Nc+1:(Nc+1)*(Nc+1)] .= PDEparam.bdy.(1.0, FEMparam.coarse_y[2:end])
        F[Nc*Nc+Nf+2:(Nc+1)*(Nc+1)-1] .= PDEparam.bdy.(FEMparam.coarse_x[2:end-1],1.0)
        
        A[b,b] .= sparse(I, length(b),length(b));
    else
        @info "other boundary condition not supported now"
    end

    @info "[Assembly] finish assembly of stiffness matrice"
    
    return A, F, val
end

function MsFEM_LocalAssembly(MsFEMparam, PDEparam, i, j) 

    Nf = MsFEMparam.Nf

    xlow = coarse_x[i];
    xhigh = coarse_x[i+1];
    ylow = coarse_y[j];
    yhigh = coarse_y[j+1];
    x = collect(LinRange(xlow, xhigh, Nf+1))
    y = collect(LinRange(ylow, yhigh, Nf+1))

    # boundary location
    b = reduce(vcat,collect.(
        [1:Nf+1,
        Nf+2:Nf+1:(Nf+1)*(Nf+1),
        2*(Nf+1):Nf+1:(Nf+1)*(Nf+1),
        Nf*Nf+Nf+2:(Nf+1)*(Nf+1)-1]
        )
    )

    # linear boundary conditions
    vec_bd_f = reduce(vcat, collect.(
        [LinRange(1, 0, Nf+1), LinRange(1-1/Nf, 0, Nf), zeros(1,Nf), zeros(1,Nf-1), 
        LinRange(0, 1, Nf+1), zeros(1,Nf), LinRange(1-1/Nf, 0, Nf), zeros(1,Nf-1),
        zeros(1,Nf+1), zeros(1,Nf), LinRange(1/Nf, 1, Nf), LinRange(1/Nf, 1-1/Nf, Nf-1),
        zeros(1,Nf+1), LinRange(1/Nf, 1, Nf), zeros(1,Nf), LinRange(1-1/Nf, 1/Nf, Nf-1)]
        )
    )
    bd_f = reshape(vec_bd_f, 4*Nf, 4) # 4 edges

    # f = collect.([LinRange(1, 0, Nf+1), LinRange(1-1/Nf, 0, Nf),zeros(1,Nf),zeros(1,Nf-1);
    #     LinRange(0, 1, Nf+1),zeros(1,Nf),LinRange(1-1/Nf, 0, Nf),zeros(1,Nf-1);
    #     zeros(1,Nf+1),zeros(1,Nf),LinRange(1/Nf, 1, Nf),LinRange(1/Nf, 1-1/Nf, Nf-1);
    #     zeros(1,Nf+1),LinRange(1/Nf, 1, Nf),zeros(1,Nf),LinRange(1-1/Nf, 1/Nf, Nf-1)]); # need to update

    A = basefun(MsFEMparam, PDEparam, X, Y, m, n, Nf); # harmonic extension matrix
    local_A = copy(A); # local inner product A
    A[b,:] .= 0; 
    F = zeros((Nf+1)^2, 4)
    F[b,:] = bd_f
    A[b,b] .= sparse(I, length(b),length(b));
    local_basis = A\F
    local_K = local_basis'*local_A*local_basis; # energy inner product 

    rhs_F = zeros(Nf^2,count);
    for u = 1:Nf
        for v = 1:Nf
            for x1=0:1
                for y1=0:1
                    xf=(x(u+1)+x(u))/2;
                    yf=(y(v+1)+y(v))/2;
                    F[v*Nf+u-Nf,:] = F[v*Nf+u-Nf,:] + PDEparam.rhs(xf,yf)*local_basis[(v+y1-1)*(Nf+1)+u+x1,:]*(x(2)-x(1))^2/4;
                end
            end
        end
    end
    f = sum(F);
    return local_basis, local_K, local_f
end

function basefun(MsFEMparam, PDEparam, X, Y, m, n, N_f)
    # % solve a combination of fine basis functions, assembling the stiffness
    # % matrix
    
    # % the function serves as to solve local boundary value problem on the 
    # % coarse mesh, based on the stiffness matrix in "elementstiff1"
    
    # % for the sake of simplicity, we temporarily use nodes of the fine mesh
    # % as the same nodes that we use when we invoke numerical quadrature to
    # % assemble the global stiffness matrix, the rationale being that we
    # % can either increase the scale of fine mesh or increase the quadrature 
    # % accuracy if we want to do higher order.... (because locally on fine
    # % meshes, the basis function is simply a linear function
    
    # % setting parameters H is the size of the coarse mesh, N is the number of
    # %nodes on each coarse rectangular, i.e. fine mesh size is H/N
    
    
    # %N needs to be inputted !!!!
    
    x = collect(LinRange(X(m), X(m+1), N_f+1));
    y = collect(LinRange(Y(n), Y(n+1), N_f+1));
    nNodes = (N_f+1)*(N_f+1);

    # sparse assembling
    I = zeros(16*N_f^2,1);
    J = copy(I)
    K = copy(I)
    
    for i = 1:N_f
        for j = 1:N_f  
            [k] = elementstiff1(PDEparam, x, y, i, j);
            for p = 1:4
                global_p = MsFEMparam.loc2glo(N_f, i, j, p);
                for q = 1:4
                    index=16*N_f*(i-1)+16*(j-1)+4*(p-1)+q;
                    global_q = MsFEMparam.loc2glo(N_f, i, j, q);
                    I[index] = global_p;
                    J[index] = global_q;
                    K[index] = k[p, q];
                end
            end
        end
    end
    # assemble boundary
    A = sparse(I,J,K,nNodes,nNodes);
    return A
end
    
function elementstiff1(PDEparam, X, Y, m, n)
    # %the stiffness matrix required locally to compute local nodal basis 
    # %for the bilinear boundary value with 1 at node "node"
    
    # %X Y are the fine mesh, we shall use local bilinear basis on the 
    # %fine mesh as basis to compute the harmonic problem for bilinear
    # %boundary value problem on the coarse mesh
    
    # %H is the size of the coarse mesh 
    
    
    # %H, nSamples and coefficients a needs to be inputted  !!!!!!!!!
    
    # %nSamples = 3; % sample points in trapz in each direction
    K = zeros(4, 4);
    xlow = X[m];
    xhigh = X[m+1];
    ylow = Y[n];
    yhigh = Y[n+1];
    
    x = (xlow+xhigh)/2;
    y = (ylow+yhigh)/2;
    
    for i = 1:4 
        for j = 1:4
            if i==j
                K[i, j] = 2/3*PDEparam.a(x,y);
            elseif i==j+2 || i==j-2
                    K[i,j] = -1/3*PDEparam.a(x,y);
            else 
                K[i,j] = -1/6*PDEparam.a(x,y);
            end
        end
    end
    return K
end
    
function harmext(MsFEMparam, PDEparam, X, Y, m, n, N_f, i)
    # %compute the local harmonic extension corresponding to each coarse edge
    # %i=1, corresponds to the horizontal edges, i=2, corresponds to the vertical
    # %ones. m, n are indices
    
    # %use dirichlet solver for adajacent patches, obtaining L1.L2 as a linear
    # %combination of the corresponding patches, N is matrix of the inner product
    
    b = reduce(vcat,collect.([1:N_f+1,N_f+2:N_f+1:(N_f+1)*(N_f+1),2*(N_f+1):N_f+1:(N_f+1)*(N_f+1),N_f*N_f+N_f+2:(N_f+1)*(N_f+1)-1]));
    K1 = basefun(MsFEMparam, PDEparam, X, Y, m, n, N_f);
    f1= zeros(4*N_f,N_f-1);
    f2= zeros(4*N_f,N_f-1);
    if i==1
        K2 = basefun(MsFEMparam, PDEparam, X, Y, m, n+1, N_f);
        f1[3*N_f+2:4*N_f,:] .= sparse(LinearAlgebra/I,N_f-1,N_f-1)
        f2[2:N_f,:] = sparse(LinearAlgebra.I,N_f-1,N_f-1)
    else
        K2 = basefun(MsFEMparam, PDEparam, X, Y, m+1, n, N_f);
        f1[2*N_f+2:3*N_f,:] .= sparse(LinearAlgebra.I,N_f-1,N_f-1)
        f2[N_f+2:2*N_f,:] .= sparse(LinearAlgebra.I,N_f-1,N_f-1)
    end
    M1 = K1;
    M2 = K2;
    F1 = -K1[:,b]*f1;
    K1[b,:] .=0; K1[:,b] .= 0; F1[b,:] = f1; 
    K1[b,b] = sparse(LinearAlgebra.I,length(b),length(b))
    L1 = K1\F1;
    F2 = -K2[:,b]*f2;
    K2[b,:] .= 0; K2[:,b] .= 0; F2[b,:] = f2; 
    K2[b,b] .= sparse(LinearAlgebra.I,length(b),length(b))
    L2=K2\F2;
    
    N1 = L1'*M1*L1;
    N2 = L2'*M2*L2;
    N = N1 + N2;
    return L1, L2, N
end
    
function basefun1(MsFEMparam, PDEparam, X, Y, m, n, N_f,t)
    # % solve a combination of fine basis functions, assembling the oversampled
    # % stiffness matrix
    # %t=1, corresponds to the horizontal edges, t=2, corresponds to the vertical
    # %ones. m, n are indices
    
    # % the function serves as to solve local boundary value problem on the 
    # % coarse mesh, based on the stiffness matrix in "elementstiff1"
    
    # % for the sake of simplicity, we temporarily use nodes of the fine mesh
    # % as the same nodes that we use when we invoke numerical quadrature to
    # % assemble the global stiffness matrix, the rationale being that we
    # % can either increase the scale of fine mesh or increase the quadrature 
    # % accuracy if we want to do higher order.... (because locally on fine
    # % meshes, the basis function is simply a linear function
    
    # % setting parameters H is the size of the coarse mesh, N is the number of
    # %nodes on each coarse rectangular, i.e. fine mesh size is H/N
    
    
    # %N needs to be inputted !!!!
    N_c = length(X)-1;
    if t==1
        if m==1
            x = linspace(X(m), X(m+2), 2*N_f+1);
        elseif m==N_c
            x = linspace(X(m-1), X(m+1), 2*N_f+1);
        else
            x = linspace(X(m-1), X(m+2), 3*N_f+1);
        end
        y = linspace(Y(n), Y(n+2), 2*N_f+1);
    else
        if n==1
            y = linspace(Y(n), Y(n+2), 2*N_f+1);
        elseif n==N_c
            y = linspace(Y(n-1), Y(n+1), 2*N_f+1);
        else
            y = linspace(Y(n-1), Y(n+2), 3*N_f+1);
        end
        x = linspace(X(m), X(m+2), 2*N_f+1);
    end
    
    nNodes = length(x)*length(y);
    N_x=length(x)-1;
    N_y=length(y)-1;
    # sparse assembling
    I=zeros(16*N_x*N_y,1);J=I;K=I;
    
    for i = 1:N_x
        for j = 1:N_y  
            [k] = elementstiff1(PDEparam, x, y, i, j);
            for p = 1:4
                global_p = MsFEMparam.loc2glo(N_x, i, j, p);
                for q = 1:4
                    index=16*N_y*(i-1)+16*(j-1)+4*(p-1)+q;
                    global_q = MsFEMparam.loc2glo(N_x, i, j, q);
                    I[index] = global_p;
                    J[index] = global_q;
                    K[index] = k[p, q];
                end
            end
        end
    end
    
    # assemble boundary
    
    A=sparse(I,J,K,nNodes,nNodes);
    
    return A, N_x, N_y, x, y
end