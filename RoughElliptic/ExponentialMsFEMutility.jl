function ExponentialMsFEM_GlobalAssembly(MsFEMparam, PDEparam, l)
    # N_c coarse D.O.F each dimension
    # N_f coarse D.O.F each dimension 
    # l: num of edge basis functions per edge  
    N_c = MsFEMparam.Nc
    N_f = MsFEMparam.Nf
    coarse_x = MsFEMparam.coarse_x
    coarse_y = MsFEMparam.coarse_y

    Nfinc = N_f+1;
    nNodes = (N_c+1)*(N_c+1)+2*N_c*(N_c+1)*N_basis;
    N_basis = l+1;

    # sparse assembling
    numb = (4+4*N_basis)^2*(N_c)^2;
    I = zeros(numb)
    J = copy(I)
    K = copy(I)
    F = zeros(nNodes);
    count = 4 + 4*N_basis;
    val = zeros(Nfinc^2,count,N_c^2);
    
    # there might be inner patches, patches on the edge, and nodal patches
    for i = 1:N_c
        for j = 1:N_c  
            [value,k, f] = MsFEM_LocalAssembly(PDEparam, coarse_x, coarse_y, i, j,N_f, N_basis);
            val[:,:,(i-1)*N_c+j] = value;
            for p = 1:count
                if p<5
                    global_p = PDEparam.loc2glo(N_c, i, j, p);
                elseif p<5+N_basis
                    global_p = (i-1+(j-1)*N_c)*N_basis+(N_c+1)^2+p-4;
                elseif p<5+2*N_basis
                    global_p = (i-1+(j)*N_c)*N_basis+(N_c+1)^2+p-4-N_basis;
                elseif p<5+3*N_basis
                    global_p = (i-1+(j-1)*(N_c+1)+N_c*(N_c+1))*N_basis+(N_c+1)^2+p-4-2*N_basis;
                else
                    global_p = (i+(j-1)*(N_c+1)+N_c*(N_c+1))*N_basis+(N_c+1)^2+p-4-3*N_basis;
                end
                for q = 1:count
                    if q<5
                        global_q = MsFEMparam.loc2glo(N_c, i, j, q);
                    elseif q<5+N_basis
                        global_q = (i-1+(j-1)*N_c)*N_basis+(N_c+1)^2+q-4;
                    elseif q<5+2*N_basis
                        global_q = (i-1+(j)*N_c)*N_basis+(N_c+1)^2+q-4-N_basis;
                    elseif q<5+3*N_basis
                        global_q = (i-1+(j-1)*(N_c+1)+N_c*(N_c+1))*N_basis+(N_c+1)^2+q-4-2*N_basis;
                    else
                        global_q = (i+(j-1)*(N_c+1)+N_c*(N_c+1))*N_basis+(N_c+1)^2+q-4-3*N_basis;
                    end
                    index = (4+4*N_basis)^2*(N_c)*(i-1)+(4+4*N_basis)^2*(j-1)+(4+4*N_basis)*(p-1)+q;
                    I[index] = global_p;
                    J[index] = global_q;
                    K[index] = k[p, q];
                end
                F[global_p] = F[global_p] + f[p];
            end
        end
    end
    
    b = zeros(4*(N_c)+4*N_c*N_basis);
    b[1:4*(N_c)+2*N_c*N_basis]=reduce(vcat,collect.([
        1:N_c+1,N_c+2:N_c+1:(N_c+1)*(N_c+1),2*(N_c+1):N_c+1:(N_c+1)*(N_c+1),
        N_c*N_c+N_c+2:(N_c+1)*(N_c+1)-1,(N_c+1)^2+1:(N_c+1)^2+N_basis*N_c,
        (N_c+1)^2+1+N_c^2*N_basis:(N_c+1)^2+N_basis*N_c+N_c^2*N_basis
        ]));

    for i=1:N_basis
        b[4*N_c+2*N_c*N_basis+1+(i-1)*2*N_c:4*N_c+2*N_c*N_basis+(i)*2*N_c]=
        reduce(vcat,collect.([i+N_c*(N_c+1)*N_basis+(N_c+1)^2:N_basis*(N_c+1):i+N_basis*(N_c+1)*(N_c-1)+N_c*(N_c+1)*N_basis+(N_c+1)^2,
            i+N_c*N_basis+N_c*(N_c+1)*N_basis+(N_c+1)^2:N_basis*(N_c+1):i+N_c*N_basis+N_basis*(N_c+1)*(N_c-1)+N_c*(N_c+1)*N_basis+(N_c+1)^2])); 
    end

    A = sparse(I,J,K,nNodes,nNodes);
    A[b,:] .= 0; A[:,b] .= 0; F[b] .= 0; 
    A[b,b] .= sparse(LinearAlgebra.I, length(b),length(b));
    
    return A, F
end

function MsFEM_LocalAssembly(PDEparam, X, Y, m, n, N_f, N_basis) 
    N_c = length(X)-1;

    xlow = coarse_x[i];
    xhigh = coarse_x[i+1];
    ylow = coarse_y[j];
    yhigh = coarse_y[j+1];
    x = collect(LinRange(xlow, xhigh, N_f+1))
    y = collect(LinRange(ylow, yhigh, N_f+1))

    b=reduce(vcat,collect.([1:N_f+1,N_f+2:N_f+1:(N_f+1)*(N_f+1),2*(N_f+1):N_f+1:(N_f+1)*(N_f+1),N_f*N_f+N_f+2:(N_f+1)*(N_f+1)-1]))

    f=collect.([LinRange(1, 0, N_f+1), LinRange(1-1/N_f, 0, N_f),zeros(1,N_f),zeros(1,N_f-1);
        LinRange(0, 1, N_f+1),zeros(1,N_f),LinRange(1-1/N_f, 0, N_f),zeros(1,N_f-1);
        zeros(1,N_f+1),zeros(1,N_f),LinRange(1/N_f, 1, N_f),LinRange(1/N_f, 1-1/N_f, N_f-1);
        zeros(1,N_f+1),LinRange(1/N_f, 1, N_f),zeros(1,N_f),LinRange(1-1/N_f, 1/N_f, N_f-1)]); # need to update

    A = basefun(MsFEMparam, PDEparam, X, Y, m, n, N_f);
    B = A;
    F = -A[:,b] * f';
    A[b,:] .= 0; A[:,b] .= 0; F[b,:] .= f'; 
    A[b,b] .= sparse(I, length(b),length(b));
    value = A\F;

    # edge basis
    count = 4;
    if n>1
        [L1,L2,N] = harmext(MsFEMparam, PDEparam, X, Y, m, n-1, N_f,1);
        [R,P,bub] = restrict(X, Y, m, n-1,N_f,1);
        [V,D] = eigs(R'*N*R,P,N_basis);
        value[:,count+1:count+N_basis] = L2*R*V;
        value[:,count+N_basis+1] = L2*bub;
    else
        value[:,count+1:count+N_basis+1] = 0;
    end
    count += N_basis +1;
    if n<N_c
        [L1,L2,N] = harmext(MsFEMparam, PDEparam, X, Y, m, n, N_f, 1);
        [R,P,bub] = restrict(X, Y, m, n, N_f, 1);
        [V,D] = eigs(R'*N*R,P,N_basis);
        value[:,count+1:count+N_basis] = L1*R*V;
        value[:,count+N_basis+1] = L1*bub;
    else
        value[:,count+1:count+N_basis+1] .= 0;
    end

    count += N_basis + 1;
    if m>1
        [L1,L2,N] = harmext(MsFEMparam, PDEparam, X, Y, m-1, n, N_f, 2);
        [R,P,bub] = restrict(X, Y, m-1, n, N_f, 2);
        [V,D] = eigs(R'*N*R,P,N_basis);
        value[:,count+1:count+N_basis] = L2*R*V;
        value[:,count+N_basis+1] = L2*bub;
    else
        value[:,count+1:count+N_basis+1] .= 0;
    end
    count += N_basis + 1;
    if m<N_c
        [L1,L2,N] = harmext(MsFEMparam, PDEparam, X, Y, m, n, N_f, 2);
        [R,P,bub] = restrict(X, Y, m, n, N_f, 2);
        [V,D] = eigs(R'*N*R,P,N_basis);
        value[:,count+1:count+N_basis] = L1*R*V;
        value[:,count+N_basis+1] = L1*bub;
    else
        value[:,count+1:count+N_basis+1] .= 0;
    end
    count += N_basis + 1;
    
    
    
    K=value'*B*value;
    # k and f separate using local stiffness matrix%%%%%%

    F = zeros(N_f^2,count);
    for u = 1:N_f
        for v = 1:N_f
            for x1=0:1
                for y1=0:1
                    xf=(x(u+1)+x(u))/2;
                    yf=(y(v+1)+y(v))/2;
                    F[v*N_f+u-N_f,:] = F[v*N_f+u-N_f,:] + PDEparam.rhs(xf,yf)*value[(v+y1-1)*(N_f+1)+u+x1,:]*(x(2)-x(1))^2/4;
                end
            end
        end
    end
    f = sum(F);
    return value, K, f
end
    

function ExponentialMsFEM_Solver(MsFEMparam,PDEparam,l)

    A, F = ExponentialMsFEM_GlobalAssembly(MsFEMparam, PDEparam, l)
    @info "[Assembly] finish assembly of stiffness matrice"
    u = A\F;

    @info "[Linear solver] linear system solved"
    result = zeros((N_c*N_f+1));
    
    for i = 1:N_c
        for j = 1:N_c

            xlow = coarse_x[i];
            xhigh = coarse_x[i+1];
            ylow = coarse_y[j];
            yhigh = coarse_y[j+1];
    
            xs = LinRange(xlow, xhigh, Nfinc);
            ys = LinRange(ylow, yhigh, Nfinc);
            zs = zeros(Nfinc, Nfinc);
            value = val[:,:,(i-1)*N_c+j];
            for p = 1:count
                if p<5
                    global_p = MsFEMparam.loc2glo(N_c, i, j, p);
                elseif p<5+N_basis
                    global_p = (i-1+(j-1)*N_c)*N_basis+(N_c+1)^2+p-4;
                elseif p<5+2*N_basis
                    global_p = (i-1+(j)*N_c)*N_basis+(N_c+1)^2+p-4-N_basis;
                elseif p<5+3*N_basis
                    global_p = (i-1+(j-1)*(N_c+1)+N_c*(N_c+1))*N_basis+(N_c+1)^2+p-4-2*N_basis;
                else
                    global_p = (i+(j-1)*(N_c+1)+N_c*(N_c+1))*N_basis+(N_c+1)^2+p-4-3*N_basis;
                end
                nodevalue = u[global_p];
                valuep = reshape(value[:,p],nSamples,nSamples);
                zs = zs + nodevalue* valuep;
            end
            # bubble part
            zs = zs + bubble(x,y,i,j,N_f);
            result[(i-1)*N_f+1:(i-1)*N_f+nSamples,(j-1)*N_f+1:(j-1)*N_f+Nfinc] = zs;
            # surf(xs, ys, zs'); # for plot the solution
        end
    end
    # result=reshape(result,(N_c*N_f+1)^2,1);
    return result
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
    
function restrict(X, Y, m, n, N_f, t)
    # %compute the matrix corresponding to the oversampled patch and its
    # %restriction operator on the edge, based on the stiffness matrix assembled
    # %in basefun1, and use local subproblem to solve for the bubble and harmonic
    # %parts.
    
    # %P is the inner product corresponding to the oversampled norm, R is the
    # %restriction operator
    fine = (X(2)-X(1))/N_f;
    N_c = length(X)-1;
    
    [A,N_x,N_y,x,y]=basefun1(MsFEMparam, PDEparam, X, Y, m, n,N_f,t);
    b=[1:N_x+1,N_x+2:N_x+1:(N_x+1)*(N_y+1),2*(N_x+1):N_x+1:(N_x+1)*(N_y+1),N_x*N_y+N_y+2:(N_x+1)*(N_y+1)-1];
    # %assemble boundary b
    
    # %solve the basis function corresponding to the harmonic part, whose dof
    # %corresponds to the boundary values
    
    # %pay attention to the case of boundary patches
    if t==1 && n==1 && 2<m<N_c-1 || t==2 && n<3 && 1<m<N_c-1
        f = zeros(2*(N_x+N_y),N_x+2*N_y-1);
        f[N_x+2:2*(N_x+N_y),:] .= sparse(LinearAlgebra.I, N_x+2*N_y-1, N_x+2*N_y-1)
    elseif t==1 && n==N_c-1 && 2<m<N_c-1 || t==2 && n>N_c-2 && 1<m<N_c-1
        f = zeros(2*(N_x+N_y),N_x+2*N_y-1);
        f[1:N_x+N_y,1:N_x+N_y] = sparse(LinearAlgebra.I, N_x+N_y, N_x+N_y);
        f[N_x+N_y+2:N_x+2*N_y,N_x+N_y+1:N_x+2*N_y-1] .= sparse(LinearAlgebra.I, N_y-1, N_y-1);
    elseif t==1 && m<3 && 1<n<N_c-1 || t==2 && m==1 && 2<n<N_c-1
        f = zeros(2*(N_x+N_y),2*N_x+N_y-1);
        f[2:N_x+1,1:N_x] .= sparse(LinearAlgebra.I, N_x, N_x);
        f[N_x+N_y+2:2*N_x+2*N_y,N_x+1:2*N_x+N_y-1] .= sparse(LinearAlgebra.I, N_x+N_y-1,N_x+N_y-1);
    elseif t==1 && m>N_c-2 && 1<n<N_c-1 || t==2 && m==N_c-1 && 2<n<N_c-1
        f = zeros(2*(N_x+N_y),2*N_x+N_y-1);
        f[1:N_x,1:N_x] .= sparse(LinearAlgebra.I, N_x, N_x);
        f[N_x+2:N_x+N_y+1,N_x+1:N_x+N_y] .= sparse(LinearAlgebra.I, N_y, N_y);
        f[N_x+2*N_y+2:2*(N_x+N_y),N_x+N_y+1:2*N_x+N_y-1] .= sparse(LinearAlgebra.I, N_x-1, N_x-1);
    elseif t==1 && n==1 && m<3 || t==2 && n<3 && m==1
        f = zeros(2*(N_x+N_y),N_x+N_y-1);
        f[N_x+N_y+2:2*(N_x+N_y),:] .= sparse(LinearAlgebra.I, N_x+N_y-1, N_x+N_y-1);
    elseif t==1 && n==1 && m>N_c-2 || t==2 && n<3 && m==N_c-1
        f = zeros(2*(N_x+N_y),N_x+N_y-1);
        f[N_x+2:N_x+N_y+1,1:N_y] .= sparse(LinearAlgebra.I, N_y, N_y);
        f[N_x+2*N_y+2:2*N_x+2*N_y,N_y+1:N_x+N_y-1] .= sparse(LinearAlgebra.I, N_x-1, N_x-1);
    elseif t==1 && n==N_c-1 && m<3 || t==2 && n>N_c-2 && m==1
        f = zeros(2*(N_x+N_y),N_x+N_y-1);
        f[2:N_x+1,1:N_x] .= sparse(LinearAlgebra.I, N_x, N_x);
        f[N_x+N_y+2:N_x+2*N_y,N_x+1:N_x+N_y-1] .= sparse(LinearAlgebra.I, N_y-1, N_y-1);
    elseif t==1 && n==N_c-1 && m>N_c-2 || t==2 && n>N_c-2 && m==N_c-1
        f = zeros(2*(N_x+N_y),N_x+N_y-1);
        f[1:N_x,1:N_x] .= sparse(LinearAlgebra.I, N_x, N_x);
        f[N_x+2:N_x+N_y,N_x+1:N_x+N_y-1] .= sparse(LinearAlgebra.I, N_y-1, N_y-1);
    else
        f = zeros(2*(N_x+N_y),2*(N_x+N_y)-1);
        f[1:2*(N_x+N_y)-1,:] .= sparse(LinearAlgebra.I, 2*(N_x+N_y)-1, 2*(N_x+N_y)-1);
    end
    
    
    B = A;
    F = -A(:,b)*f;
    B[b,:] .= 0; B[:,b] .= 0; F[b,:] = f;
    B[b,b] = sparse(LinearAlgebra.I, length(b),length(b))
    harm = B\F;
    
    # %solve the basis function corresponding to the bubble part, whose dof
    # %corresponds to the right hand side values
    
    # %assemble rhs
    G = zeros((N_x+1)*(N_y+1), 1);
    for i = 1:N_x
        for j = 1:N_y 
            for p=1:4
                xf=(x(i+1)+x(i))/2;
                yf=(y(j+1)+y(j))/2;
                G[MsFEMparam.loc2glo(N_x, i, j, p)]= G[MsFEMparam.loc2glo(N_x, i, j, p)] + PDEparam.rhs(xf,yf)*fine^2/4;
            end
        end
    end
    G[b] .= 0; 
    bub = B\G;
    
    
    # %compute inner product of energy
    # %P=fine^2*speyes(N_x*N_y+2*(N_x+N_y));
    # %P(2*(N_x+N_y),2*(N_x+N_y))=harm'*A*harm;
    P=harm'*A*harm;
    # %indentifying the edge
        leng=N_f+1;
    if t==1
        if m==1
            c=[(N_x+1)*N_f+1:(N_x+1)*N_f+N_f+1];
        else 
            c=[(N_x+1)*N_f+N_f+1:(N_x+1)*N_f+2*N_f+1];
        end
    else
        if n==1
            c=[N_f+1:N_x+1:(N_x+1)*N_f+N_f+1];
        else
            c=[(N_x+1)*N_f+N_f+1:N_x+1:(N_x+1)*N_f*2+N_f+1];
        end
    end
    
    # %intepolate
    R = harm[c,:];
    R = R-linspace(1,0,leng)'*R(1,:)-linspace(0,1,leng)'*R(leng,:);
    R = R[2:leng-1,:]
    bub = bub[c];
    bub = bub-linspace(1,0,leng)'*bub(1)-linspace(0,1,leng)'*bub(leng);
    bub = bub[2:leng-1];
    return R, P, bub
end

function bubble(X, Y, m, n,N_f)
    # %solve the bubble part
    x = linspace(X(m), X(m+1), N_f+1);
    y = linspace(Y(n), Y(n+1), N_f+1);
    nNodes = (N_f+1)*(N_f+1);
    # %sparse assembling
    I = zeros(16*N_f^2,1);J=I;K=I;
    F = zeros(nNodes, 1);
    
    for i = 1:N_f
        for j = 1:N_f  
            [k,f] = elementstiff2(x, y, i, j);
            for p = 1:4
                global_p = loc2glo(N_f, i, j, p);
                for q = 1:4
                    index=16*N_f*(i-1)+16*(j-1)+4*(p-1)+q;
                    global_q = loc2glo(N_f, i, j, q);
                    I[index] = global_p;
                    J[index] = global_q;
                    K[index] = k(p, q);
                end
                F[global_p] = F[global_p] + f[p];
            end
        end
    end
    
    # %assemble boundary
    
    A=sparse(I,J,K,nNodes,nNodes);  
    # %assemble boundary
    b=[1:N_f+1,N_f+2:N_f+1:(N_f+1)*(N_f+1),2*(N_f+1):N_f+1:(N_f+1)*(N_f+1),N_f*N_f+N_f+2:(N_f+1)*(N_f+1)-1,];
    
    A=sparse(I,J,K,nNodes,nNodes);
    
    A[b,:] .= 0; A[:,b] .= 0; F[b] .= 0; 
    A[b,b] = sparse(LinearAlgebra.I,length(b),length(b))
    u = A\F;
    re = reshape(u,N_f+1,N_f+1);
    
    return re
end