struct MsFEM_2ScaleUnifQuadMesh{Ti,Tf}
    Nc::Ti
    Nf::Ti # number of D.O.F in each dimension
    coarse_x::Vector{Tf}
    coarse_y::Vector{Tf}
    loc2glo::Function
    # all but the first attribute are automatically generated
end

# constructor
function MsFEM_2ScaleUnifQuadMesh(Nc,Nf)
    xx = collect(LinRange(0, 1, Nc+1))
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
    return MsFEM_2ScaleUnifQuadMesh(Nc,Nf,xx,yy,loc2glo)
end