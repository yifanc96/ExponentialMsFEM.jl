# PDEs
abstract type AbstractPDEs end

struct VarElliptic <: AbstractPDEs
    # domain is fixed [0,1]*[0,1]
    a::Function
    rhs::Function
    bdy::Function
    bdy_type::String
end