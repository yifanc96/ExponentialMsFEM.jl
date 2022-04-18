# PDEs
abstract type AbstractPDEs end

mutable struct VarElliptic <: AbstractPDEs
    # domain is fixed [0,1]*[0,1]
    a::Function
    rhs::Function
    bdy_type::Function # 1 Diri, 2 Neum, 3 Robin
    bdy_Diri::Function
    bdy_Neum::Function
    bdy_Robin::Function
end