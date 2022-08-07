module SpacecraftDataset

export Spacecraft

struct Spacecraft 
    PosVector :: Matrix{Any}
    TruePos :: Matrix{Any}
    P :: Matrix{Any}
    P_xc :: Matrix{Any}
    P_xx :: Matrix{Any}
end
end