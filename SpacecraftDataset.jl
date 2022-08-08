module SpacecraftDataset

export Spacecraft

struct Spacecraft 
    PosVector :: Matrix{Any} #Pos+Vel [r0;v0;r1;v1;...] estimate
    TruePos :: Matrix{Any} # Pos+Vel [r0;v0;r1;v1;...] true values
    P :: Matrix{Any} # Covariance Matrix of the filter
    P_xc :: Matrix{Any} # Cross Correlations between chief and deputy filter
    P_xx :: Matrix{Any} # Chief Filter Covariance
end
end