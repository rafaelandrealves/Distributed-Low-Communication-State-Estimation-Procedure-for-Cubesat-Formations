
# -- Event-Trigger Novel Distributed Low-Communication State Estimator for 4 Spacecraft Formation with included dynamics with RK4 integration
# -- Made by: Robotic Exploration Lab at Carnegie-Mellon University

# -------------------------------------------------------------------------
# -------------- Event-Trigger Distributed Considered EKF -----------------
# -------------------------------------------------------------------------
#

using LinearAlgebra, MATLAB, ForwardDiff, StaticArrays
using SparseArrays, IterativeSolvers, Infiltrator
const FD = ForwardDiff
using SatelliteDynamics
using Random
using BlockDiagonals
using Debugger




function workspace()
    atexit() do
        run(`$(Base.julia_cmd())`)
    end
    exit()
 end

####### Measurement Set for chief #######
function measurement_chief(x_chief)
    # Function that holds the chief measurement model, that includes only local GPS sensing 
    # Input arguments
    # - Pos+Vel [r0;v0;r1;v1;...]
    # - Pos+Vel [r0;v0;r1;v1;...] for chief
    # Output arguments
    # - Measurement Vector
    # From Robotic Exploration Lab at CMU 
    r1 = SVector(x_chief[1],x_chief[2],x_chief[3])

    
    # current measurement of Rel. Pos. 
    return [r1[1] r1[2] r1[3]]'
                   

end

####### Generate dataset for chief spacecraft #######
function generate_data_chief(x0,T,dt,R_chief)
    # Function that propagates the dynamics over T iterations 
    # Input arguments
    # - Pos+Vel [r0;v0;r1;v1;...]
    # - Number of Iterations
    # - Timestamp
    # - Measurement Noise Covariance
    # Output arguments
    # - Array of positions through time
    # - Measurements through time
    # From Robotic Exploration Lab at CMU 
    
    X = fill(zeros(6,1),T)
    Y = fill(zeros(3,1),T)


    X[1] = x0

    u = SVector(0,0,0)
    for i = 1:(T-1)
        t = (i-1)*dt
        aux,non = StateTransDeputiesRK4(dt,X[i])
        X[i+1] = aux #+ sqrt(Q)*randn(nx)
        aux = measurement_chief(X[i]) + sqrt(R_chief)*randn(size(R_chief,1),1)
        Y[i]=aux;

    end
    
    Y[T] = measurement_chief(X[T]) +  sqrt(R_chief)*randn(size(R_chief,1),1)

    return X,Y
end

####### Dynamics #######
function DynamicsJ2_MultSats(t,x,FLAG_Normalized)
    # Function that includes Normalized Multiple Satellite Dynamics w/
    # Jacobians for Two-Body+J2+Drag with exponential Model.
    # Input arguments
    # - Timestamp
    # - Pos+Vel [r0;v0;r1;v1;...]
    # - Flag on Normalization 
    # Output arguments
    # - xDot [v0;a0;...]
    # - Dot Jacobians
    # Original for Robotic Exploration Lab at Carnegie Mellon University

    ## A is dx = f = [v;ac]; A = delta_f
    # Since d_phi(t,t0) = delta_f/delta_x * phi(t,t0)

    R_E = 6378.137; #km

    nx=size(x,1);
    nt= size(x,2)
    xd=zeros(nx,nt); A = [];
    
    ## Two-body dynamics & drag
    ## Chapter 7.80 Montenbruck book
    mu = 3.986E5; raw0 = 1.225;# kg/m^3
    w_E = [0; 0; 2*pi/86184]; # Earths angular velocity vector -> Rel velocity of Satellite vs the athmospheric, assume atmosphere co-rotates with Earth -> vr =

    Area = 0.01;
    m = 1;
    C_d = 2.22;
    H0 = 7.9; # km, characteristic height
    #J2 oblateness term
	J2=0.0010826359;


    A = zeros(nx,nx);
    for i = 1:(nx/6)
        i = round(Int,i)
        r = x[((i-1)*6+1):((i-1)*6+3)]; #km
        v = x[((i-1)*6+4):((i-1)*6+6)]; #km/s 
        
        #J2 oblateness term
        J2=0.0010826359;


        ## Drag Model - Based on Montenbruck and Julia SatelliteToolbox.jl Package

        

        # Compute the atmospheric density [kg/m³] at the altitude `h` \\[m] (above the
        # ellipsoid) using the exponential atmospheric model:
        #                     ┌            ┐
        #                     │    h - h₀  │
        #     ρ(h) = ρ₀ ⋅ exp │ - ──────── │ ,
        #                     │      H     │
        #                     └            ┘
        # in which `ρ₀`, `h₀`, and `H` are parameters obtained from tables that depend
        # only on `h`.    

        """{
            _expatmosphere_h₀
        Base altitude for the exponential atmospheric model [km].
        }
        """
        expatmosphere_h0 = [0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                               130, 140, 150, 180, 200, 250, 300, 350, 400, 450,
                               500, 600, 700, 800, 900, 1000];
        """
        #{
            _expatmosphere_ρ₀
        Nominal density for the exponential atmospheric model [kg/m³].
        #}
        """
         expatmosphere_Raw0 = [1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3,
                                   3.206e-4, 8.770e-5, 1.905e-5, 3.396e-6, 5.297e-7,
                                   9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9, 2.070e-9,
                                   5.464e-10, 2.789e-10, 7.248e-11, 2.418e-11,
                                   9.518e-12, 3.725e-12, 1.585e-12, 6.967e-13,
                                   1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15,
                                   3.019e-15];

        """#{

            _expatmosphere_H
        Scale height for the exponential atmospheric model [km].
        #}"""

         expatmosphere_H = [7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549,
                                  5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149,
                                  22.523, 29.740, 37.105, 45.546, 53.628, 53.298,
                                  58.515, 60.828, 63.822, 71.835, 88.667, 124.64,
                                  181.05, 268.00];

        h = (norm(r) - R_E);
        id = [];
        id = (h >= 1000) ? 28 : findfirst( (x)->x > 0 , expatmosphere_h0 .- h ) - 1;

        if h >= 1000
           id[1] = 28; 
        end
        
        h0 = expatmosphere_h0[id[1] - 1];
        raw0 = expatmosphere_Raw0[id[1] - 1];
        H0 = expatmosphere_H[id[1] - 1];

        deltaRaw_x = -(raw0/H0)*((r[1]*exp(-(h - h0)/H0))/(norm(r)));
        deltaRaw_y = -(raw0/H0)*((r[2]*exp(-(h - h0)/H0))/(norm(r)));
        deltaRaw_z = -(raw0/H0)*((r[3]*exp(-(h - h0)/H0))/(norm(r)));
        deltaRaw = [deltaRaw_x deltaRaw_y deltaRaw_z];#kg/m^3


        Raw = raw0*exp(-(h - h0)/H0);# Density at height, kg/m^3

        v_rel = (v-cross(w_E,r));# km/s

        delta_acc_deltaV = -0.5*C_d*(m/Area)*Raw*((v_rel*v_rel')./norm(v_rel)^2 + I(3)*norm(v_rel)^2);

        A[((i-1)*6+4):((i-1)*6+6),((i-1)*6+1):((i-1)*6+3)] = A[((i-1)*6+4):((i-1)*6+6),((i-1)*6+1):((i-1)*6+3)] + -0.5*C_d*(m/Area)*norm(v_rel)^2*(v_rel./norm(v_rel))*deltaRaw 
            -delta_acc_deltaV*([0 -7.292E-5 0; 7.292E-5 0 0; 0 0 0]);

        ## J2

        A[((i-1)*6+1):((i-1)*6+3),((i-1)*6+4):((i-1)*6+6)] = I(3);

        A_E = 0.5*J2*R_E^2;
        J2_x = -((r[2]^2 + r[3]^2 - 2*r[1]^2)/(norm(r)^5)) + ((15*A_E*(r[2]^2 + r[3]^2 - 6*r[1]^2)*r[3]^2)/(norm(r)^9)) 
            - ((3*(r[2]^2 + r[3]^2 - 4*r[1]^2))/(norm(r)^7));

        J2_x_y =  ((3*r[2]*r[1])/(norm(r)^5)) - ((105*A_E*r[2]*r[1]*r[3]^2)/(norm(r)^9)) + ((15*r[2]*r[1])/(norm(r)^7));

        J2_x_z = ((3*r[3]*r[1])/(norm(r)^5)) - ((15*A_E*r[3]*(-5*r[3]^2 + 2*r[1]^2 + 2*r[2]^2)*r[1])/(norm(r)^9)) + ((15*r[3]*r[1])/(norm(r)^7));


        J2_y_z =  ((3*r[3]*r[2])/(norm(r)^5)) - ((15*A_E*r[3]*(-5*r[3]^2 + 2*r[2]^2 + 2*r[1]^2)*r[2])/(norm(r)^9)) + ((15*r[3]*r[2])/(norm(r)^7));

        J2_y =  -((r[1]^2 + r[3]^2 - 2*r[2]^2)/(norm(r)^5)) + ((15*A_E*(r[1]^2 + r[3]^2 - 6*r[2]^2)*r[3]^2)/(norm(r)^9)) - ((3*(r[1]^2 + r[3]^2 - 4*r[2]^2))/(norm(r)^7));             

        J2_y_x = ((3*r[2]*r[1])/(norm(r)^5)) - ((105*A_E*r[2]*r[1]*r[3]^2)/(norm(r)^9)) + ((15*r[2]*r[1])/(norm(r)^7));


        J2_z = -((r[1]^2 + r[2]^2 - 2*r[3]^2)/(norm(r)^5)) + ((15*A_E*(-4*r[3]^2 + 3*r[1]^2 + 3*r[2]^2)*r[3]^2)/(norm(r)^9)) - ((9*(r[1]^2 + r[2]^2 - 4*r[3]^2))/(norm(r)^7));

        J2_z_x = ((3*r[3]*r[1])/(norm(r)^5)) - ((105*A_E*r[1]*r[3]^3)/(norm(r)^9)) + ((45*r[3]*r[1])/(norm(r)^7));

        J2_z_y = ((3*r[3]*r[2])/(norm(r)^5)) - ((105*A_E*r[2]*r[3]^3)/(norm(r)^9)) + ((45*r[3]*r[2])/(norm(r)^7));

        A[((i-1)*6+4):((i-1)*6+6),((i-1)*6+1):((i-1)*6+3)] = A[((i-1)*6+4):((i-1)*6+6),((i-1)*6+1):((i-1)*6+3)] + mu*[J2_x J2_x_y J2_x_z;J2_y_x J2_y J2_y_z;J2_z_x J2_z_y J2_z];

        ## Accelerations

        f_drag  = -0.5*(C_d*(m/Area))*Raw*norm(v_rel)^2*(v_rel/norm(v_rel)); #km/s^2


        f_J2 = mu*[ (-r[1]/(norm(r)^3) + (0.5*J2*R_E^2)*(15*r[1]*r[3]^2/norm(r)^7 
            - 3*r[1]/norm(r)^5));
                 (-r[2]/(norm(r)^3) + (0.5*J2*R_E^2)*(15*r[2]*r[3]^2/norm(r)^7 
            - 3*r[2]/norm(r)^5));
                 (-r[3]/(norm(r)^3) + (0.5*J2*R_E^2)*(15*r[3]*r[3]^2/norm(r)^7 
            - 9*r[3]/norm(r)^5));
            ]; #km/s^2

        xd[(i-1)*6+1:(i-1)*6+6] = [v;(f_J2+f_drag)];
    end  


    
    
    return xd,A
    
end

####### Perform RK4 Integration #######
function StateTransDeputiesRK4(t,x)
    
    ## Normal Rk4 and addition of jacobian determination

    x_dot1, A1 = DynamicsJ2_MultSats(t,x,0);
 
    x_dot2, A2 = DynamicsJ2_MultSats(t,x + .5*t*x_dot1,0);
    
    x_dot3, A3 = DynamicsJ2_MultSats(t,x + .5*t*x_dot2,0);
    
    x_dot4, A4 = DynamicsJ2_MultSats(t,x + t*x_dot3,0);
 
    x_new = x + (1/6) * t * (x_dot1 + 2 * x_dot2 + 2 * x_dot3 + x_dot4); 

    dk1_dx1 = t*A1;
    dx2_dx1 = I(size(A1,1)) + .5*dk1_dx1;
    dk2_dx1 = t*A2*dx2_dx1;
    dx3_dx1 = I(size(A1,1)) + .5*dk2_dx1; 
    dk3_dx1 = t*A3*dx3_dx1;
    dx4_dx1 = I(size(A1,1)) + dk3_dx1;
    dk4_dx1 = t*A4*dx4_dx1; 
    A_d = I(size(A1,1)) + (1/6)*(dk1_dx1 + 2*dk2_dx1 + 2*dk3_dx1 + dk4_dx1);

    return x_new,A_d
end

####### Measurement Set for deputies #######
function measurement_deputies(x,x_chief)
    # Function that holds the deputy measurement model, that inclues only relative-range for the entire formation 
    # Input arguments
    # - Pos+Vel [r0;v0;r1;v1;...]
    # - Pos+Vel [r0;v0;r1;v1;...] for chief
    # Output arguments
    # - Measurement Vector
    # From Robotic Exploration Lab at CMU 

    r1 = SVector(x_chief[1],x_chief[2],x_chief[3])

    
    r2 = SVector(x[1],x[2],x[3])

    r3 = SVector(x[7],x[8],x[9])
    
    r4 = SVector(x[13],x[14],x[15])


    # current measurement of Rel. Pos. 
    return [norm(r2-r1);
                   norm(r2-r3);
                   norm(r2-r4);
                   norm(r3-r1);
                   norm(r3-r4);
                   norm(r4-r1);;]
                   

end

####### Generate Dataset #######

function generate_data_deputies(x0,X_chief,T,dt,R)
    # Function that propagates the dynamics over T iterations 
    # Input arguments
    # - Pos+Vel [r0;v0;r1;v1;...]
    # - Number of Iterations
    # - Timestamp
    # - Measurement Noise Covariance
    # Output arguments
    # - Array of positions through time
    # - Measurements through time
    # From Robotic Exploration Lab at CMU 
    
    X = fill(zeros(18,1),T)
    Y = fill(zeros(size(R,1),1),T)


    X[1] = x0

    u = SVector(0,0,0)
    for i = 1:(T-1)
        t = (i-1)*dt
        aux,non = StateTransDeputiesRK4(dt,X[i])
        X[i+1] = aux #+ sqrt(Q)*randn(nx)
        aux = measurement_deputies(X[i],X_chief[i]) + sqrt(R)*randn(size(R,1),1)
        Y[i] = aux;
    end
    
    Y[T] = measurement_deputies(X[T],X_chief[T]) +  sqrt(R)*randn(size(R,1),1)

    return X,Y
end

####### Chief Filter Architecture #######

function ChiefFilterEKFFunctionwCRLB(x_old,y1,P_old,T,R,Q,Info_Jk_chief,FLAG_GPS)
    # Function corresponding to the Chief Filter architecture
    # Input arguments
    # - Previous Pos+Vel [r0;v0;r1;v1;...]
    # - Measurements - Abs. Inertial Position
    # - Previous Covariance Matrix
    # - Timestamp
    # - Process Noise Covariance Matrix
    # - Measurement Noise Covariance Matrix
    # - Information Matrix
    # - GPS Flag for Sensor Selection
    # Output arguments
    # - New estimate Pos+Vel [r0;v0;r1;v1;...]
    # - New Covariance Matrix
    # - Jacobians
    # - New Information Matrix

    ############# Filter Processing #############
    x_pos_old = x_old;
    P = P_old;

    # V = sqrt(Q)*randn(3,1); #Observation Noise

    # Dynamics Propagations
    
    x_new,phi = StateTransDeputiesRK4(T, x_pos_old);

    
    x_pos = x_new; 
    phi_t = phi;
    
    P = phi_t*P*phi_t'+  Q;

    # Measurement Processing

    H = hcat(I(3),zeros(3,3));

    Info_Jk_chief = inv(Q + phi_t*inv(Info_Jk_chief)*(phi_t)')+ H'*inv(R)*H;

    ############# State Refining #############
    if FLAG_GPS == 1

        y_hat = [x_pos[1]; 
                    x_pos[2]; 
                    x_pos[3];;]; 


        y = (y1-y_hat) ;
        Kgain = P*H'/(H*P*H' + R);

        x_new =  x_pos + Kgain*y;
        P = P - Kgain*H*P;

    end
    
    return x_new,P,phi,Info_Jk_chief
end         
       
####### Deputy Filter Architecture #######

function DeputyFilterEKFFunctionwCRLB(x_old,x_new_chief,P_chief,y_true,P,P_xc,P_xx,InfoJ_k,T,R,Q,FLAG_RR)
    # Function corresponding to the Chief Filter architecture
    # Input arguments
    # - Previous Pos+Vel [r0;v0;r1;v1;...]
    # - Chief Pos+Vel [r0;v0;r1;v1;...]
    # - Chief Covariance Matrix
    # - Measurements - Relative-Range
    # - Past Covariance Matrix
    # - Cross-Correlations Matrix between chief and deputy
    # - Past Information Matrix
    # - Timestamp
    # - Process Noise Covariance Matrix
    # - Measurement Noise Covariance Matrix
    # - Rel. Range Flag for Sensor Selection
    # Output arguments
    # - New estimate Pos+Vel [r0;v0;r1;v1;...]
    # - New Covariance Matrix
    # - New Cross-Correlations Matrix between chief and deputy
    # - New Naive Covariance Matrix
    # - Information Matrix for determiantion of Crámer-Rao Lower Bound

    # From Robotic Exploration Lab at CMU 
    ############# Filter Processing #############

    x_new,phi = StateTransDeputiesRK4(T,x_old);


    P_cc = P_chief;

        
    x_pos = x_new;
    phi_t = phi;


    P = phi_t*P*phi_t'+  Q;
    P_xx = phi_t*P_xx*phi_t'+  Q;

    Fc = zeros(18,6);

    P = P + phi_t*P_xc*Fc' + Fc*P_xc'*phi_t' + Fc*P_cc*Fc';
    P_xc = phi_t*P_xc + Fc*P_cc;

    ## Meas. Prediction 
    H = zeros(Number_meas,6*(Number_of_Sats-1)); row_count = 1;
    if FLAG_RR == 1
        y_hat = zeros(Number_meas,1);
        for meas_processing_sat = 1:(Number_of_Sats-1)

            # First Range w/GPS Sats

            y_hat[row_count] = norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-x_new_chief[1:3,:]);

            H[row_count,(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+6] =  [((x_pos[(meas_processing_sat-1)*6+1]-(x_new_chief[1]))/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_new_chief[1:3,:]))) 
                    ((x_pos[(meas_processing_sat-1)*6+2]-x_new_chief[2])/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_new_chief[1:3,:]))) 
                    ((x_pos[(meas_processing_sat-1)*6+3]-x_new_chief[3])/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_new_chief[1:3,:])))
                    0 
                    0
                    0]; 
            row_count = row_count+1;
            for h = (meas_processing_sat+1):(Number_of_Sats-1)

                y_hat[row_count] = norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_pos[(h-1)*6+1:(h-1)*6+3,:]));


                # Measurement Processing

                H[row_count,(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+6] = [((x_pos[(meas_processing_sat-1)*6+1] - x_pos[(h-1)*6+1])/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_pos[(h-1)*6+1:(h-1)*6+3,:]))) 
                    ((x_pos[(meas_processing_sat-1)*6+2] - x_pos[(h-1)*6+2])/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_pos[(h-1)*6+1:(h-1)*6+3,:])))
                    ((x_pos[(meas_processing_sat-1)*6+3] - x_pos[(h-1)*6+3])/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_pos[(h-1)*6+1:(h-1)*6+3,:])))
                    0
                    0
                    0];


                H[row_count,(h-1)*6+1:(h-1)*6+6] =  -1*[((x_pos[(meas_processing_sat-1)*6+1] - x_pos[(h-1)*6+1])/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_pos[(h-1)*6+1:(h-1)*6+3,:]))) 
                    ((x_pos[(meas_processing_sat-1)*6+2] - x_pos[(h-1)*6+2])/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_pos[(h-1)*6+1:(h-1)*6+3,:])))
                    ((x_pos[(meas_processing_sat-1)*6+3] - x_pos[(h-1)*6+3])/norm(x_pos[(meas_processing_sat-1)*6+1:(meas_processing_sat-1)*6+3,:]-(x_pos[(h-1)*6+1:(h-1)*6+3,:])))
                    0
                    0
                    0];                  
                row_count = row_count +1;
            end



        end

        # y1 = Y[i];

        ############# State Refining #############
        # Consider Parameters covariance matrix, includes the influence of the chief state on the deputy measurement model
        H_c = zeros(row_count-1,6); 
        index = [1 4 6];# GPS Meas
        for g = 1:Number_of_Sats-1
            H_c[index[g],:] =  -1*[((x_pos[(g-1)*6+1]-(x_new_chief[1]))/norm(x_pos[(g-1)*6+1:(g-1)*6+3,:]-(x_new_chief[1:3,:]))) 
                ((x_pos[(g-1)*6+2]-x_new_chief[2])/norm(x_pos[(g-1)*6+1:(g-1)*6+3]-(x_new_chief[1:3,:]))) 
                ((x_pos[(g-1)*6+3]-x_new_chief[3])/norm(x_pos[(g-1)*6+1:(g-1)*6+3]-(x_new_chief[1:3,:])))
                0
                0
                0]; 
        end

        y = y_true-y_hat;


        P_xy = P*H' + P_xc*H_c';
        P_yy =   H*P*H' + R + H*P_xc*H_c' + H_c*P_xc'*H' + H_c*P_cc*H_c';

        # Calculate the Kalman gain.
        K = P_xy / P_yy;
        aux =  x_pos + K*y;

        x_pos = aux;

        # Correct the state covariance and the state-consider covariance.
        P = P - K*H*P - K*H_c*P_xc'; # New part

        Kgain = P_xx*H'/(H*P_xx*H' + R);
        P_xx = P_xx - Kgain*H*P_xx;

        P_xc = P_xc - K*H*P_xc - K*H_c*P_cc;
    end

    ############ Crámer-Rao Lower-bound ############

    InfoJ_k = inv(Q + phi_t*inv(InfoJ_k)*(phi_t)')+ H'*inv(R)*H;

    return x_pos,P,P_xc,P_xx,InfoJ_k

end         
       

####### Data Struct Defining Formation Objects #######

include("SpacecraftDatasetETC.jl")
using .SpacecraftDatasetETC

# ----------------------------------------------------------
## Main for Event-Trigger Distributed Consider EKF - Contains deputy Architecture
# ----------------------------------------------------------

# V-R3x Mission Setting
x0 = permutedims([6895.6 0 0 0 -0.99164 7.5424 6895.6 3e-05 1e-05 -0.0015 -0.99214 7.5426 6895.6 1e-05 3e-06 0.005 -0.98964 7.5422 6895.6 -2e-05 4e-06 0.00545 -0.99594 7.5423])

# Optimized Formation Mission Setting
#x0 = permutedims([-1295.9 -929.67 6793.4 -7.4264 0.1903 -1.3906 171.77 6857.3 -1281.5 1.0068 -1.3998 -7.3585 -4300.5 1654.8 -5241 -4.6264 3.4437 4.8838 -2197.8 -3973.4 -5298.7 1.377 5.6601 -4.8155])


Number_of_Sats = 4;


# Covariance Matrixes Parameters
xabs = 10^(-2);
vabs = 10^(-6);
r =  1*10^(-3);v = 10^(-5); q = 1.07*10^(-3);
q_chief= 1.07*10^(-4);



P_sat = [xabs^2 0 0 0 0 0; 0 xabs^2 0 0 0 0;0 0 xabs^2 0 0 0;0 0 0 vabs^2 0 0;0 0 0 0 vabs^2 0;0 0 0 0 0 vabs^2];

Factorization_mea = cumsum(1:(Number_of_Sats-2));

Number_meas = (Number_of_Sats-1) + Factorization_mea[end];

r_chief =  1*10^(-4);v_chief = 10^(-5);

R = Matrix( (q^2)I, Number_meas, Number_meas);
R_chief = Matrix( (q_chief^2)I, 3, 3)

a = Diagonal([r*ones(3); v*ones(3)] .^ 2)
Q = Matrix(BlockDiagonal([a,a,a]))
Q_chief = Diagonal([r_chief*ones(3); v_chief*ones(3)] .^ 2)

GM = 398600.4418;


# sample time is 60 seconds
dt = 60.0

# initial time for sim
T = 395
SizeOfDataSet = 395
# T = 15
# SizeOfDataSet = 15
ElapSEC = 0:dt:dt*(SizeOfDataSet)

# Generate Datasets from an initial starting state for the formation

X_chief,Y_chief = generate_data_chief(x0[1:6,:],T,dt,R_chief)

X,Y = generate_data_deputies(x0[7:end,:],X_chief,T,dt,R)


# Gaussian Initial Deviation


x_pos_init = x0; true_pos = x0;
Initial_Dev = [0.1*(randn(3,1));10^(-5)*(randn(3,1));0.1*(randn(3,1));10^(-5)*(randn(3,1));0.1*(randn(3,1));10^(-5)*(randn(3,1));0.1*(randn(3,1));10^(-5)*(randn(3,1));;];

x_pos_init = x_pos_init + Initial_Dev;
Pos_init = x_pos_init[7:end,:]; Pos_init_chief = x_pos_init[1:6,:];
Pos_True = true_pos[7:end,:]; Pos_True_chief = true_pos[1:6,:];



P_Init_aux = Matrix(BlockDiagonal([P_sat,P_sat,P_sat]));

PCov = 2*Diagonal(vec((Initial_Dev[7:end,:]).^2)) + P_Init_aux; 

PCov_chief = Diagonal(vec((Initial_Dev[1:6,:]).^2)) + P_sat;

P_xc = zeros(18,6)

# Initialize Formation data structure that holds the navigation related information

SpacecraftChief = SpacecraftETC(Pos_init_chief,Pos_True_chief,PCov_chief,zeros(18,6),zeros(18,18),inv(Q_chief));
SpacecraftDep1 = SpacecraftETC(Pos_init,Pos_True,PCov,P_xc,PCov,inv(Q));
SpacecraftDep2 = SpacecraftETC(Pos_init,Pos_True,PCov,P_xc,PCov,inv(Q));
SpacecraftDep3 = SpacecraftETC(Pos_init,Pos_True,PCov,P_xc,PCov,inv(Q));

Formation = [SpacecraftChief,SpacecraftDep1,SpacecraftDep2,SpacecraftDep3]

xteo = true_pos[7:end,:];
P_cc = Diagonal(vec((Initial_Dev[1:6,:]).^2)) + P_sat;

# Event-Triggering Init. 

Trace_P_consider = zeros(SizeOfDataSet,1);
Trace_P =  zeros(SizeOfDataSet,1);
Trace_P_dep =  zeros(SizeOfDataSet,1);   


InfoJ_k = inv(Q);


Trace_CramerRaoLB= zeros(SizeOfDataSet,1);
Trace_CramerRaoLB_Chief= zeros(SizeOfDataSet,1);


# Confidence-Levels for Sensor Selection
CL_Chief = 5
CL_Deputy = 1.5

for i = 2:SizeOfDataSet


    ############# Filter Processing #############
        T = ElapSEC[i]-ElapSEC[i-1];

        x_pos_old = Formation[2,i-1].PosVector;
        x_pos_old_chief = Formation[1,i-1].PosVector;

        P = Formation[2,i-1].P;
        P_xc = Formation[2,i-1].P_xc;

        P_chief = Formation[1,i-1].P;
        W = sqrt(Q)*randn(6*(Number_of_Sats-1),1); #Process Noise
        V = sqrt(R)*randn(Number_meas,1); #Observation Noise


        global X[i] = X[i] + W;
        global X_chief[i]  = X_chief[i] + sqrt(Q_chief)*randn(6,1);

        # Dynamics Propagations - Event-Trigger Conditions for Chief - Due to the initial poorly-observable scenario for the V-R3x mission
        # force the filter to also use GPS in the first 10 iterations.
        if any(x->x==i,[1:1:10;]) || tr(convert(Matrix{Float64},Formation[1,i-1].P)) > CL_Chief*tr(inv(Formation[1,i-1].InfoMatrix))
            
            FLAG_GPS = 1;

        else

            FLAG_GPS = 0;

        end

        #### S/C 1
        x_new_chief,P_chief,phi_t_chief,aux1_Info_Jk_chief = ChiefFilterEKFFunctionwCRLB(Formation[1,i-1].PosVector,Y_chief[i],Formation[1,i-1].P,T,R_chief,Q_chief,Formation[1,i-1].InfoMatrix,FLAG_GPS);
        SpacecraftChief = SpacecraftETC(x_new_chief,X_chief[i],P_chief,zeros(18,6),zeros(18,18),aux1_Info_Jk_chief);

        

        ########## Deputies ##########

        if FLAG_GPS == 1 || tr(convert(Matrix{Float64},Formation[2,i-1].P)) > CL_Deputy*tr(inv(Formation[2,i-1].InfoMatrix)) ||
            tr(convert(Matrix{Float64},Formation[3,i-1].P)) > CL_Deputy*tr(inv(Formation[3,i-1].InfoMatrix)) ||
            tr(convert(Matrix{Float64},Formation[4,i-1].P)) > CL_Deputy*tr(inv(Formation[4,i-1].InfoMatrix))

            Flag_RR = 1;

        else

            Flag_RR = 0;

        end

        #### S/C 2 
        x_pos,P,P_xc,P_xx,Info_Mat = DeputyFilterEKFFunctionwCRLB(Formation[2,i-1].PosVector,x_new_chief,P_chief,Y[i],Formation[2,i-1].P,Formation[2,i-1].P_xc,Formation[2,i-1].P_xx,Formation[2,i-1].InfoMatrix,T,R,Q,Flag_RR)
        SpacecraftDep1 = SpacecraftETC(x_pos,X[i],P,P_xc,P_xx,Info_Mat);

        #### S/C 3
        x_pos,P,P_xc,P_xx,Info_Mat = DeputyFilterEKFFunctionwCRLB(Formation[3,i-1].PosVector,x_new_chief,P_chief,Y[i],Formation[3,i-1].P,Formation[3,i-1].P_xc,Formation[3,i-1].P_xx,Formation[3,i-1].InfoMatrix,T,R,Q,Flag_RR)
        SpacecraftDep2 = SpacecraftETC(x_pos,X[i],P,P_xc,P_xx,Info_Mat);

        #### S/C 4
        x_pos,P,P_xc,P_xx,Info_Mat = DeputyFilterEKFFunctionwCRLB(Formation[4,i-1].PosVector,x_new_chief,P_chief,Y[i],Formation[4,i-1].P,Formation[4,i-1].P_xc,Formation[4,i-1].P_xx,Formation[4,i-1].InfoMatrix,T,R,Q,Flag_RR)
        SpacecraftDep3 = SpacecraftETC(x_pos,X[i],P,P_xc,P_xx,Info_Mat);

        # Update Formation Data Structures

        global Formation = hcat(Formation, [SpacecraftChief,SpacecraftDep1,SpacecraftDep2,SpacecraftDep3])

            
  
end

############# Error Handling #############
EKF_dev_min1 = zeros(SizeOfDataSet,1);
EKF_dev_min2 = zeros(SizeOfDataSet,1);
EKF_dev_min3 = zeros(SizeOfDataSet,1);
EKF_dev_min4 = zeros(SizeOfDataSet,1);

for k = 1:SizeOfDataSet
        EKF_dev_min1[k] = norm(Formation[1,k].PosVector[1:3,:]-Formation[1,k].TruePos[1:3,:]);
        EKF_dev_min2[k] = norm(Formation[2,k].PosVector[1:3,:]-Formation[2,k].TruePos[1:3,:]);
        EKF_dev_min3[k] = norm(Formation[3,k].PosVector[7:9,:]-Formation[3,k].TruePos[7:9,:]);
        EKF_dev_min4[k] = norm(Formation[4,k].PosVector[13:15,:]-Formation[4,k].TruePos[13:15,:]);
end   


mat"

figure
txt5 = ['Position Error S/C 1'];
txt6 = ['Position Error S/C 2'];
txt7 = ['Position Error S/C 3'];
txt8 = ['Position Error S/C 4'];

plot($ElapSEC(1:395,:)./(60*60),$EKF_dev_min1,'--','DisplayName',txt5);
hold on;
plot($ElapSEC(1:395,:)./(60*60),$EKF_dev_min2,'--','DisplayName',txt6);
plot($ElapSEC(1:395,:)./(60*60),$EKF_dev_min3,'--','DisplayName',txt7);
plot($ElapSEC(1:395,:)./(60*60),$EKF_dev_min4,'--','DisplayName',txt8);
grid on;
set(gca, 'YScale', 'log');

legend show;
title(['Tracking position error through time'],'FontSize',14);
grid on;
ylabel('Deviation [Km]','FontSize',12);
xlabel('Time [h]','FontSize',12);


Chief_RMSE_Consider_sat1 = sqrt((1/(395-300))*sum($EKF_dev_min1(300:end,:).^2))
Deputies_RMSE_Consider_sat2 = sqrt((1/(395-300))*sum($EKF_dev_min2(300:end,:).^2))
Deputies_RMSE_Consider_sat3 = sqrt((1/(395-300))*sum($EKF_dev_min3(300:end,:).^2))
Deputies_RMSE_Consider_sat4 = sqrt((1/(395-300))*sum($EKF_dev_min4(300:end,:).^2))

"