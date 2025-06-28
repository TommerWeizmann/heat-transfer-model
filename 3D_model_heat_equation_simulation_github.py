#The aim of this project is to develop a solver for the 3D Heat equation
#Building on solutions developed yesterday, we have a solution
#Aim is to use finite difference to numerically solve the 3D heat equation
#-------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
#-------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
# Algorithm to solve the 3D heat equation:
#Step 1: Define the domain Lx=Ly=Lz=L and step distance dx=dy=dz=L/(N-1)
#Step 2: Pick time step dt and simultion time
#Step 3: Initialize the temperature distribution T(x,y,z,0) = T0(x,y,z)
#Step 4: Define the source term f(x,y,z,t)
#Step 5: Time loop for each time step t:
#   Step 5.1: Compute the laplcian of the temperature distribution
#   Step 5.2: Compute the source term at this time step
#   Step 5.3: Update the temperature distribution using the heat equation
#   Step 5.4: Apply boundary conditions T(x,y,z,t) = T0(x,y,z) and dT/dn = 0 at the boundaries
#Step 6: Store results
# Function to solve the 3D heat equation with a Gaussian laser pulse
#--------------------------------------------------------------------------------------------

def solve_heat_equation_3D(length, step_number, dt, total_time, rho, c, k, P, w, tau, T0):
    I0 = P / (np.pi * w**2)  # Peak intensity in W/m^2
    alpha = k / (rho * c)  # Thermal diffusivity in m^2/s
    
    dx = dy = dz = length / (step_number - 1)
    time_steps = int(total_time / dt)
    T = np.full((step_number, step_number, step_number), T0, dtype=float)
    
    def calculate_source(x, y, z, tau, I0, w, t, mu_a):
        delta = 1 / mu_a  # Penetration depth in m
        spatial_term = np.exp(-(x**2 + y**2)/w**2)*np.exp(-z/delta) 
        temporal_term = np.exp(-(t**2) / (tau**2))
        return (I0   / (rho * c)) * spatial_term * temporal_term

    for t in range(time_steps):
        current_time = t * dt
        
        # Calculate laplacian using finite difference
        laplacian = np.zeros_like(T)
        for i in range(1, step_number - 1):
            for j in range(1, step_number - 1):
                for k in range(1, step_number - 1):
                    laplacian[i, j, k] = (
                        (T[i + 1, j, k] + T[i - 1, j, k] - 2 * T[i, j, k]) / dx**2 +
                        (T[i, j + 1, k] + T[i, j - 1, k] - 2 * T[i, j, k]) / dy**2 +
                        (T[i, j, k + 1] + T[i, j, k - 1] - 2 * T[i, j, k]) / dz**2
                    )
        
        # Calculate source term
        source = np.zeros_like(T)
        for i in range(step_number):
            for j in range(step_number):
                for k in range(step_number):
                    x = (i - step_number//2) * dx  # Center the source
                    y = (j - step_number//2) * dy
                    z = k * dz
                    source[i, j, k] = calculate_source(x, y, z, tau, I0, w, current_time, mu_a)
        
        # Update temperature (method shown in the report file)
        T_new = T + dt * (alpha * laplacian + source)
        T = T_new
        
        #Boundry conditions Neumann
        T[0, :, :] = T[1, :, :]
        T[-1, :, :] = T[-2, :, :]
        T[:, 0, :] = T[:, 1, :]
        T[:, -1, :] = T[:, -2, :]
        T[:, :, 0] = T[:, :, 1]
        T[:, :, -1] = T[:, :, -2]
    
    return T

if __name__ == "__main__":
    # defult parameters (held constant)
    length = 0.001  # 1 mm (reduced domain size)
    P = 1e6        # 1 MW
    w = 1e-4       # 0.1 mm
    tau = 1e-6     # 1 microsecond
    grid_points = 6  # Reduced from 50 to 6
    total_time = 1e-4  # 100 microseconds
    rho, c, k = 2700, 900, 237  # Aluminum properties
    T0 = 20
    alpha = k / (rho * c)
    mu_a = 0.1 * length  # Increased absorption

    # Parameter lists for testing
    length_list = [0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.07,0.08,0.09,0.1]
    P_list = [1e6, 2e6, 3e6, 4e6, 5e6]  # 1-5 MW
    w_list = [1e-7,1e-6,1e-5 ,1e-4, 1e-3, 1e-2, 1e-1]
    tau_list = [1e-7, 1e-6, 1e-5]
    # Choose which parameter to vary
    parameter_to_vary = "BeamRadius"  # Options: , "Power", "Length" , "PulseDuration"

    # Header
    print(f"{'Beam Radius (m)':<15} {'Max Temp (°C)':<15}")

    if parameter_to_vary == "Power":
        dx = length / (grid_points - 1)
        dt = 1e-7  # 100 nanoseconds
        for i, P in enumerate(P_list):
            print(f"Calculating for {P/1e6:.1f} MW...", end='\r')
            T = solve_heat_equation_3D(length, grid_points, dt, total_time, rho, c, k, P, w, tau, T0)
            max_temp = np.max(T)
            print(f"{P/1e6:<15.1f} {max_temp:<15.2f}")

    elif parameter_to_vary == "Length":
        dx = length / (grid_points - 1)
        dt = 1e-7  # 100 nanoseconds
        for length in length_list:
            print(f"Calculating for length {length:.3e} m...", end='\r')
            T = solve_heat_equation_3D(length, grid_points, dt, total_time, rho, c, k, P, w, tau, T0)
            max_temp = np.max(T)
            print(f"{length:<15.3e} {max_temp:<15.2f}")

    elif parameter_to_vary == "BeamRadius":
        dx = length / (grid_points - 1)
        dt = 1e-7  # 100 nanoseconds
        for w in w_list:
            print(f"Calculating for beam radius {w:.2e} m...", end='\r')
            T = solve_heat_equation_3D(length, grid_points, dt, total_time, rho, c, k, P, w, tau, T0)
            max_temp = np.max(T)
            print(f"{w:<15.2e} {max_temp:<15.2f}")

    elif parameter_to_vary == "PulseDuration":
        dx = length / (grid_points - 1)
        dt = 1e-7  # 100 nanoseconds
        for tau in tau_list:
            print(f"Calculating for pulse duration {tau:.2e} s...", end='\r')
            T = solve_heat_equation_3D(length, grid_points, dt, total_time, rho, c, k, P, w, tau, T0)
            max_temp = np.max(T)
            print(f"{tau:<15.2e} {max_temp:<15.2f}")
            