import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

#This program generates a simulation of the 3D heat equation with a source term for square 
#surface to mimic a cube shape and circular surface to mimic a slab in cylindrical coordinates
#The z length is the domain of the object and delta is the penetration depth.
#The following is the algorithm used to solve the heat equation using finite difference
#------------------------------------------------------------------------------------------
#Algorithm:
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
#------------------------------------------------------------------------------------------
#Cubical surface function:
#The function takes the following parameters:
#length - domain length in the z direction
#step_number - Grid_size
#total_time - running time of the simulation
#rho - density of the material (various rho values and other thermal parameters are given the main part)
#c - specific latent heat
#k - thermal conductivity
#P - laser power (source power)
#Sigma - Pulse duration 
#T0 = initial temperature (set to 20 to simulate room temperature
#mu_a - absorption coefficient (varries with differnet parameters, but given rough values in the main part)
#track_centre - tracking the centre of the laser for different grids
def solve_heat_equation_3D(length, step_number, dt, total_time, rho, c, k, P, w, sigma, T0, mu_a, track_center=False):
    I0 = P / (np.pi * w**2)
    alpha = k / (rho * c)
    delta = 1 / mu_a
    dx = dy = dz = length / (step_number - 1)
    time_steps = max(1, int(total_time / dt))

    T = np.full((step_number, step_number, step_number), T0, dtype=float)
    X = (np.arange(step_number) - step_number // 2) * dx
    Y = (np.arange(step_number) - step_number // 2) * dy
    Z = np.arange(step_number) * dz
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

    max_T_overall = T0  # Track the highest temperature reached at any time
    center_temps = []   # Track center temperature over time if requested
    center_idx = step_number // 2

    for t in range(time_steps):
        current_time = t * dt
        #The source term is made of spatial and temporal term multiplied together
        #They are both seperated for simplicity and merged together for the source_term 
        spatial_term = np.exp(-(X ** 2 + Y**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-((current_time)**2) / (2 * sigma ** 2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        laplacian = np.zeros_like(T)
        laplacian[1:-1,1:-1,1:-1] = (
            (T[2:,1:-1,1:-1] + T[:-2,1:-1,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dx**2 +
            (T[1:-1,2:,1:-1] + T[1:-1,:-2,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dy**2 +
            (T[1:-1,1:-1,2:] + T[1:-1,1:-1,:-2] - 2*T[1:-1,1:-1,1:-1]) / dz**2
        )

        T_new = T + dt * (alpha * laplacian + source_term)
        max_T_overall = max(max_T_overall, np.max(T_new))  # Track max at each step

        if track_center:
            center_temps.append(T_new[center_idx, center_idx, center_idx])

        T = T_new

        # Neumann boundary conditions (zero gradient)
        #Completley insulated material such that no heat is escaping
        T[0,:,:] = T[1,:,:]
        T[-1,:,:] = T[-2,:,:]
        T[:,0,:] = T[:,1,:]
        T[:,-1,:] = T[:,-2,:]
        T[:,:,0] = T[:,:,1]
        T[:,:,-1] = T[:,:,-2]

    if track_center:
        return T, max_T_overall, np.array(center_temps)
    else:
        return T, max_T_overall
#This is the same as the previous function but in cylindrical coordinates. 
#Only difference is that now we are taking into account the radius of the surface rather then the x-y distances.
#Circular surface function:
def solve_heat_equation_3D_circular(length, step_number, dt, total_time, rho, c, k, P, w, sigma, T0, mu_a, track_center=False):
    I0 = P / (np.pi * w**2)
    alpha = k / (rho * c)
    delta = 1 / mu_a
    dx = dy = dz = length / (step_number - 1)
    time_steps = max(1, int(total_time / dt))

    T = np.full((step_number, step_number, step_number), T0, dtype=float)
    X = (np.arange(step_number) - step_number // 2) * dx
    Y = (np.arange(step_number) - step_number // 2) * dy
    Z = np.arange(step_number) * dz
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
    r = np.sqrt(X **2 + Y **2)

    max_T_overall = T0
    center_temps = []
    center_idx = step_number // 2

    for t in range(time_steps):
        current_time = t * dt

        spatial_term = np.exp(-(r**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-((current_time)**2) / (2 * sigma ** 2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        laplacian = np.zeros_like(T)
        laplacian[1:-1,1:-1,1:-1] = (
            (T[2:,1:-1,1:-1] + T[:-2,1:-1,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dx**2 +
            (T[1:-1,2:,1:-1] + T[1:-1,:-2,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dy**2 +
            (T[1:-1,1:-1,2:] + T[1:-1,1:-1,:-2] - 2*T[1:-1,1:-1,1:-1]) / dz**2
        )

        T_new = T + dt * (alpha * laplacian + source_term)
        max_T_overall = max(max_T_overall, np.max(T_new))
        if track_center:
            center_temps.append(T_new[center_idx, center_idx, center_idx])
        T = T_new

        # Neumann boundary conditions (zero gradient)
        T[0,:,:] = T[1,:,:]
        T[-1,:,:] = T[-2,:,:]
        T[:,0,:] = T[:,1,:]
        T[:,-1,:] = T[:,-2,:]
        T[:,:,0] = T[:,:,1]
        T[:,:,-1] = T[:,:,-2]

    if track_center:
        return T, max_T_overall, np.array(center_temps)
    else:
        return T

#Main program here:
#How to use: 
#------------
#Choose the model you wish (square/circular surface)
#Choose the parameters you wish to use
#Choose the thermal and optical properties to simulate the model as close to reality as possible
#To run experiments varrying various parametrs, generate list and loop through the lists to generate data

#Defult parameters here: 
length = 0.003 #3 cm - sweet spot
step_number = 30
dz = length / (step_number - 1)
P = 0.8 #Watts 
sigma = 1e-4 #seconds
T0 = 20 #Room temperature

# Metal properties (k: W/mK, rho: kg/m3, c: J/kgK)
# Source: typical values, check materials handbooks for precise data
metal_properties = {
    "Aluminum":    {"k": 237,  "rho": 2700,  "c": 897},
    "Copper":      {"k": 401,  "rho": 8960,  "c": 385},
    "Gold":        {"k": 318,  "rho": 19300, "c": 129},
    "Iron":        {"k": 80,   "rho": 7874,  "c": 449},
    "Lead":        {"k": 35,   "rho": 11340, "c": 128},
    "Nickel":      {"k": 90.9, "rho": 8908,  "c": 444},
    "Silver":      {"k": 429,  "rho": 10490, "c": 235},
    "Steel":       {"k": 50,   "rho": 7850,  "c": 486},
    "Titanium":    {"k": 21.9, "rho": 4507,  "c": 522},
    "Tungsten":    {"k": 173,  "rho": 19300, "c": 134},
    "Zinc":        {"k": 116,  "rho": 7140,  "c": 388},
    "Magnesium":   {"k": 156,  "rho": 1740,  "c": 1023},
    "Platinum":    {"k": 71.6, "rho": 21450, "c": 133},
    "Chromium":    {"k": 93.9, "rho": 7190,  "c": 448},
    "Brass":       {"k": 109,  "rho": 8530,  "c": 380},
    "Bronze":      {"k": 60,   "rho": 8800,  "c": 380},
    "StainlessSteel": {"k": 16, "rho": 8000,  "c": 500},
}
# Example: select Aluminum
k_thermal = metal_properties["Aluminum"]["k"]
rho = metal_properties["Aluminum"]["rho"]
c = metal_properties["Aluminum"]["c"]
alpha = k_thermal / (rho * c)

wavelength = 800e-9 #wavlength of 800nm, can vary as needed
# Extinction coefficients (k) at 800 nm for each metal (approximate values)
# THIS IS A ROUGH ESTIMATE LIST FOR THE SIMULATION 
#In case of specific requirements for simulation, please measure/use confirmed values

extinction_coefficients = {
    "Aluminum":    8.4,
    "Copper":      2.6,
    "Gold":        5.4,
    "Iron":        3.1,
    "Lead":        1.6,
    "Nickel":      3.3,
    "Silver":      4.0,
    "Steel":       2.5,
    "Titanium":    3.5,
    "Tungsten":    3.4,
    "Zinc":        1.0,
    "Magnesium":   1.2,
    "Platinum":    4.2,
    "Chromium":    3.3,
    "Brass":       2.0,
    "Bronze":      2.0,
    "StainlessSteel": 2.5,
}
k_optical = extinction_coefficients["Aluminum"]  # Change key as needed for other materials
alpha = k_thermal / (rho * c)
mu_a = ( 4 * np.pi * k_optical) / wavelength 

#Applying stabillity condition
dt =  dz ** 2 /(6 * alpha)

# Example: Varying power and plotting max temperature for each
power_values = [0.2e-3, 0.5e-3, 0.8e-3, 1.0e-3
               ]  # Watts
max_temps = []
center_temps_list = []

for P_test in power_values:
    T_result, max_T_overall, center_temps = solve_heat_equation_3D(
        length, step_number, dt, total_time=1e-3,  # 1 ms simulation
        rho=rho, c=c, k=k_thermal, P=P_test, w=0.0005, sigma=sigma, T0=T0, mu_a=mu_a,
        track_center=True
    )
    max_temps.append(max_T_overall)
    center_temps_list.append(center_temps)
    print(f"Power: ", P_test, f"Max temperature: ", max_T_overall)

plt.figure()
plt.plot(power_values, max_temps, marker='o')
plt.xlabel('Laser Power (W)')
plt.ylabel('Max Temperature (°C)')
plt.title('Max Temperature (any time) vs Laser Power')
plt.grid(True)
plt.show()
#Results as excepted, nearly dosent vary but with linear increaese. As before

#Example 2: Varrying beam radius for the circular surface
beam_radius_list = [0.000005,0.00005,0.0005, 0.005, 0.05, 0.5]
max_temps_beam = []

for beam_test in beam_radius_list:
    T_result, max_T_overall, centre_temp = solve_heat_equation_3D_circular(
        length, step_number, dt, total_time = 1e-3,
        rho = rho, c = c, k = k_thermal, P = P_test, w = beam_test, sigma = sigma, T0 = T0, mu_a = mu_a,
        track_center=True
    )
    max_temps_beam.append(max_T_overall)
    print(f"Beam Radius: ", beam_test, f" Max temperature: ", max_T_overall)

plt.figure()
plt.scatter(beam_radius_list, max_temps_beam, marker='o')
plt.xscale('log')  # Optional: log scale for better visualization
plt.grid(True)
plt.xlabel("Beam radius (m)")
plt.ylabel("Max Temperature (°C)")
plt.title("Max Temperature vs Beam radius")
plt.show() #Similar results to before. 
