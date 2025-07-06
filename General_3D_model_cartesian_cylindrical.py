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

# ===== GLOBAL PARAMETERS =====
# Geometry and grid parameters
length = 0.003  # 3 cm - sweet spot
step_number = 30
dz = length / (step_number - 1)

# Laser parameters
P = 0.1  # Watts (reduced to prevent numerical issues)
sigma = 1e-3  # seconds (increased for longer pulse)
T0 = 20  # Room temperature

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

# Select material (change this to test different materials)
material = "Aluminum"
k_thermal = metal_properties[material]["k"]
rho = metal_properties[material]["rho"]
c = metal_properties[material]["c"]
alpha = k_thermal / (rho * c)

# Optical properties
wavelength = 800e-9  # wavelength of 800nm, can vary as needed
# Extinction coefficients (k) at 800 nm for each metal (approximate values)
# THIS IS A ROUGH ESTIMATE LIST FOR THE SIMULATION 
# In case of specific requirements for simulation, please measure/use confirmed values
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

k_optical = extinction_coefficients[material]
mu_a = (4 * np.pi * k_optical) / wavelength

# Time step (stability condition)
dt = dz**2 / (6 * alpha)

# Simulation time
total_time = 1e-2  # 10 ms simulation

# ===== FUNCTIONS =====

def solve_heat_equation_3D(w, track_center=False):
    """
    Solve 3D heat equation for cubical geometry
    w: beam radius (m)
    track_center: whether to track center temperature over time
    """
    I0 = P / (np.pi * w**2)
    delta = 1 / mu_a
    dx = dy = dz
    time_steps = max(1, int(total_time / dt))

    T = np.full((step_number, step_number, step_number), T0, dtype=float)
    X = (np.arange(step_number) - step_number // 2) * dx
    Y = (np.arange(step_number) - step_number // 2) * dy
    Z = np.arange(step_number) * dz
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

    max_T_overall = T0
    center_temps = []
    center_idx = step_number // 2

    for t in range(time_steps):
        current_time = t * dt
        
        # Source term
        spatial_term = np.exp(-(X**2 + Y**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-((current_time)**2) / (2 * sigma**2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        # Diagnostic print for first time step
        if t == 0:
            print(f"3D - Beam radius: {w}, Source max: {np.max(source_term):.2e}")

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
        return T, max_T_overall

def solve_heat_equation_cylindrical(w, track_center=False):
    """
    Solve 2D heat equation for cylindrical geometry (r-z plane)
    w: beam radius (m)
    track_center: whether to track center temperature over time
    """
    I0 = P / (np.pi * w**2)
    delta = 1 / mu_a
    dr = dz
    time_steps = max(1, int(total_time / dt))

    T = np.full((step_number, step_number), T0, dtype=float)  # T[r, z]
    r_vals = np.linspace(0, length, step_number)
    z_vals = np.linspace(0, length, step_number)
    R, Z = np.meshgrid(r_vals, z_vals, indexing='ij')

    max_T_overall = T0
    center_temps = []

    for t in range(time_steps):
        current_time = t * dt
        
        # Source term
        spatial_term = np.exp(-(R**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-((current_time)**2) / (2 * sigma**2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        # Diagnostic print for first time step
        if t == 0:
            print(f"Cylindrical - Beam radius: {w}, Source max: {np.max(source_term):.2e}")

        laplacian = np.zeros_like(T)

        # Correct cylindrical Laplacian
        for i in range(1, step_number - 1):
            r = r_vals[i]
            for j in range(1, step_number - 1):
                # ∂²T/∂r² term
                dr2_term = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dr**2
                # (1/r) ∂T/∂r term
                dr_term = (1/r) * (T[i+1, j] - T[i-1, j]) / (2 * dr)
                # ∂²T/∂z² term
                dz_term = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dz**2
                
                laplacian[i, j] = dr2_term + dr_term + dz_term

        # Special handling for r=0 (axis of symmetry)
        for j in range(1, step_number - 1):
            # At r=0, use L'Hôpital's rule: lim(r→0) (1/r) ∂T/∂r = ∂²T/∂r²
            dr2_term = (T[1, j] - 2*T[0, j] + T[1, j]) / dr**2  # Symmetric about r=0
            dz_term = (T[0, j+1] - 2*T[0, j] + T[0, j-1]) / dz**2
            laplacian[0, j] = 2 * dr2_term + dz_term  # Factor of 2 from L'Hôpital's rule

        T_new = T + dt * (alpha * laplacian + source_term)

        # Debug: Check if temperature is updating
        if t == 0:
            print(f"After first update, max T: {np.max(T_new):.2f}")

        # TRACK MAXIMUM TEMPERATURE BEFORE APPLYING BOUNDARY CONDITIONS
        max_T_overall = max(max_T_overall, np.max(T_new))

        if track_center:
            center_temps.append(T_new[0, step_number//2])  # r=0, center z

        # Apply boundary conditions AFTER tracking max temperature
        T_new[0, :] = T_new[1, :]  # r=0 boundary
        T_new[-1, :] = T_new[-2, :]  # r=L boundary
        T_new[:, 0] = T_new[:, 1]  # z=0 boundary
        T_new[:, -1] = T_new[:, -2]  # z=L boundary

        T = T_new

    if track_center:
        return T, max_T_overall, np.array(center_temps)
    else:
        return T, max_T_overall
# ===== MAIN PROGRAM =====
#How to use: 
#------------
#Choose the model you wish (square/circular surface)
#Choose the parameters you wish to use
#Choose the thermal and optical properties to simulate the model as close to reality as possible
#To run experiments varrying various parametrs, generate list and loop through the lists to generate data

# Example 1: Varying power for 3D cubical geometry
print("=== 3D Cubical Geometry - Power Variation ===")
power_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # Watts
max_temps_3D = []
center_temps_list_3D = []

for P_test in power_values:
    P = P_test  # Update global P
    T_result, max_T_overall, center_temps = solve_heat_equation_3D(
        w=0.001, track_center=True
    )
    max_temps_3D.append(max_T_overall)
    center_temps_list_3D.append(center_temps)
    print(f"Power: {P_test} W, Max temperature: {max_T_overall:.2f}°C")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(power_values, max_temps_3D, marker='o', linewidth=2, markersize=8)
plt.xlabel('Laser Power (W)')
plt.ylabel('Max Temperature (°C)')
plt.title('3D Cubical - Max Temperature vs Laser Power')
plt.grid(True)

# Example 2: Varying beam radius for cylindrical geometry
print("\n=== Cylindrical Geometry - Beam Radius Variation ===")
beam_radius_list = [0.0001,0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
max_temps_cyl = []

P = 0.1  # Reset to original power
for beam_test in beam_radius_list:
    T_result, max_T_overall, centre_temp = solve_heat_equation_cylindrical(
        w=beam_test, track_center=True
    )
    max_temps_cyl.append(max_T_overall)
    print(f"Beam Radius: {beam_test} m, Max temperature: {max_T_overall:.2f}°C")

plt.subplot(1, 2, 2)
plt.scatter(beam_radius_list, max_temps_cyl, marker='o', s=100)
plt.grid(True)
plt.xlabel("Beam radius (m)")
plt.ylabel("Max Temperature (°C)")
plt.title("Cylindrical - Max Temperature vs Beam radius")

plt.tight_layout()
plt.show()
#Results are the same as before.
#Values tends to increase as 1/r**2 for smaller beam radius
