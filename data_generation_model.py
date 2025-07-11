import numpy as np
import csv
import itertools
#The aim of this program is to generate data for machine learnig model to train on. This data is generated based on the solver generated previously.

# ===== GLOBAL PARAMETERS =====
step_number = 20  # Reduced from 30 for speed
length = 0.003

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

T0 = 20

def solve_heat_equation_3D(w, k, rho, c, P, sigma, wavelength, length, material, return_source=False):
    step_number_local = 20  # Reduced for speed
    dz = length / (step_number_local - 1)
    dx = dy = dz
    alpha = k / (rho * c)
    dt = dz**2 / (6 * alpha)
    mu_a = (4 * np.pi * extinction_coefficients[material]) / wavelength
    delta = 1 / mu_a
    I0 = P / (np.pi * w**2)
    time_steps = max(1, int(1e-2 / dt))
    T = np.full((step_number_local, step_number_local, step_number_local), T0)
    X = (np.arange(step_number_local) - step_number_local // 2) * dx
    Y = (np.arange(step_number_local) - step_number_local // 2) * dy
    Z = np.arange(step_number_local) * dz
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
    max_T_overall = T0
    center_idx = step_number_local // 2
    source_value = None

    for t in range(time_steps):
        current_time = t * dt
        spatial_term = np.exp(-(X**2 + Y**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-((current_time)**2) / (2 * sigma**2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        if t == 0 and return_source:
            source_value = source_term[center_idx, center_idx, 0]

        laplacian = np.zeros_like(T)
        laplacian[1:-1,1:-1,1:-1] = (
            (T[2:,1:-1,1:-1] + T[:-2,1:-1,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dx**2 +
            (T[1:-1,2:,1:-1] + T[1:-1,:-2,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dy**2 +
            (T[1:-1,1:-1,2:] + T[1:-1,1:-1,:-2] - 2*T[1:-1,1:-1,1:-1]) / dz**2
        )

        T_new = T + dt * (alpha * laplacian + source_term)
        max_T_overall = max(max_T_overall, np.max(T_new))
        T = T_new

        # Boundary conditions (Neumann)
        T[0,:,:] = T[1,:,:]; T[-1,:,:] = T[-2,:,:]
        T[:,0,:] = T[:,1,:]; T[:,-1,:] = T[:,-2,:]
        T[:,:,0] = T[:,:,1]; T[:,:,-1] = T[:,:,-2]

    return T, max_T_overall, source_value

def solve_heat_equation_cylindrical(w, k, rho, c, P, sigma, wavelength, length, material, return_source=False, track_center=False):
    step_number_local = 20  # Reduced for speed
    dz = length / (step_number_local - 1)
    dr = dz
    alpha = k / (rho * c)
    dt = dz**2 / (6 * alpha)
    mu_a = (4 * np.pi * extinction_coefficients[material]) / wavelength
    delta = 1 / mu_a
    I0 = P / (np.pi * w**2)
    time_steps = max(1, int(1e-2 / dt))
    T = np.full((step_number_local, step_number_local), T0, dtype=float)
    r_vals = np.linspace(0, length, step_number_local)
    z_vals = np.linspace(0, length, step_number_local)
    R, Z = np.meshgrid(r_vals, z_vals, indexing='ij')
    max_T_overall = T0
    center_temps = []
    source_value = None

    for t in range(time_steps):
        current_time = t * dt
        spatial_term = np.exp(-(R**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-((current_time)**2) / (2 * sigma**2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        if t == 0 and return_source:
            source_value = source_term[0, 0]

        laplacian = np.zeros_like(T)
        for i in range(1, step_number_local - 1):
            r = r_vals[i]
            for j in range(1, step_number_local - 1):
                dr2_term = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dr**2
                dr_term = (1/r) * (T[i+1, j] - T[i-1, j]) / (2 * dr)
                dz_term = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dz**2
                laplacian[i, j] = dr2_term + dr_term + dz_term
        for j in range(1, step_number_local - 1):
            dr2_term = (T[1, j] - 2*T[0, j] + T[1, j]) / dr**2
            dz_term = (T[0, j+1] - 2*T[0, j] + T[0, j-1]) / dz**2
            laplacian[0, j] = 2 * dr2_term + dz_term

        T_new = T + dt * (alpha * laplacian + source_term)
        max_T_overall = max(max_T_overall, np.max(T_new))
        T_new[0, :] = T_new[1, :]
        T_new[-1, :] = T_new[-2, :]
        T_new[:, 0] = T_new[:, 1]
        T_new[:, -1] = T_new[:, -2]
        T = T_new

    if track_center:
        if return_source:
            return T, max_T_overall, np.array(center_temps)
        else:
            return T, max_T_overall, np.array(center_temps)
    else:
        if return_source:
            return T, max_T_overall
        else:
            return T, max_T_overall
#Note that this data was generated for h = 0 (no heat flux)
# === Dataset Generation ===
#power_values = np.linspace(0.1, 1.0, 5)  # Reduced from 5 to 3
#beam_radii = np.linspace(2e-5, 1e-2, 5)  # Reduced from 5 to 3
#wavelengths = [800e-9]
#lengths = np.linspace(0.001, 1, 5)  # Reduced from 3 to 2
#sigmas = np.linspace(1e-3, 1e-1, 5)
#geometries = ["Cartesian", "Cylindrical"]
#materials = [
    #"Aluminum", "Copper", "Gold", "Iron", "Lead", "Nickel", "Silver", "Steel",
    #"Titanium", "Tungsten", "Zinc", "Magnesium", "Platinum", "Chromium",
    #"Brass", "Bronze", "StainlessSteel"
#]

#with open('new_simulation_data_to_test.csv', mode='w', newline='') as file:
    #writer = csv.writer(file)
    #writer.writerow([
        #'SimulationID', 'Geometry', 'Power_W', 'BeamRadius_m', 'Material',
        #'Wavelength_m', 'DomainLength_m', 'Sigma_s',
        #'MaxTemperature_C', 'InitialSourceTerm_W_per_kg'
    #])
    
    #sim_id = 0
    #for geometry, P_test, w_test, material_test, wl_test, length_test, sigma_test in itertools.product(
        #geometries, power_values, beam_radii, materials, wavelengths, lengths, sigmas):

        #sim_id += 1
        
        
        #k = metal_properties[material_test]['k']
        #rho = metal_properties[material_test]['rho']
        #c = metal_properties[material_test]['c']

        #if geometry.lower() == "cartesian":
            #T_result, max_T, src_val = solve_heat_equation_3D(
                #w=w_test, k=k, rho=rho, c=c, P=P_test, sigma=sigma_test,
                #wavelength=wl_test, length=length_test, material=material_test,
                #return_source=True
            #)
        #elif geometry.lower() == "cylindrical":
            #T_result, max_T, src_val = solve_heat_equation_cylindrical(
                #w=w_test, k=k, rho=rho, c=c, P=P_test, sigma=sigma_test,
                #wavelength=wl_test, length=length_test, material=material_test,
                #return_source=True
            #)
        #else:
            #continue

        #writer.writerow([
            #sim_id, geometry, P_test, w_test, material_test, wl_test, length_test, sigma_test,
            #max_T, src_val
        #])

        #print(f"Simulated: ID={sim_id}, Geometry={geometry}, P={P_test:.2f}W, w={w_test:.1e}m, Material={material_test}, "
              #f"Wavelength={wl_test*1e9:.1f}nm, Length={length_test*1000:.1f}mm, Sigma={sigma_test*1000:.1f}ms, "
              #f"MaxT={max_T:.2f}C, SourceVal={src_val:.2e}")
