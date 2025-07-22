import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

# ===== GLOBAL PARAMETERS =====
step_number = 10  # Grid size in each dimension for Cartesian
length = 0.003  # Cube side length (m)
T0 = 20  # Ambient temperature (C)
total_time = 1  # Total simulation time in seconds (increase as needed)
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
def solve_heat_equation_3D(
    w, k, rho, c, P, sigma, wavelength, length, material, 
    return_source=False, track_time=False
):
    step_number_local = step_number
    dz = length / (step_number_local - 1)
    dx = dy = dz
    alpha = k / (rho * c)
    dt = dz**2 / (6 * alpha)
    mu_a = (4 * np.pi * extinction_coefficients[material]) / wavelength
    delta = 1 / mu_a
    I0 = P / (np.pi * w**2)
    time_steps = max(1, int(total_time / dt))

    T = np.full((step_number_local, step_number_local, step_number_local), T0)
    X = (np.arange(step_number_local) - step_number_local // 2) * dx
    Y = (np.arange(step_number_local) - step_number_local // 2) * dy
    Z = np.arange(step_number_local) * dz
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

    max_T_overall = T0
    center_idx = step_number_local // 2
    source_value = None
    T_time_series = []

    for t in range(time_steps):
        current_time = t * dt
        spatial_term = np.exp(-(X**2 + Y**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-((current_time)**2) / (2 * sigma**2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        if t == 0 and return_source:
            source_value = source_term[center_idx, center_idx, 0]

        if current_time > 3 * sigma:
            source_term[:] = 0.0

        laplacian = np.zeros_like(T)
        laplacian[1:-1,1:-1,1:-1] = (
            (T[2:,1:-1,1:-1] + T[:-2,1:-1,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dx**2 +
            (T[1:-1,2:,1:-1] + T[1:-1,:-2,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dy**2 +
            (T[1:-1,1:-1,2:] + T[1:-1,1:-1,:-2] - 2*T[1:-1,1:-1,1:-1]) / dz**2
        )

        T_new = T + dt * (alpha * laplacian + source_term)

        # Neumann BCs (zero-flux)
        T_new[0,:,:] = T_new[1,:,:]
        T_new[-1,:,:] = T_new[-2,:,:]
        T_new[:,0,:] = T_new[:,1,:]
        T_new[:,-1,:] = T_new[:,-2,:]
        T_new[:,:,-1] = T_new[:,:,-2]

        # Convective heat loss at z=0 surface
        # This models heat loss to the environment via convection.
        # The rate is controlled by 'h' (W/m²·K): higher h = faster cooling.
        # The temperature change is proportional to (T_surface - T_ambient).
        h = 15.0
        T_ambient = T0
        q_conv = h * (T_new[:, :, 0] - T_ambient)
        dT = q_conv * dt / (rho * c * dz)
        T_new[:, :, 0] -= dT

        T = T_new
        max_T_overall = max(max_T_overall, np.max(T))

        if track_time and t % 100 == 0:
            T_time_series.append(T.copy())

    if track_time:
        return np.array(T_time_series), max_T_overall, source_value
    else:
        return T, max_T_overall, source_value


def solve_heat_equation_cylindrical_optimized(
    w, k, rho, c, P, sigma, wavelength, length, radius, material,
    return_source=False, track_time=False
):
    step_number_r = 10
    step_number_z = 10
    dr = radius / (step_number_r - 1)
    dz = length / (step_number_z - 1)
    alpha = k / (rho * c)
    dt = min(dr, dz)**2 / (6 * alpha)

    mu_a = (4 * np.pi * extinction_coefficients[material]) / wavelength
    delta = 1 / mu_a
    I0 = P / (np.pi * w**2)
    time_steps = max(1, int(total_time / dt))

    T = np.full((step_number_r, step_number_z), T0)
    r = np.linspace(0, radius, step_number_r)
    z = np.linspace(0, length, step_number_z)
    R, Z = np.meshgrid(r, z, indexing='ij')

    max_T_overall = T0
    source_value = None
    T_time_series = []

    for t in range(time_steps):
        current_time = t * dt
        spatial_term = np.exp(-(R**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(- (current_time**2) / (2 * sigma**2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        if t == 0 and return_source:
            source_value = source_term[0, 0]

        if current_time > 3 * sigma:
            source_term[:] = 0.0

        laplacian = np.zeros_like(T)

        # Internal points (r>0)
        r_mid = r[1:-1].reshape(-1,1)
        T_r_plus = T[2:, 1:-1]
        T_r_minus = T[:-2, 1:-1]
        T_r = T[1:-1, 1:-1]
        T_z_plus = T[1:-1, 2:]
        T_z_minus = T[1:-1, :-2]

        d2Tdr2 = (T_r_plus - 2*T_r + T_r_minus) / dr**2
        dTdr = (T_r_plus - T_r_minus) / (2*dr)
        radial_term = d2Tdr2 + dTdr / r_mid
        d2Tdz2 = (T_z_plus - 2*T_r + T_z_minus) / dz**2

        laplacian[1:-1, 1:-1] = radial_term + d2Tdz2

        # r=0 axis special case
        d2Tdr2_r0 = 2 * (T[1, 1:-1] - T[0, 1:-1]) / dr**2
        d2Tdz2_r0 = (T[0, 2:] - 2*T[0, 1:-1] + T[0, :-2]) / dz**2
        laplacian[0, 1:-1] = d2Tdr2_r0 + d2Tdz2_r0

        # Neumann BC at r=max (insulated)
        laplacian[-1, 1:-1] = (
            (T[-2, 1:-1] - T[-1, 1:-1]) / dr**2 +
            (T[-1, 2:] - 2*T[-1, 1:-1] + T[-1, :-2]) / dz**2
        )

        # Update temperature
        T_new = T + dt * (alpha * laplacian + source_term)

        # Neumann BC at r=max (insulated)
        T_new[-1, :] = T_new[-2, :]

        # Convective heat loss at z=0 surface
        # This models heat loss to the environment via convection.
        # The rate is controlled by 'h' (W/m²·K): higher h = faster cooling.
        # The temperature change is proportional to (T_surface - T_ambient).
        h = 15.0
        T_new[:, 0] -= h * dt * (T_new[:, 0] - T0) / (rho * c * dz)

        # Neumann BC at z=max (insulated)
        T_new[:, -1] = T_new[:, -2]

        T = T_new
        max_T_overall = max(max_T_overall, np.max(T))

        if track_time:
            T_time_series.append(T.copy())

    if track_time:
        return np.array(T_time_series), max_T_overall, source_value
    else:
        return T, max_T_overall, source_value


def main():
    #Sweeping through variables and plotting their relationship to the maximum temperature
    material = "Aluminum"
    props = metal_properties[material]

    w = 1e-3
    P = 10.0
    sigma = 1e-3
    wavelength = 800e-9
    radius = 0.0015
    length = 0.003

    power_list = np.linspace(0.1, 10, 10)
    beam_radius_list = np.linspace(1e-7, 1e-3, 10)
    length_list = np.linspace(0.001, 1.5, 10)
    wavelength_list = np.linspace(400e-9, 800e-9, 10)
    radius_list = np.linspace(0.01, 1,10)
    

    
if __name__ == "__main__":
    def extract_features(material, geometry="cube", **params):
        props = metal_properties[material]
        epsilon = extinction_coefficients[material]

        # Derived properties
        k, rho, c = props["k"], props["rho"], props["c"]
        alpha = k / (rho * c)
        mu_a = (4 * np.pi * epsilon) / params["wavelength"]
        delta = 1 / mu_a
        I0 = params["P"] / (np.pi * params["w"]**2)
        E = params["P"] * params["sigma"]

        if geometry == "cube":
            # Remove 'radius' if present
            params_cube = params.copy()
            params_cube.pop("radius", None)
            result = solve_heat_equation_3D(
                **params_cube, k=k, rho=rho, c=c, material=material,
                return_source=True, track_time=True
            )
            T_series, T_max, source_value = result
            T_final = T_series[-1]
            volume = params["length"]**3
            area = 6 * (params["length"]**2)
            center_idx = step_number // 2
        else:
            result = solve_heat_equation_cylindrical_optimized(
                **params, k=k, rho=rho, c=c, material=material,
                return_source=True, track_time=True
            )
            T_series, T_max, source_value = result
            T_final = T_series[-1]
            volume = np.pi * params["radius"]**2 * params["length"]
            area = 2 * np.pi * params["radius"] * (params["length"] + params["radius"])
            center_idx = step_number // 2  # For cylinder, use z center

        # Metrics
        avg_T = np.mean(T_final)
        var_T = np.var(T_final)
        grad = np.gradient(T_final)
        grad_mag = np.sqrt(sum(g**2 for g in grad))
        mean_grad = np.mean(grad_mag)

        # T_max location and time
        flat_idx = np.argmax(T_series)
        max_step = flat_idx // T_series[0].size
        T_max_time = max_step * (params["sigma"] / 10)  # approx time step size

        # Cooling rate after pulse
        temps = [np.max(frame) for frame in T_series]
        pulse_end_idx = int(3 * params["sigma"] / (params["sigma"] / 10))
        post_pulse = np.array(temps[pulse_end_idx:])
        times = np.arange(len(post_pulse)) * (params["sigma"] / 10)

        if len(post_pulse) > 1:
            slope, _ = np.polyfit(times, post_pulse, 1)
        else:
            slope = np.nan  # or 0, or some default value

        noise_level = 0.07  # 7% noise, realistic for scientific data

        def add_noise(value, noise_level):
            return value + np.random.normal(0, noise_level * np.abs(value))

        return {
            # Inputs
            "material": material,
            "geometry": geometry,
            "P": params["P"],
            "w": params["w"],
            "sigma": params["sigma"],
            "wavelength": params["wavelength"],
            "length": params["length"],
            "radius": params.get("radius", None),
            # Derived
            "I0": I0,
            "E": E,
            "mu_a": mu_a,
            "delta": delta,
            "volume": volume,
            "area": area,
            "surface_to_volume": area / volume,
            # Outputs
            "T_max": T_max,
            "T_max_noisy": add_noise(T_max, noise_level),
            "T_avg": avg_T,
            "T_avg_noisy": add_noise(avg_T, noise_level),
            "T_var": var_T,
            "T_var_noisy": add_noise(var_T, noise_level),
            "T_min": np.min(T_final),
            "T_median": np.median(T_final),
            "T_90th": np.percentile(T_final, 90),
            "T_max_time": T_max_time,
            "cooling_rate": slope,
            "source_peak": source_value,
            "max_temp_location": np.unravel_index(np.argmax(T_final), T_final.shape),
            # Time series at center (optional)
            "T_center_time_series": [T[center_idx, center_idx, center_idx] for T in T_series] if geometry == "cube" else [T[0,center_idx] for T in T_series],
            # Simulation meta
            # (add dt, n_steps, step_number, h if you want)
        }

    #Save data to csv:

# =====================
# Save features to CSV
# =====================
import pandas as pd

if __name__ == "__main__":
    # Sweeping through the parameters to generate data for csv file
    materials = ["Aluminum",
    "Copper",
    "Gold",
    "Iron",
    "Lead",
    "Nickel",
    "Silver",
    "Steel",
    "Titanium",
    "Tungsten",
    "Zinc",
    "Magnesium",
    "Platinum",
    "Chromium",
    "Brass",
    "Bronze",
    "StainlessSteel"]
    geometries = ["cube", "cylinder"]
    # Listing the parameters to sweep through
    #Day 1- varrying power 0.1 - 0.5, step 0.1
    #     - beam radius 1e-6 - 1e-3
    #     - wavelength 400e-9 - 700e-9
    #     - radius 0.01 - 0.03
    power_list = np.linspace(0.1, 0.5, 5)
    beam_radius_list = np.linspace(1e-6, 1e-3, 5)
    length_list = np.linspace(0.001, 1.5, 5)
    wavelength_list = np.linspace(400e-9, 700e-9, 4)
    radius_list = np.linspace(0.01, 0.05, 5)
    data = []

    # Create all combinations as a list of tuples
    param_grid = list(itertools.product(
        power_list, beam_radius_list, length_list, wavelength_list, radius_list, materials, geometries
    ))
    print(f"Total samples to generate: {len(param_grid)}")

    for idx, (power, beam_radius, length, wavelength, radius, material, geometry) in enumerate(
        tqdm(param_grid, desc="Generating data")
    ):
        if geometry == "cube":
            params = {
                "w": beam_radius,
                "P": power,
                "sigma": 1e-3,
                "wavelength": wavelength,
                "length": length,
                "radius": None  # Not used for cube
            }
        else:
            params = {
                "w": beam_radius,
                "P": power,
                "sigma": 1e-3,
                "wavelength": wavelength,
                "length": length,
                "radius": radius
            }
        print(f"Processing: material={material}, geometry={geometry}, power={power:.2f}, beam_radius={beam_radius:.2e}, length={length:.3f}, wavelength={wavelength:.1e}, radius={radius if geometry=='cylinder' else 'N/A'}", flush=True)
        features = extract_features(material, geometry, **params)
        data.append(features)

        # Save every 100 iterations
        if (idx + 1) % 100 == 0:
            df_partial = pd.DataFrame(data)
            df_partial.to_csv("heat_flux_data_day_3_partial.csv", index=False)
            print(f"Partial data saved to heat_flux_data_day_3_partial.csv at iteration {idx + 1}")

    # Final save
    df = pd.DataFrame(data)
    df.to_csv("heat_flux_data_day_3.csv", index=False)
    print("Data saved to heat_flux_data_day_3.csv")

