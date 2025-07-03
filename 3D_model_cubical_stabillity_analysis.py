import numpy as np
import matplotlib.pyplot as plt
#Solving the heat equation as before, now the code is gaussin in temporal term too
def solve_heat_equation_3D(length, step_number, dt, total_time, rho, c, k, P, w, sigma, T0, mu_a, early_stop_threshold=1e-5):
    I0 = P / (np.pi * w**2)
    alpha = k / (rho * c)
    delta = 1 / mu_a

    dx = dy = dz = length / (step_number - 1)
    time_steps = int(total_time / dt)

    T = np.full((step_number, step_number, step_number), T0, dtype=float)
    X = (np.arange(step_number) - step_number // 2) * dx
    Y = (np.arange(step_number) - step_number // 2) * dy
    Z = np.arange(step_number) * dz
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')

    for t in range(time_steps):
        current_time = t * dt
        spatial_term = np.exp(-(X**2 + Y**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-current_time**2 / (2 * sigma**2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        laplacian = np.zeros_like(T)
        laplacian[1:-1,1:-1,1:-1] = (
            (T[2:,1:-1,1:-1] + T[:-2,1:-1,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dx**2 +
            (T[1:-1,2:,1:-1] + T[1:-1,:-2,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dy**2 +
            (T[1:-1,1:-1,2:] + T[1:-1,1:-1,:-2] - 2*T[1:-1,1:-1,1:-1]) / dz**2
        )

        T_new = T + dt * (alpha * laplacian + source_term)

        max_delta = np.max(np.abs(T_new - T))
        T = T_new

        # Neumann boundary conditions (zero flux)
        T[0,:,:] = T[1,:,:]
        T[-1,:,:] = T[-2,:,:]
        T[:,0,:] = T[:,1,:]
        T[:,-1,:] = T[:,-2,:]
        T[:,:,0] = T[:,:,1]
        T[:,:,-1] = T[:,:,-2]

    return T

# === MAIN SIMULATION ==
if __name__ == "__main__":
    # --- Physical Parameters ---
    rho, c, k = 1050, 1200, 0.03  # Polystyrene
    P, w = 70, 0.001             # Power [W], beam waist [m]
    mu_a, T0 = 1e4, 20
    length_base = 0.001 #base length for simulation (minimum 1cm)
    length = 0.01 #can be varied to any chosen length
    total_time_base = 1 #1 second simulation as a base
    total_time = total_time_base * (length / length_base)**2
    step_number = 30
    sigma = total_time / 10
    alpha = k / (rho * c)
    dx = length / (step_number - 1)
    dt_max = (1/6) * dx**2 / alpha
    print(f"FTCS stability limit dt_max ≈ {dt_max:.2e} s")

    # --- Stability Analysis ---
    N_steps = 1000
    dt_values = []
    max_temps = []

    dt_ref = 0.1 * dt_max
    T_ref = solve_heat_equation_3D(length, step_number, dt_ref, N_steps * dt_ref,
                                   rho, c, k, P, w, sigma, T0, mu_a)
    T_ref_max = np.max(T_ref)

    dt_test = dt_ref
    while dt_test <= 2.5 * dt_max:
        T_test = solve_heat_equation_3D(length, step_number, dt_test, N_steps * dt_test,
                                        rho, c, k, P, w, sigma, T0, mu_a)
        T_max = np.max(T_test)
        dt_values.append(dt_test)
        max_temps.append(T_max)
        relative_error = abs(T_max - max_temps)
        print(f"dt = {dt_test:.2e} s → T_max = {T_max:.2f} °C")
        print(f"Relative error = {relative_error} °C")
        dt_test *= 1.5

    # Final run for heatmap (using reference dt)
    T_final = T_ref
    mid_z = step_number // 2
    extent = [-length/2*1000, length/2*1000, -length/2*1000, length/2*1000]  # in mm

    # --- Plot both side-by-side ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stability plot
    ax1.plot(dt_values, max_temps, 'o-', label="Max Temp")
    ax1.axhline(T_ref_max, color='r', linestyle='--', label="Reference Temp")
    ax1.axvline(dt_max, color='k', linestyle=':', label='FTCS dt_max')
    ax1.set_xlabel("Time Step Size (s)")
    ax1.set_ylabel("Max Temperature (°C)")
    ax1.set_title("Stability Analysis: Max Temp vs dt")
    ax1.set_xscale("log")
    ax1.grid(True)
    ax1.legend()

    # Right: Temperature heatmap
    im = ax2.imshow(T_final[:, :, mid_z], extent=extent, origin='lower', cmap='hot')
    ax2.set_title("Temperature at mid-depth (XY slice)")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)")

    plt.tight_layout()
    plt.savefig("combined_plot.png", dpi=300)
    plt.show()

    print(f"Max temperature: {np.max(T_final):.2f} °C")
    print(f"Running length vs relative error analysis onwards")
    #---------------------------------------------------------------------------
    # Useing prev values
    base_length = length  # e.g., 0.01 m from your code above
    T_ref = solve_heat_equation_3D(base_length, step_number, dt_ref, N_steps * dt_ref,
                                  rho, c, k, P, w, sigma, T0, mu_a)
    T_ref_max = np.max(T_ref)

    lengths_to_test = np.linspace(base_length, 0.05, 100)  # test from current length to 5 cm, 5 points
    relative_errors = []

    for test_length in lengths_to_test:
        total_time_test = total_time_base * (test_length / length_base)**2
        sigma_test = total_time_test / 10
        dx_test = test_length / (step_number - 1)
        alpha = k / (rho * c)
        dt_max_test = (1/6) * dx_test**2 / alpha
        dt_test = min(dt_ref, dt_max_test * 0.9)  # keep dt stable for each length

        T_test = solve_heat_equation_3D(test_length, step_number, dt_test, N_steps * dt_test,
                                        rho, c, k, P, w, sigma_test, T0, mu_a)
        T_max = np.max(T_test)
        relative_error = abs(T_max - T_ref_max) / T_ref_max
        relative_errors.append(relative_error)
        print(f"Length = {test_length*100:.1f} cm, Max Temp = {T_max:.2f} °C, Relative Error = {relative_error:.4f}")

    # Plot relative error vs length
    plt.figure()
    plt.plot(lengths_to_test * 100, relative_errors, 'o-', label="Relative Error")
    plt.xlabel("Length (5cm)")
    plt.ylabel("Relative Error ($\%$)")
    plt.title("Relative Error vs Length")
    plt.grid(True)
    plt.legend()
    plt.savefig("Relative error vs length.png")
    plt.show()

    
    

