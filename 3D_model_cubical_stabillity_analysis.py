import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#Solving the heat equation using previous model
def solve_heat_equation_3D(length, step_number, dt, total_time, rho, c, k, P, w, tau, T0, mu_a):
    I0 = P / (np.pi * w**2)
    alpha = k / (rho * c)
    delta = 1 / mu_a
    dx = dy = dz = length / (step_number - 1)
    time_steps = max(1, int(total_time / dt))
    #Changed method for higher efficency, numpy vectorization method
    T = np.full((step_number, step_number, step_number), T0, dtype=float)
    X = (np.arange(step_number) - step_number // 2) * dx
    Y = (np.arange(step_number) - step_number // 2) * dy
    Z = np.arange(step_number) * dz
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
    #Calculating the source term and the laplacian
    for t in range(time_steps):
        current_time = t * dt
        
        spatial_term = np.exp(-(X**2 + Y**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-current_time / tau)
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term
        #Method for laplcian is vectorized as well for efficiency.
        laplacian = np.zeros_like(T)
        laplacian[1:-1,1:-1,1:-1] = (
            (T[2:,1:-1,1:-1] + T[:-2,1:-1,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dx**2 +
            (T[1:-1,2:,1:-1] + T[1:-1,:-2,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dy**2 +
            (T[1:-1,1:-1,2:] + T[1:-1,1:-1,:-2] - 2*T[1:-1,1:-1,1:-1]) / dz**2
        )
        #Updating the temp
        T_new = T + dt * (alpha * laplacian + source_term)
        T = T_new
        #Boundry conditions
        T[0,:,:] = T[1,:,:]
        T[-1,:,:] = T[-2,:,:]
        T[:,0,:] = T[:,1,:]
        T[:,-1,:] = T[:,-2,:]
        T[:,:,0] = T[:,:,1]
        T[:,:,-1] = T[:,:,-2]

    return T
#Main program -- Running the proejct aim
if __name__ == "__main__":
    length = 0.01         # 1 cm
    step_number = 30
    P = 70                # Watts
    w = 0.001             # 1 mm beam waist
    tau = 1e-3            # 1 ms
    total_time = 0.01     # 10 ms
    rho, c, k = 2700, 900, 237  # Aluminum
    T0 = 20
    mu_a = 1e4

    alpha = k / (rho * c)
    dx = length / (step_number - 1)
    dt_max = (1/6) * dx**2 / alpha
    
    print(f"Calculated dt_max for stability: {dt_max:.2e} s")

    N_steps = 1000
    dt_test = 0.1 * dt_max
    dt_list = []
    max_temp_list = []
    #Ref result
    results_original = solve_heat_equation_3D(length, step_number, dt_test, N_steps*dt_test, rho, c, k, P, w, tau, T0, mu_a)
    max_temp_original = np.max(results_original)
    print(f"Reference run max temperature: {max_temp_original:.4f} °C")
    #Stabillity analysis results loop
    while dt_test <= 2 * dt_max:
        total_time_test = N_steps * dt_test
        T_test = solve_heat_equation_3D(length, step_number, dt_test, total_time_test, rho, c, k, P, w, tau, T0, mu_a)
        max_temp = np.max(T_test)
        dt_list.append(dt_test)
        max_temp_list.append(max_temp)
        print(f"dt={dt_test:.2e} s, Max Temp = {max_temp:.4f} °C")
        dt_test *= 1.5

    # Final temperature distribution at original dt
    T_final = solve_heat_equation_3D(length, step_number, dt_test / 1.5**3, total_time, rho, c, k, P, w, tau, T0, mu_a)
    mid_z = step_number // 2
    
    # --- Figure 1: Stability plot ---
    fig1, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(dt_list, max_temp_list, marker='o', label='Max Temp for each dt')
    ax1.axhline(max_temp_original, color='r', linestyle='--', label=f'Reference dt={dt_list[0]:.1e}')
    ax1.set_xlabel('Time step size (s)')
    ax1.set_ylabel('Max Temperature (°C)')
    ax1.set_title('Max Temperature vs Time Step Size (Stability Analysis)')
    ax1.set_xscale('log')
    ax1.grid(True)
    ax1.legend()

    #Save the figure and show
    fig1.savefig('stability_plot.png', dpi=300)
    plt.show()

    # --- Figure 2: Temperature heatmap ---
    fig2, ax2 = plt.subplots(figsize=(6,6))
    im = ax2.imshow(T_final[:, :, mid_z], extent=[-length/2*100, length/2*100, -length/2*100, length/2*100], origin='lower')
    ax2.set_title("Temperature at mid-depth (XY slice)")
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Y (cm)")
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Temperature (°C)")

    #Save the figure and show
    fig2.savefig('temperature_heatmap.png', dpi=300)
    plt.show()
    print(dt_max)