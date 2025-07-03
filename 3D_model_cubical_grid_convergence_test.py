import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

#Aim of this program is to conduct a grid convergence test to see how fast will my model
#converge for the right answer and to confirm its validity. 

#Instruction
#-------------
#When running the program do not aim to plot it with the simuulation, since it takes a while
#to run the program, best method will be to keep it as is and use the generated data file to 
#then plot the data in a software of your choice.
#Otherwise, each time you would like to adjust the plot, you will have to wait a long time.
#----------------------------------------------------------------------------------------------
#Solving the heat equation as before, note that now the temporal term is a gaussian
#Update for other models will be implmented in due time.
#The change will not affect the qualititive properties found out until now.
#----------------------------------------------------------------------------------------
def solve_heat_equation_3D(length, step_number, dt, total_time, rho, c, k, P, w, sigma, T0, mu_a):
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

    for t in range(time_steps):
        current_time = t * dt

        spatial_term = np.exp(-(X**2 + Y**2) / w**2) * np.exp(-Z / delta)
        temporal_term = np.exp(-((current_time)**2) / (2 * sigma ** 2))
        source_term = ((I0 * mu_a) / (rho * c)) * spatial_term * temporal_term

        laplacian = np.zeros_like(T)
        laplacian[1:-1,1:-1,1:-1] = (
            (T[2:,1:-1,1:-1] + T[:-2,1:-1,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dx**2 +
            (T[1:-1,2:,1:-1] + T[1:-1,:-2,1:-1] - 2*T[1:-1,1:-1,1:-1]) / dy**2 +
            (T[1:-1,1:-1,2:] + T[1:-1,1:-1,:-2] - 2*T[1:-1,1:-1,1:-1]) / dz**2
        )

        T_new = T + dt * (alpha * laplacian + source_term)
        T = T_new

        # Neumann boundary conditions (zero gradient)
        T[0,:,:] = T[1,:,:]
        T[-1,:,:] = T[-2,:,:]
        T[:,0,:] = T[:,1,:]
        T[:,-1,:] = T[:,-2,:]
        T[:,:,0] = T[:,:,1]
        T[:,:,-1] = T[:,:,-2]

    return T

if __name__ == "__main__":
    length = 0.1          # meters
    P = 70                # Watts
    w = 0.001              # beam waist (meters)
    sigma = 1e-4            # relaxation time (seconds)
    total_time = 0.01     # total simulation time (seconds)
    rho, c, k = 1050, 200, 0.03  # polystyrene properties
    T0 = 20               # initial temperature (deg C)

    # Absorption coefficient from penetration depth (1 mm)
    penetration_depth = 0.001  # meters
    mu_a = 1 / penetration_depth
    alpha = k / (rho * c)
    grid_size_list = [20,30, 40,50, 60,70, 80,90,100,110,120,130,140,150,160,170,180]
    error_list = []
    h = []
    #CFL Factor to maintain stable dt
    cfl_factor = 0.05
    min_steps = 100
    #grid size of 200**3 will be the reference solution we compare againts
    ref_step_number = 200
    ref_grid_spacing = length / (ref_step_number - 1)
    dt_ref = cfl_factor * ref_grid_spacing**2 / alpha
    dt_ref = min(dt_ref, total_time / min_steps)
    T_ref = solve_heat_equation_3D(length, ref_step_number, dt_ref, total_time, rho, c, k, P, w, sigma, T0, mu_a)
    print(f"Reference grid {ref_step_number} max T: {np.max(T_ref):.2f}")

    for step_number in grid_size_list:
        current_grid_spacing = length / (step_number - 1)
        dt = cfl_factor * current_grid_spacing**2 / alpha

        T_current = solve_heat_equation_3D(length, step_number, dt, total_time, rho, c, k, P, w, sigma, T0, mu_a)
        print(f"Grid size {step_number} max T: {np.max(T_current):.2f}")

        h.append(current_grid_spacing)

        zoom_factors = (
            T_ref.shape[0] / T_current.shape[0],
            T_ref.shape[1] / T_current.shape[1],
            T_ref.shape[2] / T_current.shape[2]
        )
        T_current_resampled = zoom(T_current, zoom_factors, order=1)
        #Turning the error to be a scalar value
        error_norm = np.linalg.norm(T_ref - T_current_resampled) / np.linalg.norm(T_ref)
        error_list.append(error_norm)
        #Calculating the number of steps -- not interested in it as of now
        num_steps = int(total_time / dt)
        print(f"Grid {step_number}: dt={dt:.2e}, steps={num_steps}, relative error={error_norm:.5e}")

    print(f"Std dev of errors: {np.std(error_list):.5f}")

    log10_h = np.log10(h)
    log10_error = np.log10(error_list)
    #Saving the data to a file so it can be used later to plot
    data = np.column_stack((log10_h, log10_error))
    np.savetxt("convergence_data.csv", data, delimiter=",", header="log10(h),log10(Relative Error)", comments='')
    #Calculating statistical properties
    slope, intercept, r_value, p_value, std_err = linregress(log10_h, log10_error)

    print(f"Order of convergence (slope): {slope:.3f}")
    print(f"Intercept: {intercept:.3f}")
    print(f"R^2 value: {r_value**2:.4f}")
    print(f"p-value for slope: {p_value:.4e}")
    print(f"Standard error of slope: {std_err:.4f}")
    
    #-------------------------------------------------------------------------------------------
    #coeffs = np.polyfit(log_h, log_error, 1)
    #order_of_convergence = coeffs[0]
    #print(f"Estimated order of convergence: {order_of_convergence:.3f}")

    #plt.plot(log_h, log_error, 'o-', label='log(Error)')
    #plt.xlabel('log(Grid spacing h)')
    #plt.ylabel('log(Relative Error)')
    #plt.legend()
    #plt.savefig('Grid convergence')
    #plt.show()
      #Calculating statstical properties and plotting data
    # Linear regression (gives slope, intercept, r_value, p_value, std_err)
    #----------------------------------------------------------------------------------------------------------
    
