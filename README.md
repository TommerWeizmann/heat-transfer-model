# Heat Transfer Model

This repository contains a 3D analytical model solving the heat equation with a laser induced Gaussian source term, based on the general solution provided by Evans. The model simulates transient temperature distributions in a metallic medium using aluminum as the reference material under varying laser parameters such as beam power, pulse duration, and beam radius.

### Key Findings
- Temperature increases linearly with laser power.
- Temperature responds nonlinearly to pulse duration.
- Temperature decreases with increasing beam radius.
- A decay in local temperature rise over distance is observed.

The model assumes constant material properties and ideal boundary conditions, making it suitable for controlled environments like laboratory simulations.

### Important Note on the Absorption Coefficient (μₐ)
The current computational model omits the absorption coefficient (μₐ) as a simplification. While this affects the absolute temperature scale, it does not change qualitative trends or conclusions. This choice helps highlight visual results without extremely large or small values. Future versions may incorporate μₐ for more accurate energy scaling.

### Further Work
- **Error Analysis:** Quantify numerical errors and assess model accuracy.
- **Grid Convergence Study:** Verify results are independent of spatial discretization.
- **Time-Step Sensitivity Test:** Identify the optimal time step balancing speed and accuracy.
- **Material Property Variation:** Simulate different materials by varying thermal conductivity, density, and specific heat capacity; validate results against theory or experiments.
- **Investigation of Limitations and Uncertainties:** Analyze model assumptions, numerical approximations, and parameter variability.

