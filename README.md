# Heat Transfer Model

This repository contains a three-dimensional heat transfer model for simulating transient temperature distributions under laser heating. The model combines the analytical framework of the heat equation with a Gaussian laser source term and a finite-difference numerical implementation. Aluminum is used as the reference material, although the framework can be adapted to other materials by modifying the relevant physical parameters.

This project was completed as an independent summer project following my first year of undergraduate study and is retained as a record of my early work in numerical methods, heat transfer, and scientific computing.

## Key Findings

* Local temperature increases approximately linearly with laser power.
* Temperature exhibits a nonlinear response to pulse duration.
* Smaller beam radii produce higher localized temperatures due to increased energy concentration.
* Temperature rise decreases with distance from the laser interaction region.
* Stability analysis confirmed the expected FTCS stability behaviour, with divergence occurring as the time step approaches the theoretical stability limit.
* Grid refinement studies demonstrated consistent convergence of the numerical solution, with an observed convergence rate of approximately 1.24.
* Simulations showed a transition in behaviour around a 3 cm domain length, beyond which further increases in domain size produced comparatively small changes in the calculated error. This observation may be useful for selecting computational domain sizes in similar simulations.

## Numerical Verification

The model was evaluated using:

* FTCS stability analysis
* Grid convergence analysis
* Sensitivity studies on key laser parameters

These analyses provide evidence that the numerical implementation behaves consistently and converges under grid refinement. They do not constitute experimental validation, and future work could compare the model against analytical benchmarks, experimental measurements, or established simulation software.

## Assumptions and Limitations

The current model assumes:

* Constant material properties
* Idealized boundary conditions
* A controlled environment without complex material effects such as phase transitions or temperature-dependent properties

As a result, the model is most appropriate for investigating qualitative trends and numerical behaviour rather than making high-precision engineering predictions.

## Important Note on the Absorption Coefficient (μₐ)

The current implementation omits the absorption coefficient (μₐ) as a modelling simplification. While this affects the absolute temperature scale, it does not significantly alter the qualitative relationships observed between temperature and the primary laser parameters. Future versions may incorporate material-specific absorption coefficients to improve physical realism and energy scaling.
