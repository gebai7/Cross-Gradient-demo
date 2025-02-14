# Cross-Gradient Tool

The Cross Gradient Method is a geophysical joint inversion technique that enhances geological consistency by enforcing structural similarity between different physical models. It utilizes the cross-product constraint of model gradient vectors to achieve structural coupling without relying on prior petrophysical relationships.



## Usage Guide
### Initialization
Running the Demo: python examples/demo_cross_gradient.py

```
cg_solver = CrossGradient()
cg_solver.dx_dz(
    model_A, 
    model_B,
    a_b='ab',      # Similarity mode
    ni=300,        # Iterations (default:300)
    tol_=1e-20,    # Convergence tolerance (default:1e-20)
    smooth=True,   # Smoothing switch
    coef_x=2,      # X-direction smooth factor (default:2)
    coef_y=2       # Y-direction smooth factor (default:2)
)
```




## Authors
### Original Author
**Diego Domenzain**  
- GitHub Repository: [diegozain/alles](https://github.com/diegozain/alles)  
### Python Implementation Author  
**Bai Lige**  
- GitHub: [gebai7](https://github.com/gebai7)  
