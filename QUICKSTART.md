# GSW PyTorch Quick Start Guide

## Installation

```bash
# Create virtual environment with uv
uv venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install package
uv pip install -e .

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

## Basic Usage

```python
import torch
import gsw_torch

# Create input tensors
SA = torch.tensor([35.0], dtype=torch.float64)  # Absolute Salinity, g/kg
t = torch.tensor([15.0], dtype=torch.float64)   # In-situ temperature, °C
p = torch.tensor([100.0], dtype=torch.float64)  # Pressure, dbar

# Convert to Conservative Temperature
CT = gsw_torch.CT_from_t(SA, t, p)
print(f"Conservative Temperature: {CT.item():.4f} °C")

# Calculate specific volume and expansion coefficients
specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)
print(f"Specific volume: {specvol.item():.6e} m³/kg")
print(f"Thermal expansion: {alpha.item():.6e} 1/K")
print(f"Saline contraction: {beta.item():.6e} kg/g")

# Calculate buoyancy frequency
SA_profile = torch.tensor([35.0] * 10, dtype=torch.float64)
CT_profile = torch.linspace(20.0, 10.0, 10, dtype=torch.float64)
p_profile = torch.linspace(0, 1000, 10, dtype=torch.float64)

N2, p_mid = gsw_torch.Nsquared(SA_profile, CT_profile, p_profile)
print(f"Buoyancy frequency squared: {N2[0].item():.6e} 1/s²")
```

## Working with Profiles

```python
import torch
import gsw_torch

# Create a vertical profile
n_levels = 50
SA = torch.full((n_levels,), 35.0, dtype=torch.float64)
t = torch.linspace(25.0, 5.0, n_levels, dtype=torch.float64)  # Surface to deep
p = torch.linspace(0, 5000, n_levels, dtype=torch.float64)

# Convert to Conservative Temperature
CT = gsw_torch.CT_from_t(SA, t, p)

# Calculate stability
N2, p_mid = gsw_torch.Nsquared(SA, CT, p, axis=0)

# Calculate Turner angle and stability ratio
Tu, Rsubrho, p_mid = gsw_torch.Turner_Rsubrho(SA, CT, p, axis=0)
```

## Geostrophic Calculations

```python
import torch
import gsw_torch

# Create section data
lon = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64)
lat = torch.tensor([45.0, 45.0, 45.0, 45.0], dtype=torch.float64)

# Calculate distances between stations
dist = gsw_torch.distance(lon, lat)

# Calculate Coriolis parameter
f = gsw_torch.f(lat)

# Calculate geostrophic velocity (requires geo_strf_dyn_height)
# geo_strf = ...  # Calculate dynamic height first
# u, mid_lon, mid_lat = gsw_torch.geostrophic_velocity(geo_strf, lon, lat)
```

## Automatic Differentiation

All implemented functions support PyTorch's automatic differentiation:

```python
import torch
import gsw_torch

# Enable gradient tracking
SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
CT = torch.tensor([15.0], dtype=torch.float64)
p = torch.tensor([100.0], dtype=torch.float64)

# Calculate specific volume
specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)

# Calculate gradient
specvol.backward()
print(f"d(specvol)/d(SA): {SA.grad.item():.6e}")
```

## GPU Support

All functions work on GPU by using CUDA tensors:

```python
import torch
import gsw_torch

# Create tensors on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SA = torch.tensor([35.0], dtype=torch.float64, device=device)
CT = torch.tensor([15.0], dtype=torch.float64, device=device)
p = torch.tensor([100.0], dtype=torch.float64, device=device)

# All operations run on GPU
specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)
```

## Numpy Compatibility

Functions automatically convert numpy arrays to torch tensors:

```python
import numpy as np
import gsw_torch

# Use numpy arrays directly
SA = np.array([35.0, 36.0, 37.0])
CT = np.array([15.0, 14.0, 13.0])
p = np.array([100.0, 500.0, 1000.0])

# Functions handle conversion automatically
specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)

# Results are torch tensors
print(type(specvol))  # <class 'torch.Tensor'>
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_conversions.py

# Run with verbose output
pytest -v

# Run integration tests
pytest tests/test_integration.py
```

## Common Patterns

### Temperature Conversions

```python
# In-situ temperature <-> Conservative Temperature
t = torch.tensor([15.0], dtype=torch.float64)
CT = gsw_torch.CT_from_t(SA, t, p)
t_recovered = gsw_torch.t_from_CT(SA, CT, p)
```

### Salinity Conversions

```python
# Practical Salinity <-> Absolute Salinity
SP = torch.tensor([35.0], dtype=torch.float64)
lon = torch.tensor([0.0], dtype=torch.float64)
lat = torch.tensor([0.0], dtype=torch.float64)

SA = gsw_torch.SA_from_SP(SP, p, lon, lat)
SP_recovered = gsw_torch.SP_from_SA(SA, p, lon, lat)
```

### Stability Analysis

```python
# Calculate buoyancy frequency
N2, p_mid = gsw_torch.Nsquared(SA, CT, p)

# Calculate Turner angle
Tu, Rsubrho, p_mid = gsw_torch.Turner_Rsubrho(SA, CT, p)

# Calculate IPV ratio
IPV_ratio, p_mid = gsw_torch.IPV_vs_fNsquared_ratio(SA, CT, p, p_ref=0)
```

## Notes

- All functions use `torch.float64` (double precision) by default for numerical accuracy
- Functions accept both torch tensors and numpy arrays as input
- Scalar results are returned as 0-dimensional tensors (use `.item()` to get Python scalars)
- Functions that are not yet implemented will raise `NotImplementedError`
- See `IMPLEMENTATION_STATUS.md` for current implementation status
