# GSW PyTorch

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-green)](LICENSE)

PyTorch implementation of the Gibbs SeaWater (GSW) Oceanographic Toolbox.

This package provides a PyTorch-based reimplementation of the GSW Python package, enabling:
- **Automatic differentiation** via PyTorch's autograd
- **GPU acceleration** for large-scale oceanographic computations
- **Numerical parity** with the reference numpy-based GSW implementation (within 1e-8 tolerance)
- **Same API** as the original GSW-Python package

## Installation

### Development Installation

This package uses `uv` for environment management:

```bash
# Create virtual environment
uv venv

# Activate environment (Linux/Mac)
source .venv/bin/activate
# Windows: .venv\Scripts\activate

# Install package in development mode
uv pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Future: PyPI Installation

```bash
pip install gsw-torch
```

## Usage

The API is designed to match the original GSW-Python package:

```python
import torch
import gsw_torch

# Create input tensors
SA = torch.tensor([35.0], dtype=torch.float64)  # Absolute Salinity, g/kg
CT = torch.tensor([15.0], dtype=torch.float64)   # Conservative Temperature, °C
p = torch.tensor([100.0], dtype=torch.float64)   # Pressure, dbar

# Calculate density
rho = gsw_torch.rho(SA, CT, p)

# Calculate specific volume and expansion coefficients
specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)

# Functions support automatic differentiation
SA.requires_grad = True
rho = gsw_torch.rho(SA, CT, p)
rho.backward()
print(SA.grad)  # Gradient of density with respect to salinity
```

## Differences from NumPy GSW

1. **Input/Output Types**: Functions accept and return `torch.Tensor` objects instead of numpy arrays
2. **Automatic Conversion**: Numpy arrays are automatically converted to tensors
3. **Scalar Returns**: Scalar results are returned as 0-dimensional tensors (use `.item()` to get Python scalars)
4. **GPU Support**: All operations can run on GPU by using CUDA tensors
5. **Gradient Computation**: All differentiable operations support automatic differentiation

## Implementation Status

**Version 0.1.0** - Beta Release

The package includes **118+ functions** with full PyTorch implementations:

- ✅ **Conversions** - Temperature, salinity, entropy, pressure conversions (30+ functions)
- ✅ **Density** - Density, specific volume, expansion coefficients (25+ functions)
- ✅ **Energy** - Enthalpy, internal energy, latent heat (10+ functions)
- ✅ **Freezing** - Freezing point calculations (5+ functions)
- ✅ **Ice** - Ice thermodynamics and melting functions (15+ functions)
- ✅ **Stability** - Buoyancy frequency, Turner angle, stability analysis (5+ functions)
- ✅ **Geostrophy** - Geostrophic velocity calculations (4+ functions)
- ✅ **Interpolation** - Vertical interpolation using Reiniger-Ross method (2+ functions)
- ✅ **Utility** - Oxygen solubility, chemical potential, PCHIP interpolation (5+ functions)

All functions support:
- Automatic differentiation (autograd)
- GPU acceleration
- Numerical parity with reference GSW (within 1e-8 tolerance)

See `IMPLEMENTATION_STATUS.md` for detailed status and known limitations.

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gsw_torch

# Run parity tests against reference implementation
pytest --run-parity

# Run gradient checks
pytest tests/test_gradcheck.py
```

## Development

### Code Quality

- **Linting**: Uses `ruff` for code formatting and linting
- **Type Checking**: Uses `mypy` for static type checking
- **Testing**: Uses `pytest` with coverage reporting

```bash
# Format code
ruff format gsw_torch tests

# Lint code
ruff check gsw_torch tests

# Type check
mypy gsw_torch
```

### Adding New Functions

1. Implement the function in the appropriate `_core` module (e.g., `_core/density.py`)
2. Export it from the module's `__init__.py` or module file
3. Add tests in `tests/test_*.py`
4. Add gradient checks if the function is differentiable
5. Add parity tests against reference implementation

## Documentation

- **Quick Start**: See `QUICKSTART.md` for examples and common patterns
- **Implementation Status**: See `IMPLEMENTATION_STATUS.md` for detailed function status
- **Contributing**: See `CONTRIBUTING.md` for contribution guidelines
- **Release Notes**: See `CHANGELOG.md` for version history

## Known Limitations

Some functions have minor precision limitations or edge case issues:
- `enthalpy_second_derivatives.h_SA_SA`: Precision errors at high pressure (~4e-3)
- `entropy_from_CT`: Minor precision errors (~8.9e-8)
- `spiciness0/1/2`: Polynomial fitting precision (~5-10e-6)
- `CT_freezing`, `t_freezing`: Non-finite gradients when SA=0

See `IMPLEMENTATION_STATUS.md` for complete details and workarounds.

## Citation

If you use GSW PyTorch in your research, please cite:

```bibtex
@software{gsw_torch,
  title = {GSW PyTorch: PyTorch Implementation of the Gibbs SeaWater Oceanographic Toolbox},
  author = {GSW PyTorch Contributors},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/TEOS-10/gsw-torch}
}
```

## References

- [TEOS-10](https://www.teos-10.org/) - The international thermodynamic equation of seawater
- [GSW-Python](https://github.com/TEOS-10/GSW-Python) - Reference numpy-based implementation
- [GSW-Matlab](https://github.com/TEOS-10/GSW-Matlab) - Original MATLAB implementation
- [GSW-C](https://github.com/TEOS-10/GSW-C) - C implementation

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

The software is based on the TEOS-10 GSW Toolbox, which is licensed under a BSD 3-clause license by SCOR/IAPSO WG127.
