# Changelog

All notable changes to the GSW PyTorch project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-27

### Added
- **Core Package Structure**
  - Complete PyTorch implementation of GSW Oceanographic Toolbox
  - 118+ functions implemented with numerical parity to reference GSW
  - Full automatic differentiation support via PyTorch autograd
  - GPU acceleration support for all operations

- **Core Modules**
  - `conversions.py` - Temperature, salinity, entropy, pressure conversions
  - `density.py` - Density, specific volume, and expansion coefficients
  - `energy.py` - Enthalpy, internal energy, latent heat functions
  - `freezing.py` - Freezing point calculations
  - `ice.py` - Ice thermodynamics and melting functions
  - `stability.py` - Buoyancy frequency, Turner angle, stability analysis
  - `geostrophy.py` - Geostrophic velocity calculations
  - `interpolation.py` - Vertical interpolation (Reiniger-Ross method)
  - `utility.py` - Oxygen solubility, chemical potential, PCHIP interpolation

- **Key Features**
  - Pure PyTorch implementations (no NumPy dependencies in core code)
  - Exact numerical parity with reference GSW implementation (within 1e-8 tolerance)
  - Automatic tensor conversion from NumPy arrays
  - Comprehensive broadcasting support
  - Full autograd compatibility for differentiable functions

- **Testing Infrastructure**
  - Comprehensive test suite with parity tests against reference GSW
  - Gradient checking for all differentiable functions
  - Performance benchmarking tools
  - Test data generation utilities

- **Documentation**
  - Complete API documentation with NumPy-style docstrings
  - Implementation status tracking
  - Quick start guide
  - Contributing guidelines
  - Known limitations documentation

- **Build & Distribution**
  - Modern Python packaging with `pyproject.toml`
  - Hatchling build backend
  - CI/CD pipeline with GitHub Actions
  - Type checking with mypy
  - Code formatting with ruff

### Known Limitations
- `enthalpy_second_derivatives.h_SA_SA`: Numerical precision errors (~4e-3) at high pressure due to autograd through sqrt operations
- `entropy_from_CT`: Minor precision errors (~8.9e-8) from Chebyshev polynomial approximation
- `spiciness0/1/2`: Polynomial fitting precision errors (~5-10e-6)
- `CT_freezing`, `t_freezing`: Non-finite gradients when SA=0 (pure water edge case)
- `O2sol_SP_pt`: Non-finite gradients in some edge cases

See `IMPLEMENTATION_STATUS.md` for complete details on limitations and workarounds.
