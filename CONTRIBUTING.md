# Contributing to GSW PyTorch

Thank you for your interest in contributing to GSW PyTorch! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Follow the project's coding standards

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/gsw-torch.git`
3. Create a virtual environment: `uv venv`
4. Install in development mode: `uv pip install -e ".[dev]"`
5. Create a branch for your changes: `git checkout -b feature/your-feature-name`

## Development Setup

### Environment

We use `uv` for environment management:

```bash
# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install package and dev dependencies
uv pip install -e ".[dev]"
```

### Code Quality Tools

- **Ruff**: Linting and formatting
  ```bash
  ruff check gsw_torch tests
  ruff format gsw_torch tests
  ```

- **MyPy**: Type checking
  ```bash
  mypy gsw_torch
  ```

- **Pytest**: Testing
  ```bash
  pytest tests/
  pytest tests/ --cov=gsw_torch --cov-report=html
  ```

## Coding Standards

### Style Guide

- Follow PEP 8 style guidelines
- Use `ruff` for automatic formatting (line length: 100)
- Use type hints for all public functions
- Write comprehensive docstrings in NumPy format

### Docstring Format

All public functions should have NumPy-style docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of the function.

    Parameters
    ----------
    param1 : torch.Tensor
        Description of param1
    param2 : float
        Description of param2

    Returns
    -------
    result : torch.Tensor
        Description of return value

    Notes
    -----
    Additional notes about the function, implementation details, etc.

    References
    ----------
    Reference to papers, GSW-C source, etc.
    """
```

### Type Hints

- Use `torch.Tensor` for tensor inputs/outputs
- Use `typing.Union` for optional types
- Use `typing.Optional` for nullable types
- Import from `typing` module when needed

### Tensor Handling

- Default dtype: `torch.float64` (double precision)
- Use `as_tensor()` utility for input conversion
- Support automatic broadcasting
- Handle device placement (CPU/GPU) automatically

## Implementation Guidelines

### Pure PyTorch Requirement

- **No NumPy in core code**: All implementations in `gsw_torch/_core/` must use PyTorch only
- **No reference wrappers**: Functions should not call reference GSW implementations
- **GPU compatible**: All operations must work on CUDA tensors
- **Autograd compatible**: All differentiable functions must support `requires_grad=True`

### Numerical Parity

- Target accuracy: `atol=1e-8, rtol=1e-6` compared to reference GSW
- Test against reference implementation in `tests/test_parity.py`
- Document any known limitations in `IMPLEMENTATION_STATUS.md`

### Function Organization

- Core implementations go in `gsw_torch/_core/` modules
- Public API exports go in `gsw_torch/` module files
- Add functions to appropriate `__all__` lists
- Export from main `__init__.py` if part of public API

## Testing

### Test Structure

- Unit tests: `tests/test_*.py`
- Parity tests: Compare with reference GSW implementation
- Gradient tests: Verify autograd compatibility
- Integration tests: Test workflows and combinations

### Writing Tests

```python
import torch
import pytest
import gsw_torch

def test_function_name():
    """Test description."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([10.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)
    
    result = gsw_torch.function_name(SA, CT, p)
    
    assert result.shape == SA.shape
    assert torch.isfinite(result).all()
    # Add more assertions
```

### Test Markers

- `@pytest.mark.parity`: Requires reference GSW installation
- `@pytest.mark.gradcheck`: Gradient checking tests
- `@pytest.mark.comprehensive`: Comprehensive validation tests
- `@pytest.mark.autograd`: Autograd compatibility tests

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_conversions.py

# Run with coverage
pytest --cov=gsw_torch --cov-report=term-missing

# Run only parity tests
pytest -m parity

# Run only gradient checks
pytest -m gradcheck
```

## Adding New Functions

### Step-by-Step Process

1. **Implement in `_core` module**
   - Add implementation to appropriate `gsw_torch/_core/*.py` file
   - Follow existing code patterns
   - Ensure pure PyTorch implementation

2. **Export from module**
   - Add function to module's `__all__` list
   - Import in corresponding `gsw_torch/*.py` file
   - Add to main `__init__.py` if part of public API

3. **Write tests**
   - Add unit tests in `tests/test_*.py`
   - Add parity tests if reference implementation exists
   - Add gradient checks if function is differentiable

4. **Update documentation**
   - Update `IMPLEMENTATION_STATUS.md`
   - Ensure docstrings are complete
   - Add examples if helpful

5. **Verify**
   - Run tests: `pytest`
   - Check linting: `ruff check`
   - Check types: `mypy gsw_torch`
   - Verify parity: `pytest -m parity`

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Reference issue numbers if applicable
- Example: "Add CT_from_enthalpy function with iterative solver"

## Pull Request Process

1. **Before submitting**
   - Ensure all tests pass
   - Run code quality checks (ruff, mypy)
   - Update documentation as needed
   - Rebase on latest main branch

2. **PR description**
   - Describe what changes were made
   - Explain why changes were needed
   - Reference related issues
   - Include test results if applicable

3. **Review process**
   - Address reviewer feedback promptly
   - Make requested changes
   - Keep PR focused on single feature/fix

## Known Issues & Limitations

If you encounter or fix a known limitation:

1. Document it in `IMPLEMENTATION_STATUS.md`
2. Include workarounds if available
3. Note impact on users
4. Consider adding tests for edge cases

## Questions?

- Open an issue for questions or discussions
- Check `IMPLEMENTATION_STATUS.md` for implementation details
- Review existing code for examples
- Check `README.md` for usage examples

## License

By contributing, you agree that your contributions will be licensed under the BSD 3-Clause License.
