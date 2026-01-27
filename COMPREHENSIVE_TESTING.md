# Comprehensive Testing Suite

This document describes the comprehensive testing framework for validating the GSW PyTorch implementation against the reference GSW implementation.

## Overview

The comprehensive testing suite systematically tests every function in the PyTorch GSW implementation, measuring:

1. **Numerical Parity**: Compares outputs against reference implementation with 1e-8 absolute tolerance threshold
2. **Autograd Compatibility**: Verifies that gradients can be computed correctly
3. **Performance Benchmarks**: Measures runtime and calculates speedup at different sample sizes

## Quick Start

Run the complete test suite:

```bash
cd implementation
./tools/run_full_test_suite.sh
```

Or run steps individually:

```bash
# 1. Discover functions
python tools/discover_functions.py

# 2. Generate test data
python tools/generate_test_data.py

# 3. Run tests
python tools/run_comprehensive_tests.py

# 4. Generate reports
python tools/generate_reports.py --input reports/test_results.json
```

## Tools

### 1. Function Discovery (`tools/discover_functions.py`)

Discovers all functions from both implementations and creates an inventory:
- Functions in both (to test)
- Functions only in reference (missing in PyTorch)
- Functions only in PyTorch (extras)

**Output**: `reports/function_inventory.json`

### 2. Test Data Generation (`tools/generate_test_data.py`)

Generates test data for different function categories:
- Oceanographic domain (SA, CT, p, lat)
- Ice functions (t_ice, p_ice)
- Salinity conversions (SP)
- Temperature conversions (t, CT, pt)
- Freezing functions
- Enthalpy/entropy functions
- Density functions
- Geostrophy functions
- Profile functions (multi-dimensional)

Test data is generated at multiple sample sizes: 10, 100, 1000, 10000

**Output**: `tests/test_data/*.npz` files

### 3. Test Execution (`tools/run_comprehensive_tests.py`)

Runs comprehensive tests for all functions:

**Options:**
- `--functions`: Test specific functions (comma-separated)
- `--module`: Test specific module (conversions, density, etc.)
- `--skip-parity`: Skip numerical parity tests
- `--skip-grad`: Skip autograd tests
- `--skip-benchmark`: Skip performance benchmarks
- `--sample-sizes`: Custom sample sizes (comma-separated)
- `--output-dir`: Output directory for results

**Output**: `reports/test_results.json`

### 4. Report Generation (`tools/generate_reports.py`)

Generates comprehensive reports from test results:

**Output Files:**
- `reports/comprehensive_test_report.md`: Main markdown report with tables
- `reports/numerical_parity.csv`: Detailed numerical error data
- `reports/autograd_compatibility.csv`: Autograd test results
- `reports/performance_benchmarks.csv`: Performance data
- `reports/test_summary.json`: Machine-readable summary for CI/CD

## Test Results Interpretation

### Numerical Parity

- **PASS**: Max absolute error ≤ 1e-8
- **FAIL**: Max absolute error > 1e-8 (needs revision)
- **ERROR**: Function raised an exception
- **ERROR_NAN**: Result contains NaN values

### Autograd Compatibility

- **PASS**: Gradients computed successfully
- **FAIL**: Gradients failed or are non-finite
- **ERROR**: Function raised an exception during gradient computation

### Performance Benchmarks

Reports include:
- PyTorch execution time (median of 10 runs)
- Reference execution time (median of 10 runs)
- Speedup ratio (reference_time / torch_time)

## Reports

### Markdown Report

The main markdown report (`comprehensive_test_report.md`) includes:

1. **Executive Summary**
   - Total functions tested
   - Test statistics (passed/failed/errors)
   - Average speedup
   - Functions needing attention

2. **Detailed Tables**
   - Numerical parity results with error metrics
   - Autograd compatibility results
   - Performance benchmarks at all sample sizes

3. **Per-Module Breakdown**
   - Statistics by module (conversions, density, energy, etc.)

### CSV Reports

CSV files are provided for easy analysis in spreadsheets:
- Filter and sort by error magnitude
- Identify functions with performance issues
- Track autograd compatibility across functions

### JSON Summary

The JSON summary (`test_summary.json`) provides:
- Machine-readable test statistics
- Lists of failing functions
- Average speedup metrics

Suitable for CI/CD integration and automated reporting.

## Error Thresholds

- **Absolute Tolerance**: 1e-8 (as specified)
- **Relative Tolerance**: 1e-6 (for functions with large values)

Functions exceeding these thresholds are flagged in reports as needing revision.

## Integration with Existing Tests

The comprehensive testing suite integrates with the existing test infrastructure:

- Uses `tests/conftest.py` fixtures and helpers
- Respects pytest markers (`@pytest.mark.comprehensive`, etc.)
- Compatible with existing test structure

## Limitations and Notes

1. **Test Data**: Some functions may require specific input ranges or data files (e.g., SAAR data). The test framework attempts to handle these cases but may need manual configuration for edge cases.

2. **Performance**: Large sample sizes (10000) may be slow for some functions. The framework handles timeouts gracefully.

3. **Function Signatures**: Functions with complex signatures or optional parameters may need manual test input preparation.

4. **Reference Availability**: The reference GSW implementation must be available at `source_files/gsw/` or configured path.

## Future Enhancements

Potential improvements:
- GPU performance benchmarks
- Memory usage profiling
- Gradient accuracy verification (not just existence)
- Automated test input generation for edge cases
- Integration with CI/CD pipeline
