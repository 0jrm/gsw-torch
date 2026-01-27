# Comprehensive Testing Tools

This directory contains tools for comprehensive testing of the GSW PyTorch implementation against the reference GSW implementation.

## Tools Overview

### 1. `discover_functions.py`
Discovers all functions from both the reference GSW and PyTorch implementations, creating an inventory of:
- Functions in both implementations (to test)
- Functions only in reference (missing in PyTorch)
- Functions only in PyTorch (extras)

**Usage:**
```bash
cd implementation
python tools/discover_functions.py
```

Output: `reports/function_inventory.json`

### 2. `generate_test_data.py`
Generates test data for different function categories at multiple sample sizes.

**Usage:**
```bash
cd implementation
python tools/generate_test_data.py
```

Output: `tests/test_data/*.npz` files

### 3. `run_comprehensive_tests.py`
Runs comprehensive tests for all GSW functions:
- Numerical parity tests (error > 1e-8 flagged)
- Autograd compatibility tests
- Performance benchmarks

**Usage:**
```bash
cd implementation

# Run all tests
python tools/run_comprehensive_tests.py

# Test specific functions
python tools/run_comprehensive_tests.py --functions CT_from_t,rho,enthalpy

# Test specific module
python tools/run_comprehensive_tests.py --module conversions

# Skip certain test types
python tools/run_comprehensive_tests.py --skip-benchmark

# Custom sample sizes
python tools/run_comprehensive_tests.py --sample-sizes 10,100,1000
```

Output: `reports/test_results.json`

### 4. `generate_reports.py`
Generates comprehensive reports from test results:
- Markdown report with tables and summaries
- CSV reports for numerical parity, autograd, and performance
- JSON summary for CI/CD integration

**Usage:**
```bash
cd implementation
python tools/generate_reports.py --input reports/test_results.json --output-dir reports
```

Output:
- `reports/comprehensive_test_report.md`
- `reports/numerical_parity.csv`
- `reports/autograd_compatibility.csv`
- `reports/performance_benchmarks.csv`
- `reports/test_summary.json`

## Complete Workflow

1. **Discover functions:**
   ```bash
   python tools/discover_functions.py
   ```

2. **Generate test data:**
   ```bash
   python tools/generate_test_data.py
   ```

3. **Run comprehensive tests:**
   ```bash
   python tools/run_comprehensive_tests.py
   ```

4. **Generate reports:**
   ```bash
   python tools/generate_reports.py --input reports/test_results.json
   ```

## Test Results

The comprehensive test suite measures:

1. **Numerical Parity**: Compares PyTorch implementation against reference with 1e-8 absolute tolerance threshold
2. **Autograd Compatibility**: Verifies that gradients can be computed correctly
3. **Performance**: Benchmarks runtime and calculates speedup at different sample sizes (10, 100, 1000, 10000)

Functions with errors > 1e-8 are flagged in the reports as needing revision.

## Reports

### Markdown Report
The main markdown report (`comprehensive_test_report.md`) includes:
- Executive summary with statistics
- Detailed tables for numerical parity, autograd, and performance
- Per-module breakdown
- Functions needing attention

### CSV Reports
CSV files are provided for easy analysis in spreadsheets:
- `numerical_parity.csv`: Detailed numerical error data
- `autograd_compatibility.csv`: Autograd test results
- `performance_benchmarks.csv`: Performance data at all sample sizes

### JSON Summary
`test_summary.json` provides machine-readable summary for CI/CD integration.
