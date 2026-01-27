#!/bin/bash
# Convenience script to run the full comprehensive test suite

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "GSW PyTorch Comprehensive Test Suite"
echo "=========================================="
echo ""

# Step 1: Discover functions
echo "Step 1: Discovering functions..."
python tools/discover_functions.py
echo ""

# Step 2: Generate test data
echo "Step 2: Generating test data..."
python tools/generate_test_data.py
echo ""

# Step 3: Run comprehensive tests
echo "Step 3: Running comprehensive tests..."
echo "This may take a while..."
python tools/run_comprehensive_tests.py
echo ""

# Step 4: Generate reports
echo "Step 4: Generating reports..."
if [ -f "reports/test_results.json" ]; then
    python tools/generate_reports.py --input reports/test_results.json --output-dir reports
    echo ""
    echo "=========================================="
    echo "Test suite complete!"
    echo "=========================================="
    echo ""
    echo "Reports generated in: reports/"
    echo "  - comprehensive_test_report.md"
    echo "  - numerical_parity.csv"
    echo "  - autograd_compatibility.csv"
    echo "  - performance_benchmarks.csv"
    echo "  - test_summary.json"
else
    echo "ERROR: test_results.json not found. Tests may have failed."
    exit 1
fi
