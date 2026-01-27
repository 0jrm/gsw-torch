"""
Generate comprehensive test reports in markdown and CSV formats.

Reads test results JSON and generates:
1. Markdown report with tables and summaries
2. CSV reports for numerical parity, autograd, and performance
3. JSON summary for CI/CD integration
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def load_test_results(results_path: Path) -> Dict[str, Any]:
    """Load test results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def generate_markdown_report(results: Dict[str, Any], output_path: Path):
    """
    Generate comprehensive markdown report.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Test results dictionary
    output_path : Path
        Output file path
    """
    lines = []
    
    # Header
    lines.append("# GSW PyTorch Comprehensive Test Report")
    lines.append("")
    lines.append("This report contains comprehensive test results comparing the PyTorch")
    lines.append("implementation against the reference GSW implementation.")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    
    total_functions = len(results)
    parity_results = []
    autograd_results = []
    benchmark_results = []
    
    functions_with_errors = []
    functions_failing_parity = []
    functions_failing_autograd = []
    
    for func_name, func_results in results.items():
        if "error" in func_results:
            functions_with_errors.append(func_name)
            continue
        
        # Collect parity results
        if "parity" in func_results:
            for parity in func_results["parity"]:
                parity_results.append((func_name, parity))
                if parity.get("status") == "FAIL":
                    functions_failing_parity.append(func_name)
        
        # Collect autograd results
        if "autograd" in func_results and func_results["autograd"]:
            autograd = func_results["autograd"]
            autograd_results.append((func_name, autograd))
            if autograd.get("status") == "FAIL":
                functions_failing_autograd.append(func_name)
        
        # Collect benchmark results
        if "benchmark" in func_results:
            for bench in func_results["benchmark"]:
                benchmark_results.append((func_name, bench))
    
    # Summary statistics
    parity_passed = sum(1 for _, p in parity_results if p.get("status") == "PASS")
    parity_failed = sum(1 for _, p in parity_results if p.get("status") == "FAIL")
    parity_errors = sum(1 for _, p in parity_results if p.get("status") in ["ERROR", "ERROR_NAN"])
    
    autograd_passed = sum(1 for _, a in autograd_results if a.get("status") == "PASS")
    autograd_failed = sum(1 for _, a in autograd_results if a.get("status") == "FAIL")
    
    benchmark_count = len(benchmark_results)
    speedups = [
        b[1].get("speedup")
        for b in benchmark_results
        if b[1].get("speedup") is not None and b[1].get("speedup") > 0
    ]
    avg_speedup = sum(speedups) / len(speedups) if speedups else None
    
    lines.append(f"- **Total Functions Tested**: {total_functions}")
    lines.append(f"- **Functions with Errors**: {len(functions_with_errors)}")
    lines.append("")
    lines.append("### Numerical Parity Tests")
    lines.append(f"- **Total Tests**: {len(parity_results)}")
    lines.append(f"- **Passed**: {parity_passed}")
    lines.append(f"- **Failed** (error > 1e-8): {parity_failed}")
    lines.append(f"- **Errors**: {parity_errors}")
    lines.append("")
    lines.append("### Autograd Compatibility Tests")
    lines.append(f"- **Total Tests**: {len(autograd_results)}")
    lines.append(f"- **Passed**: {autograd_passed}")
    lines.append(f"- **Failed**: {autograd_failed}")
    lines.append("")
    lines.append("### Performance Benchmarks")
    lines.append(f"- **Total Benchmarks**: {benchmark_count}")
    if avg_speedup:
        lines.append(f"- **Average Speedup**: {avg_speedup:.2f}x")
    lines.append("")
    
    # Functions needing attention
    if functions_failing_parity or functions_failing_autograd or functions_with_errors:
        lines.append("### Functions Needing Attention")
        lines.append("")
        
        if functions_failing_parity:
            lines.append("**Numerical Parity Issues:**")
            for func in sorted(set(functions_failing_parity)):
                lines.append(f"- {func}")
            lines.append("")
        
        if functions_failing_autograd:
            lines.append("**Autograd Issues:**")
            for func in sorted(set(functions_failing_autograd)):
                lines.append(f"- {func}")
            lines.append("")
        
        if functions_with_errors:
            lines.append("**Functions with Errors:**")
            for func in sorted(functions_with_errors):
                lines.append(f"- {func}")
            lines.append("")
    
    # Detailed Tables
    lines.append("## Detailed Results")
    lines.append("")
    
    # Numerical Parity Table
    lines.append("### Numerical Parity Results")
    lines.append("")
    lines.append("| Function | Sample Size | Max Abs Error | Max Rel Error | Status |")
    lines.append("|----------|-------------|---------------|---------------|--------|")
    
    for func_name, parity in sorted(parity_results, key=lambda x: (x[0], x[1].get("sample_size", 0))):
        max_abs = parity.get("max_abs_error")
        max_rel = parity.get("max_rel_error")
        status = parity.get("status", "UNKNOWN")
        sample_size = parity.get("sample_size", "N/A")
        
        max_abs_str = f"{max_abs:.2e}" if max_abs is not None else "N/A"
        max_rel_str = f"{max_rel:.2e}" if max_rel is not None else "N/A"
        
        lines.append(
            f"| {func_name} | {sample_size} | {max_abs_str} | {max_rel_str} | {status} |"
        )
    
    lines.append("")
    
    # Autograd Compatibility Table
    lines.append("### Autograd Compatibility Results")
    lines.append("")
    lines.append("| Function | Gradients Work | Num Differentiable Inputs | Issues |")
    lines.append("|----------|----------------|---------------------------|--------|")
    
    for func_name, autograd in sorted(autograd_results, key=lambda x: x[0]):
        gradients_work = "Yes" if autograd.get("gradients_work") else "No"
        num_grads = autograd.get("num_differentiable_inputs", 0)
        issues = "; ".join(autograd.get("issues", [])) or "None"
        
        lines.append(
            f"| {func_name} | {gradients_work} | {num_grads} | {issues} |"
        )
    
    lines.append("")
    
    # Performance Table
    lines.append("### Performance Benchmarks")
    lines.append("")
    lines.append("| Function | Sample Size | PyTorch Time (s) | Reference Time (s) | Speedup |")
    lines.append("|----------|-------------|------------------|-------------------|---------|")
    
    for func_name, bench in sorted(benchmark_results, key=lambda x: (x[0], x[1].get("sample_size", 0))):
        sample_size = bench.get("sample_size", "N/A")
        torch_time = bench.get("torch_time")
        ref_time = bench.get("ref_time")
        speedup = bench.get("speedup")
        
        torch_time_str = f"{torch_time:.6f}" if torch_time is not None else "N/A"
        ref_time_str = f"{ref_time:.6f}" if ref_time is not None else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
        
        lines.append(
            f"| {func_name} | {sample_size} | {torch_time_str} | {ref_time_str} | {speedup_str} |"
        )
    
    lines.append("")
    
    # Per-module breakdown
    lines.append("## Per-Module Breakdown")
    lines.append("")
    
    modules = {}
    for func_name, func_results in results.items():
        if "error" in func_results:
            continue
        
        # Determine module from function name or results
        module = "unknown"
        if "parity" in func_results and func_results["parity"]:
            # Try to infer module from function name
            if any(x in func_name.lower() for x in ["ct_", "t_", "pt_", "sa_", "sp_", "entropy"]):
                module = "conversions"
            elif any(x in func_name.lower() for x in ["rho", "density", "specvol", "sigma", "alpha", "beta"]):
                module = "density"
            elif any(x in func_name.lower() for x in ["enthalpy", "energy"]):
                module = "energy"
            elif "freezing" in func_name.lower():
                module = "freezing"
            elif "ice" in func_name.lower():
                module = "ice"
            elif any(x in func_name.lower() for x in ["nsquared", "turner", "stability"]):
                module = "stability"
            elif any(x in func_name.lower() for x in ["geo", "distance", "f(", "unwrap"]):
                module = "geostrophy"
            else:
                module = "utility"
        
        if module not in modules:
            modules[module] = {"total": 0, "parity_pass": 0, "parity_fail": 0, "autograd_pass": 0, "autograd_fail": 0}
        
        modules[module]["total"] += 1
        
        if "parity" in func_results:
            for parity in func_results["parity"]:
                if parity.get("status") == "PASS":
                    modules[module]["parity_pass"] += 1
                elif parity.get("status") == "FAIL":
                    modules[module]["parity_fail"] += 1
        
        if "autograd" in func_results and func_results["autograd"]:
            autograd = func_results["autograd"]
            if autograd.get("status") == "PASS":
                modules[module]["autograd_pass"] += 1
            elif autograd.get("status") == "FAIL":
                modules[module]["autograd_fail"] += 1
    
    lines.append("| Module | Total Functions | Parity Pass | Parity Fail | Autograd Pass | Autograd Fail |")
    lines.append("|--------|----------------|-------------|-------------|--------------|---------------|")
    
    for module in sorted(modules.keys()):
        mod_data = modules[module]
        lines.append(
            f"| {module} | {mod_data['total']} | {mod_data['parity_pass']} | "
            f"{mod_data['parity_fail']} | {mod_data['autograd_pass']} | {mod_data['autograd_fail']} |"
        )
    
    lines.append("")
    
    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Markdown report saved to: {output_path}")


def generate_csv_reports(results: Dict[str, Any], output_dir: Path):
    """
    Generate CSV reports for numerical parity, autograd, and performance.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Test results dictionary
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Numerical Parity CSV
    parity_file = output_dir / "numerical_parity.csv"
    with open(parity_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Function", "Sample Size", "Max Absolute Error", "Max Relative Error",
            "Status", "Error Message"
        ])
        
        for func_name, func_results in sorted(results.items()):
            if "parity" in func_results:
                for parity in func_results["parity"]:
                    writer.writerow([
                        func_name,
                        parity.get("sample_size", ""),
                        parity.get("max_abs_error", ""),
                        parity.get("max_rel_error", ""),
                        parity.get("status", ""),
                        parity.get("error", ""),
                    ])
    
    print(f"Numerical parity CSV saved to: {parity_file}")
    
    # Autograd Compatibility CSV
    autograd_file = output_dir / "autograd_compatibility.csv"
    with open(autograd_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Function", "Gradients Work", "Num Differentiable Inputs",
            "Status", "Issues", "Error Message"
        ])
        
        for func_name, func_results in sorted(results.items()):
            if "autograd" in func_results and func_results["autograd"]:
                autograd = func_results["autograd"]
                issues = "; ".join(autograd.get("issues", [])) or ""
                writer.writerow([
                    func_name,
                    "Yes" if autograd.get("gradients_work") else "No",
                    autograd.get("num_differentiable_inputs", 0),
                    autograd.get("status", ""),
                    issues,
                    autograd.get("error", ""),
                ])
    
    print(f"Autograd compatibility CSV saved to: {autograd_file}")
    
    # Performance Benchmarks CSV
    benchmark_file = output_dir / "performance_benchmarks.csv"
    with open(benchmark_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Function", "Sample Size", "PyTorch Time (s)", "Reference Time (s)",
            "Speedup", "Status", "Error Message"
        ])
        
        for func_name, func_results in sorted(results.items()):
            if "benchmark" in func_results:
                for bench in func_results["benchmark"]:
                    writer.writerow([
                        func_name,
                        bench.get("sample_size", ""),
                        bench.get("torch_time", ""),
                        bench.get("ref_time", ""),
                        bench.get("speedup", ""),
                        bench.get("status", ""),
                        bench.get("error", ""),
                    ])
    
    print(f"Performance benchmarks CSV saved to: {benchmark_file}")


def generate_summary_json(results: Dict[str, Any], output_path: Path):
    """
    Generate JSON summary for CI/CD integration.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Test results dictionary
    output_path : Path
        Output file path
    """
    summary = {
        "total_functions": len(results),
        "parity": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "functions_failing": [],
        },
        "autograd": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "functions_failing": [],
        },
        "benchmark": {
            "total_tests": 0,
            "average_speedup": None,
        },
    }
    
    for func_name, func_results in results.items():
        if "error" in func_results:
            continue
        
        # Parity statistics
        if "parity" in func_results:
            for parity in func_results["parity"]:
                summary["parity"]["total_tests"] += 1
                status = parity.get("status", "")
                if status == "PASS":
                    summary["parity"]["passed"] += 1
                elif status == "FAIL":
                    summary["parity"]["failed"] += 1
                    if func_name not in summary["parity"]["functions_failing"]:
                        summary["parity"]["functions_failing"].append(func_name)
                elif status in ["ERROR", "ERROR_NAN"]:
                    summary["parity"]["errors"] += 1
        
        # Autograd statistics
        if "autograd" in func_results and func_results["autograd"]:
            autograd = func_results["autograd"]
            summary["autograd"]["total_tests"] += 1
            status = autograd.get("status", "")
            if status == "PASS":
                summary["autograd"]["passed"] += 1
            elif status == "FAIL":
                summary["autograd"]["failed"] += 1
                if func_name not in summary["autograd"]["functions_failing"]:
                    summary["autograd"]["functions_failing"].append(func_name)
        
        # Benchmark statistics
        if "benchmark" in func_results:
            for bench in func_results["benchmark"]:
                summary["benchmark"]["total_tests"] += 1
                speedup = bench.get("speedup")
                if speedup is not None and speedup > 0:
                    if summary["benchmark"]["average_speedup"] is None:
                        summary["benchmark"]["average_speedup"] = []
                    summary["benchmark"]["average_speedup"].append(speedup)
    
    # Calculate average speedup
    if summary["benchmark"]["average_speedup"]:
        avg = sum(summary["benchmark"]["average_speedup"]) / len(summary["benchmark"]["average_speedup"])
        summary["benchmark"]["average_speedup"] = avg
    
    # Sort failing function lists
    summary["parity"]["functions_failing"].sort()
    summary["autograd"]["functions_failing"].sort()
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary JSON saved to: {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive test reports"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to test results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output reports",
    )
    
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.input)
    if not results_path.exists():
        print(f"ERROR: Test results file not found: {results_path}")
        return
    
    print(f"Loading test results from: {results_path}")
    results = load_test_results(results_path)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports
    print("\nGenerating markdown report...")
    markdown_file = output_dir / "comprehensive_test_report.md"
    generate_markdown_report(results, markdown_file)
    
    print("\nGenerating CSV reports...")
    generate_csv_reports(results, output_dir)
    
    print("\nGenerating summary JSON...")
    summary_file = output_dir / "test_summary.json"
    generate_summary_json(results, summary_file)
    
    print("\nReport generation complete!")


if __name__ == "__main__":
    main()
