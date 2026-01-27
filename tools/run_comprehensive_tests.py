"""
Comprehensive test execution framework.

Runs numerical parity, autograd, and performance tests for all GSW functions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

import sys
from pathlib import Path

# Add tools directory to path for imports
tools_dir = Path(__file__).parent
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from test_runner import (
    benchmark_performance,
    get_pytorch_gsw,
    get_reference_gsw,
    test_autograd,
    test_numerical_parity,
)


# Default sample sizes
DEFAULT_SAMPLE_SIZES = [10, 100, 1000, 10000]


def load_function_inventory(inventory_path: Path) -> Dict[str, Any]:
    """Load function inventory from JSON file."""
    with open(inventory_path, "r") as f:
        return json.load(f)


def load_test_data(data_dir: Path, category: str, sample_size: int) -> Dict[str, np.ndarray]:
    """Load test data for a category and sample size."""
    data_file = data_dir / f"{category}_{sample_size}.npz"
    if data_file.exists():
        return dict(np.load(data_file))
    return {}


def determine_test_data_category(func_name: str) -> str:
    """
    Determine which test data category to use for a function.
    
    Parameters
    ----------
    func_name : str
        Function name
        
    Returns
    -------
    str
        Test data category name
    """
    func_lower = func_name.lower()
    
    # Ice functions
    if "ice" in func_lower:
        return "ice"
    
    # Freezing functions
    if "freezing" in func_lower:
        return "freezing"
    
    # Salinity functions
    if any(x in func_lower for x in ["sa_from", "sp_from", "salinity"]):
        return "salinity"
    
    # Temperature functions
    if any(x in func_lower for x in ["t_from", "ct_from", "pt_from", "t90", "t68"]):
        return "temperature"
    
    # Enthalpy functions
    if "enthalpy" in func_lower:
        return "enthalpy"
    
    # Entropy functions
    if "entropy" in func_lower:
        return "entropy"
    
    # Density functions
    if any(x in func_lower for x in ["rho", "density", "specvol", "sigma", "alpha", "beta"]):
        return "density"
    
    # Geostrophy functions
    if any(x in func_lower for x in ["geo", "distance", "f(", "unwrap"]):
        return "geostrophy"
    
    # Profile functions (multi-dimensional)
    if any(x in func_lower for x in ["nsquared", "turner", "interp"]):
        return "profile"
    
    # Default: oceanographic
    return "oceanographic"


def get_function_signature(func) -> List[str]:
    """Get function parameter names."""
    import inspect
    try:
        sig = inspect.signature(func)
        return list(sig.parameters.keys())
    except (ValueError, TypeError):
        return []


def prepare_test_inputs(
    func_name: str,
    params: List[str],
    test_data: Dict[str, np.ndarray],
    sample_size: int
) -> Optional[tuple]:
    """
    Prepare test inputs for a function based on its parameters.
    
    Parameters
    ----------
    func_name : str
        Function name
    params : List[str]
        Function parameter names
    test_data : Dict[str, np.ndarray]
        Available test data
    sample_size : int
        Sample size
        
    Returns
    -------
    Optional[tuple]
        Tuple of test inputs, or None if cannot prepare
    """
    inputs = []
    
    # Functions that need profile data (2D arrays)
    profile_functions = {
        "nsquared", "turner_rsubrho", "ipv_vs_fnsquared_ratio",
        "sa_ct_interp", "tracer_ct_interp"
    }
    
    # Functions that need 1D arrays for lon/lat
    geostrophy_functions = {"distance", "geostrophic_velocity"}
    
    use_profile_data = func_name.lower() in profile_functions
    use_geostrophy_data = func_name.lower() in geostrophy_functions
    
    # Map parameter names to test data keys
    param_map = {
        "sa": "SA",
        "ct": "CT",
        "p": "p",
        "lat": "lat",
        "lon": "lon",
        "t": "t",
        "t68": "t68",
        "pt": "pt",
        "sp": "SP",
        "t_ice": "t_ice",
        "p_ice": "p_ice",
        "saturation_fraction": "saturation_fraction",
        "h": "h",
        "entropy": "entropy",
        "rho": "rho",
        "geo_strf": "geo_strf",
    }
    
    for param in params:
        # Skip optional parameters with defaults (we'll test with defaults)
        param_orig = param
        if "=" in param:
            param = param.split("=")[0].strip()
        
        param_lower = param.lower()
        
        # Try to find matching test data
        found = False
        for key, data_key in param_map.items():
            if key in param_lower and data_key in test_data:
                data = test_data[data_key]
                
                # Handle profile functions - need 2D arrays
                if use_profile_data and data.ndim == 1:
                    # Reshape to 2D: (n_profiles, n_levels)
                    n_profiles = 3 if sample_size >= 1000 else 5
                    if sample_size >= len(data):
                        # Reshape to 2D
                        data = data.reshape(n_profiles, -1)
                    else:
                        # Take subset and reshape
                        data = data[:sample_size].reshape(n_profiles, -1)
                
                # Handle geostrophy functions - lon/lat need to be 1D
                if use_geostrophy_data and param_lower in ["lon", "lat"]:
                    if data.ndim > 1:
                        data = data.flatten()[:sample_size]
                    elif len(data) > sample_size:
                        data = data[:sample_size]
                
                inputs.append(data)
                found = True
                break
        
        if not found:
            # Try direct match
            if param.upper() in test_data:
                inputs.append(test_data[param.upper()])
            elif param in test_data:
                inputs.append(test_data[param])
            else:
                # Use default value based on parameter name
                if "axis" in param_lower:
                    inputs.append(0)  # Default axis
                elif "p_ref" in param_lower:
                    inputs.append(0.0)  # Default reference pressure
                elif "saturation" in param_lower:
                    inputs.append(0.0)  # Default saturation fraction
                elif "lat" in param_lower and not use_geostrophy_data:
                    # Optional lat parameter for some functions
                    if "lat" in test_data:
                        inputs.append(test_data["lat"])
                    else:
                        inputs.append(None)
                elif param_lower == "tracer" and "CT" in test_data:
                    # For tracer_ct_interp, use CT as tracer
                    inputs.append(test_data["CT"])
                elif param_lower == "p_i" and "p" in test_data:
                    # For interpolation functions, use subset of p
                    p_data = test_data["p"]
                    if p_data.ndim == 1:
                        inputs.append(p_data[:sample_size//2])
                    else:
                        inputs.append(p_data[0, :sample_size//2])
                else:
                    # Check if it's a keyword-only parameter with default
                    if "=" in param_orig:
                        # Has default, skip it
                        continue
                    # Cannot determine input
                    return None
    
    return tuple(inputs) if inputs else None


def run_tests_for_function(
    func_name: str,
    ref_func,
    torch_func,
    sample_sizes: List[int],
    test_data_dir: Path,
    skip_parity: bool = False,
    skip_grad: bool = False,
    skip_benchmark: bool = False
) -> Dict[str, Any]:
    """
    Run all tests for a single function.
    
    Parameters
    ----------
    func_name : str
        Function name
    ref_func : callable
        Reference implementation function
    torch_func : callable
        PyTorch implementation function
    sample_sizes : List[int]
        List of sample sizes to test
    test_data_dir : Path
        Directory containing test data
    skip_parity : bool
        Skip numerical parity tests
    skip_grad : bool
        Skip autograd tests
    skip_benchmark : bool
        Skip performance benchmarks
        
    Returns
    -------
    Dict[str, Any]
        Test results for the function
    """
    results = {
        "function": func_name,
        "parity": [],
        "autograd": None,
        "benchmark": [],
    }
    
    # Determine test data category
    category = determine_test_data_category(func_name)
    
    # Get function parameters
    ref_params = get_function_signature(ref_func)
    torch_params = get_function_signature(torch_func)
    
    # Use reference parameters as primary
    params = ref_params if ref_params else torch_params
    
    # Test with smallest sample size first to check if function works
    test_size = sample_sizes[0] if sample_sizes else 10
    test_data = load_test_data(test_data_dir, category, test_size)
    
    if not test_data:
        # Try oceanographic as fallback
        test_data = load_test_data(test_data_dir, "oceanographic", test_size)
    
    # Prepare test inputs
    test_inputs = prepare_test_inputs(func_name, params, test_data, test_size)
    
    if test_inputs is None:
        results["error"] = "Could not prepare test inputs"
        return results
    
    # Run numerical parity tests
    if not skip_parity:
        for size in sample_sizes:
            test_data = load_test_data(test_data_dir, category, size)
            if not test_data:
                test_data = load_test_data(test_data_dir, "oceanographic", size)
            
            test_inputs = prepare_test_inputs(func_name, params, test_data, size)
            if test_inputs:
                parity_result = test_numerical_parity(
                    func_name, ref_func, torch_func, test_inputs, size
                )
                results["parity"].append(parity_result)
    
    # Run autograd tests
    if not skip_grad:
        test_data = load_test_data(test_data_dir, category, test_size)
        if not test_data:
            test_data = load_test_data(test_data_dir, "oceanographic", test_size)
        
        test_inputs = prepare_test_inputs(func_name, params, test_data, test_size)
        if test_inputs:
            # Determine which inputs should be differentiable
            # Most GSW functions have SA, CT, p as differentiable inputs
            differentiable_indices = []
            for i, param in enumerate(params):
                param_lower = param.lower()
                if any(x in param_lower for x in ["sa", "ct", "p", "t", "pt", "h", "entropy", "rho"]):
                    differentiable_indices.append(i)
            
            autograd_result = test_autograd(
                func_name, torch_func, test_inputs, differentiable_indices
            )
            results["autograd"] = autograd_result
    
    # Run performance benchmarks
    if not skip_benchmark:
        for size in sample_sizes:
            test_data = load_test_data(test_data_dir, category, size)
            if not test_data:
                test_data = load_test_data(test_data_dir, "oceanographic", size)
            
            test_inputs = prepare_test_inputs(func_name, params, test_data, size)
            if test_inputs:
                benchmark_result = benchmark_performance(
                    func_name, ref_func, torch_func, test_inputs, size
                )
                results["benchmark"].append(benchmark_result)
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for GSW PyTorch implementation"
    )
    parser.add_argument(
        "--functions",
        type=str,
        help="Comma-separated list of specific functions to test",
    )
    parser.add_argument(
        "--module",
        type=str,
        help="Test specific module (conversions, density, etc.)",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip numerical parity tests",
    )
    parser.add_argument(
        "--skip-grad",
        action="store_true",
        help="Skip autograd tests",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip performance benchmarks",
    )
    parser.add_argument(
        "--sample-sizes",
        type=str,
        help="Comma-separated list of sample sizes (e.g., '10,100,1000')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output reports",
    )
    parser.add_argument(
        "--inventory",
        type=str,
        help="Path to function inventory JSON file",
    )
    
    args = parser.parse_args()
    
    # Determine sample sizes
    if args.sample_sizes:
        sample_sizes = [int(s.strip()) for s in args.sample_sizes.split(",")]
    else:
        sample_sizes = DEFAULT_SAMPLE_SIZES
    
    # Get implementations
    print("Loading reference GSW...")
    gsw_ref = get_reference_gsw()
    if gsw_ref is None:
        print("ERROR: Could not load reference GSW")
        sys.exit(1)
    
    print("Loading PyTorch GSW...")
    gsw_torch = get_pytorch_gsw()
    if gsw_torch is None:
        print("ERROR: Could not load PyTorch GSW")
        sys.exit(1)
    
    # Load function inventory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if args.inventory:
        inventory_path = Path(args.inventory)
    else:
        inventory_path = project_root / "reports" / "function_inventory.json"
    
    if not inventory_path.exists():
        print(f"ERROR: Function inventory not found at {inventory_path}")
        print("Please run tools/discover_functions.py first")
        sys.exit(1)
    
    inventory = load_function_inventory(inventory_path)
    
    # Determine which functions to test
    functions_to_test = list(inventory["functions_in_both"].keys())
    
    if args.functions:
        requested = set(f.strip() for f in args.functions.split(","))
        functions_to_test = [f for f in functions_to_test if f in requested]
    
    if args.module:
        # Filter by module
        module_prefix = args.module.lower()
        functions_to_test = [
            f for f in functions_to_test
            if module_prefix in inventory["functions_in_both"][f]["reference"]["module"].lower()
        ]
    
    print(f"\nTesting {len(functions_to_test)} functions...")
    print(f"Sample sizes: {sample_sizes}")
    
    # Test data directory
    test_data_dir = project_root / "tests" / "test_data"
    
    # Run tests
    all_results = {}
    
    for i, func_name in enumerate(functions_to_test, 1):
        print(f"\n[{i}/{len(functions_to_test)}] Testing {func_name}...")
        
        try:
            ref_func = getattr(gsw_ref, func_name)
            torch_func = getattr(gsw_torch, func_name)
        except AttributeError:
            print(f"  ERROR: Function {func_name} not found in one of the implementations")
            all_results[func_name] = {"error": "Function not found"}
            continue
        
        try:
            results = run_tests_for_function(
                func_name,
                ref_func,
                torch_func,
                sample_sizes,
                test_data_dir,
                skip_parity=args.skip_parity,
                skip_grad=args.skip_grad,
                skip_benchmark=args.skip_benchmark,
            )
            all_results[func_name] = results
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            all_results[func_name] = {"error": str(e)}
    
    # Save results
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
    print(f"\nTotal functions tested: {len(all_results)}")


if __name__ == "__main__":
    main()
