"""
Pytest configuration and fixtures for gsw_torch tests.
"""

import os

import numpy as np
import pytest
import torch

# Try to import reference gsw for comparison
try:
    import sys

    ref_path = os.path.join(os.path.dirname(__file__), "../reference/gsw")
    if os.path.exists(os.path.dirname(ref_path)):
        sys.path.insert(0, os.path.dirname(ref_path))
        import gsw as gsw_ref

        HAS_REFERENCE = True
    else:
        HAS_REFERENCE = False
except ImportError:
    HAS_REFERENCE = False


@pytest.fixture
def reference_gsw():
    """Fixture providing reference gsw implementation if available."""
    if not HAS_REFERENCE:
        pytest.skip("Reference gsw not available")
    return gsw_ref


@pytest.fixture
def oceanographic_domain():
    """
    Generate test data in oceanographic domain.

    Returns
    -------
    dict
        Dictionary with SA, CT, p, lat arrays
    """
    # Typical oceanographic ranges
    SA = np.linspace(30, 40, 10)  # Absolute Salinity, g/kg
    CT = np.linspace(0, 25, 10)  # Conservative Temperature, degrees C
    p = np.linspace(0, 5000, 10)  # Pressure, dbar
    lat = np.linspace(-60, 60, 10)  # Latitude, degrees

    return {
        "SA": SA,
        "CT": CT,
        "p": p,
        "lat": lat,
    }


@pytest.fixture
def check_values():
    """
    Load reference check values from gsw_cv_v3_0.npz if available.

    Returns
    -------
    dict or None
        Dictionary of check values, or None if not available
    """
    cv_path = os.path.join(
        os.path.dirname(__file__), "../reference/gsw/tests/gsw_cv_v3_0.npz"
    )
    if os.path.exists(cv_path):
        return dict(np.load(cv_path))
    return None


def assert_torch_allclose(actual, expected, rtol=1e-6, atol=1e-8, equal_nan=True):
    """
    Assert that two torch tensors are close.

    Parameters
    ----------
    actual : torch.Tensor
        Actual values
    expected : torch.Tensor or np.ndarray
        Expected values
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    equal_nan : bool
        Whether to treat NaN as equal
    """
    if isinstance(expected, np.ndarray):
        expected = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)

    if not torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan):
        diff = torch.abs(actual - expected)
        max_diff = torch.max(diff)
        max_rel_diff = torch.max(diff / (torch.abs(expected) + atol))
        raise AssertionError(
            f"Tensors not close: max_diff={max_diff}, max_rel_diff={max_rel_diff}"
        )


def calculate_errors(torch_result, ref_result, atol=1e-8):
    """
    Calculate absolute and relative errors between torch and reference results.
    
    Helper function for comprehensive testing.
    
    Parameters
    ----------
    torch_result : torch.Tensor or tuple
        PyTorch implementation result
    ref_result : np.ndarray or tuple
        Reference implementation result
    atol : float
        Absolute tolerance for relative error calculation
        
    Returns
    -------
    dict
        Dictionary with 'max_abs_error' and 'max_rel_error' keys
    """
    from pathlib import Path
    import sys
    
    # Import test_runner utilities
    tools_dir = Path(__file__).parent.parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    try:
        from test_runner import calculate_errors as calc_errors
        return calc_errors(torch_result, ref_result, atol)
    except ImportError:
        # Fallback implementation
        if isinstance(torch_result, tuple):
            if not isinstance(ref_result, tuple) or len(torch_result) != len(ref_result):
                return {"max_abs_error": float("inf"), "max_rel_error": float("inf")}
            
            max_abs = 0.0
            max_rel = 0.0
            
            for t_res, r_res in zip(torch_result, ref_result):
                t_np = t_res.detach().cpu().numpy() if isinstance(t_res, torch.Tensor) else np.array(t_res)
                r_np = r_res if isinstance(r_res, np.ndarray) else np.array(r_res)
                
                valid_mask = np.isfinite(t_np) & np.isfinite(r_np)
                if np.any(valid_mask):
                    abs_err = np.abs(t_np - r_np)
                    rel_err = abs_err / (np.abs(r_np) + atol)
                    max_abs = max(max_abs, np.max(abs_err[valid_mask]))
                    max_rel = max(max_rel, np.max(rel_err[valid_mask]))
            
            return {"max_abs_error": max_abs, "max_rel_error": max_rel}
        
        t_np = torch_result.detach().cpu().numpy() if isinstance(torch_result, torch.Tensor) else np.array(torch_result)
        r_np = ref_result if isinstance(ref_result, np.ndarray) else np.array(ref_result)
        
        valid_mask = np.isfinite(t_np) & np.isfinite(r_np)
        if not np.any(valid_mask):
            return {"max_abs_error": float("nan"), "max_rel_error": float("nan")}
        
        abs_err = np.abs(t_np - r_np)
        rel_err = abs_err / (np.abs(r_np) + atol)
        
        return {
            "max_abs_error": float(np.max(abs_err[valid_mask])),
            "max_rel_error": float(np.max(rel_err[valid_mask])),
        }


def benchmark_function(func, *args, num_warmup=3, num_runs=10):
    """
    Benchmark a function's performance.
    
    Helper function for comprehensive testing.
    
    Parameters
    ----------
    func : callable
        Function to benchmark
    *args
        Arguments to pass to function
    num_warmup : int
        Number of warmup runs
    num_runs : int
        Number of timed runs
        
    Returns
    -------
    float
        Median execution time in seconds
    """
    import time
    
    # Warmup
    for _ in range(num_warmup):
        try:
            _ = func(*args)
        except:
            pass
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = func(*args)
        end = time.perf_counter()
        times.append(end - start)
    
    return float(np.median(times))
