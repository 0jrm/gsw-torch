"""
Core test runner for comprehensive validation.

This module provides functions to test individual GSW functions for:
1. Numerical parity (error > 1e-8 flagged)
2. Autograd compatibility
3. Performance benchmarks
"""

import inspect
import time
import traceback
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# Error thresholds
ABSOLUTE_TOLERANCE = 1e-8
RELATIVE_TOLERANCE = 1e-6

# Suppress expected warnings during testing
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")


def get_reference_gsw():
    """Get reference GSW implementation."""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent
    paths_to_try = [
        project_root / "source_files",
        Path("/home/jrm22n/gsw2torch/source_files"),
        project_root / "reference",
    ]
    
    for path in paths_to_try:
        gsw_path = path / "gsw"
        if gsw_path.exists():
            sys.path.insert(0, str(path))
            try:
                import gsw as gsw_ref
                return gsw_ref
            except (ImportError, ModuleNotFoundError):
                if str(path) in sys.path:
                    sys.path.remove(str(path))
                continue
    
    return None


def get_pytorch_gsw():
    """Get PyTorch GSW implementation."""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    try:
        import gsw_torch
        return gsw_torch
    except ImportError:
        return None


def convert_to_numpy(value: Any) -> np.ndarray:
    """Convert value to numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    elif isinstance(value, (list, tuple)):
        return np.array(value)
    else:
        return np.array([value])


def convert_to_torch(value: Any, requires_grad: bool = False) -> torch.Tensor:
    """Convert value to torch tensor."""
    if isinstance(value, torch.Tensor):
        if requires_grad and not value.requires_grad:
            return value.clone().detach().requires_grad_(True)
        return value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).clone()
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor
    elif isinstance(value, (list, tuple)):
        tensor = torch.tensor(value, dtype=torch.float64)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor
    else:
        tensor = torch.tensor([value], dtype=torch.float64)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor


def calculate_errors(
    torch_result: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    ref_result: Union[np.ndarray, Tuple[np.ndarray, ...]],
    atol: float = ABSOLUTE_TOLERANCE
) -> Dict[str, float]:
    """
    Calculate absolute and relative errors between torch and reference results.
    
    Parameters
    ----------
    torch_result : Union[torch.Tensor, Tuple[torch.Tensor, ...]]
        PyTorch implementation result
    ref_result : Union[np.ndarray, Tuple[np.ndarray, ...]]
        Reference implementation result
    atol : float
        Absolute tolerance for relative error calculation
        
    Returns
    -------
    Dict[str, float]
        Dictionary with error metrics
    """
    # Handle tuples
    if isinstance(torch_result, tuple):
        if not isinstance(ref_result, tuple) or len(torch_result) != len(ref_result):
            return {"max_abs_error": float("inf"), "max_rel_error": float("inf")}
        
        max_abs = 0.0
        max_rel = 0.0
        
        for t_res, r_res in zip(torch_result, ref_result):
            t_np = convert_to_numpy(t_res)
            r_np = convert_to_numpy(r_res) if not isinstance(r_res, np.ndarray) else r_res
            
            # Handle boolean outputs
            if t_np.dtype == bool or r_np.dtype == bool:
                t_np = t_np.astype(int)
                r_np = r_np.astype(int)
            
            # Handle NaN and inf
            valid_mask = np.isfinite(t_np) & np.isfinite(r_np)
            if not np.any(valid_mask):
                continue
            
            abs_err = np.abs(t_np - r_np)
            rel_err = abs_err / (np.abs(r_np) + atol)
            
            max_abs = max(max_abs, np.max(abs_err[valid_mask]))
            max_rel = max(max_rel, np.max(rel_err[valid_mask]))
        
        return {"max_abs_error": max_abs, "max_rel_error": max_rel}
    
    # Handle single values
    t_np = convert_to_numpy(torch_result)
    r_np = convert_to_numpy(ref_result) if not isinstance(ref_result, np.ndarray) else ref_result
    
    # Handle boolean outputs (e.g., infunnel)
    if t_np.dtype == bool or r_np.dtype == bool:
        t_np = t_np.astype(int)
        r_np = r_np.astype(int)
    
    # Handle NaN and inf
    valid_mask = np.isfinite(t_np) & np.isfinite(r_np)
    if not np.any(valid_mask):
        return {"max_abs_error": float("nan"), "max_rel_error": float("nan")}
    
    abs_err = np.abs(t_np - r_np)
    rel_err = abs_err / (np.abs(r_np) + atol)
    
    return {
        "max_abs_error": float(np.max(abs_err[valid_mask])),
        "max_rel_error": float(np.max(rel_err[valid_mask])),
    }


def test_numerical_parity(
    func_name: str,
    ref_func: Callable,
    torch_func: Callable,
    test_inputs: Tuple[Any, ...],
    sample_size: int
) -> Dict[str, Any]:
    """
    Test numerical parity between reference and PyTorch implementations.
    
    Parameters
    ----------
    func_name : str
        Name of the function being tested
    ref_func : Callable
        Reference implementation function
    torch_func : Callable
        PyTorch implementation function
    test_inputs : Tuple[Any, ...]
        Test input arguments
    sample_size : int
        Sample size being tested
        
    Returns
    -------
    Dict[str, Any]
        Test results dictionary
    """
    result = {
        "function": func_name,
        "sample_size": sample_size,
        "status": "UNKNOWN",
        "max_abs_error": None,
        "max_rel_error": None,
        "error": None,
        "traceback": None,
    }
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            # Convert inputs for reference (numpy)
            ref_inputs = tuple(convert_to_numpy(inp) for inp in test_inputs)
            
            # Call reference implementation
            ref_output = ref_func(*ref_inputs)
            
            # Convert inputs for PyTorch (torch tensors)
            torch_inputs = tuple(convert_to_torch(inp) for inp in test_inputs)
            
            # Call PyTorch implementation
            torch_output = torch_func(*torch_inputs)
            
            # Calculate errors
            errors = calculate_errors(torch_output, ref_output)
            
            result["max_abs_error"] = errors["max_abs_error"]
            result["max_rel_error"] = errors["max_rel_error"]
            
            # Check if errors exceed threshold
            if np.isnan(errors["max_abs_error"]):
                result["status"] = "ERROR_NAN"
            elif errors["max_abs_error"] > ABSOLUTE_TOLERANCE:
                result["status"] = "FAIL"
            else:
                result["status"] = "PASS"
            
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    return result


def test_autograd(
    func_name: str,
    torch_func: Callable,
    test_inputs: Tuple[Any, ...],
    differentiable_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Test autograd compatibility for PyTorch function.
    
    Parameters
    ----------
    func_name : str
        Name of the function being tested
    torch_func : Callable
        PyTorch implementation function
    test_inputs : Tuple[Any, ...]
        Test input arguments
    differentiable_indices : Optional[List[int]]
        Indices of inputs that should be differentiable (None = all)
        
    Returns
    -------
    Dict[str, Any]
        Test results dictionary
    """
    result = {
        "function": func_name,
        "status": "UNKNOWN",
        "gradients_work": False,
        "num_differentiable_inputs": 0,
        "issues": [],
        "error": None,
        "traceback": None,
    }
    
    if differentiable_indices is None:
        # Try all inputs
        differentiable_indices = list(range(len(test_inputs)))
    
    try:
        # Create inputs with requires_grad=True
        grad_inputs = []
        for i, inp in enumerate(test_inputs):
            if i in differentiable_indices:
                grad_inputs.append(convert_to_torch(inp, requires_grad=True))
            else:
                grad_inputs.append(convert_to_torch(inp, requires_grad=False))
        
        # Call function
        output = torch_func(*grad_inputs)
        
        # Handle tuple outputs
        if isinstance(output, tuple):
            outputs = output
        else:
            outputs = (output,)
        
        # Compute gradients for each output
        gradients_work = True
        num_grads = 0
        
        # For tuple outputs, use torch.autograd.grad instead of backward()
        if len(outputs) > 1:
            # Multiple outputs - use autograd.grad
            try:
                # Sum all outputs to get scalar
                out_sum = sum(out.sum() if isinstance(out, torch.Tensor) else 0 for out in outputs)
                
                if isinstance(out_sum, torch.Tensor) and out_sum.requires_grad:
                    # Compute gradients for each differentiable input
                    grads = torch.autograd.grad(
                        out_sum,
                        [grad_inputs[i] for i in differentiable_indices if i < len(grad_inputs)],
                        retain_graph=False,
                        allow_unused=True
                    )
                    
                    for grad_idx, i in enumerate(differentiable_indices):
                        if i >= len(grad_inputs):
                            continue
                        if grad_idx < len(grads) and grads[grad_idx] is not None:
                            num_grads += 1
                            if not torch.isfinite(grads[grad_idx]).all():
                                gradients_work = False
                                result["issues"].append(f"Non-finite gradients for input {i}")
                        else:
                            gradients_work = False
                            result["issues"].append(f"No gradient for input {i}")
                            
            except Exception as e:
                gradients_work = False
                result["issues"].append(f"Error computing gradients: {str(e)}")
        else:
            # Single output - can use backward()
            for out_idx, out in enumerate(outputs):
                if not isinstance(out, torch.Tensor):
                    continue
                
                # Check if output requires grad
                if not out.requires_grad:
                    continue
                
                # Compute gradient
                try:
                    # Sum output to get scalar
                    out_sum = out.sum()
                    out_sum.backward(retain_graph=False)
                    
                    # Check gradients
                    for i, inp in enumerate(grad_inputs):
                        if i in differentiable_indices and isinstance(inp, torch.Tensor):
                            if inp.grad is not None:
                                num_grads += 1
                                # Check if gradient is finite
                                if not torch.isfinite(inp.grad).all():
                                    gradients_work = False
                                    result["issues"].append(
                                        f"Non-finite gradients for input {i}"
                                    )
                            else:
                                gradients_work = False
                                result["issues"].append(f"No gradient for input {i}")
                                
                except Exception as e:
                    gradients_work = False
                    result["issues"].append(f"Error computing gradient for output {out_idx}: {str(e)}")
        
        result["gradients_work"] = gradients_work
        result["num_differentiable_inputs"] = num_grads
        
        if gradients_work:
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
            
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    return result


def benchmark_performance(
    func_name: str,
    ref_func: Callable,
    torch_func: Callable,
    test_inputs: Tuple[Any, ...],
    sample_size: int,
    num_warmup: int = 3,
    num_runs: int = 10
) -> Dict[str, Any]:
    """
    Benchmark performance of PyTorch vs Reference implementation.
    
    Parameters
    ----------
    func_name : str
        Name of the function being tested
    ref_func : Callable
        Reference implementation function
    torch_func : Callable
        PyTorch implementation function
    test_inputs : Tuple[Any, ...]
        Test input arguments
    sample_size : int
        Sample size being tested
    num_warmup : int
        Number of warmup runs
    num_runs : int
        Number of timed runs
        
    Returns
    -------
    Dict[str, Any]
        Benchmark results dictionary
    """
    result = {
        "function": func_name,
        "sample_size": sample_size,
        "status": "UNKNOWN",
        "ref_time": None,
        "torch_time": None,
        "speedup": None,
        "error": None,
        "traceback": None,
    }
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            # Prepare inputs
            ref_inputs = tuple(convert_to_numpy(inp) for inp in test_inputs)
            torch_inputs = tuple(convert_to_torch(inp) for inp in test_inputs)
            
            # Warmup runs
            for _ in range(num_warmup):
                try:
                    _ = ref_func(*ref_inputs)
                except:
                    pass
                try:
                    _ = torch_func(*torch_inputs)
                except:
                    pass
            
            # Benchmark reference
            ref_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = ref_func(*ref_inputs)
                end = time.perf_counter()
                ref_times.append(end - start)
            
            # Benchmark PyTorch
            torch_times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = torch_func(*torch_inputs)
                end = time.perf_counter()
                torch_times.append(end - start)
            
            # Calculate median times
            ref_median = np.median(ref_times)
            torch_median = np.median(torch_times)
            
            result["ref_time"] = float(ref_median)
            result["torch_time"] = float(torch_median)
            result["speedup"] = float(ref_median / torch_median) if torch_median > 0 else None
            result["status"] = "PASS"
        
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    return result
