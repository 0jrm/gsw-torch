"""
Function discovery tool to inventory all functions from reference and PyTorch implementations.

This script discovers all public functions from both the reference GSW implementation
and the PyTorch implementation, then creates an inventory of:
- Functions in both implementations (to test)
- Functions only in reference (missing in PyTorch)
- Functions only in PyTorch (extras)
"""

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def get_reference_gsw():
    """Get reference GSW implementation."""
    # Try multiple paths
    project_root = Path(__file__).parent.parent.parent
    paths_to_try = [
        project_root / "source_files",
        project_root / "reference",
    ]
    # Try environment variable as fallback
    if "GSW_SOURCE_FILES" in os.environ:
        paths_to_try.append(Path(os.environ["GSW_SOURCE_FILES"]))
    
    for path in paths_to_try:
        gsw_path = path / "gsw"
        if gsw_path.exists():
            sys.path.insert(0, str(path))
            try:
                import gsw as gsw_ref
                return gsw_ref, str(path)
            except (ImportError, ModuleNotFoundError):
                if str(path) in sys.path:
                    sys.path.remove(str(path))
                continue
    
    return None, None


def get_pytorch_gsw():
    """Get PyTorch GSW implementation."""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    try:
        import gsw_torch
        return gsw_torch
    except ImportError:
        return None


def discover_functions(module: Any) -> Dict[str, Dict[str, Any]]:
    """
    Discover all public functions from a module.
    
    Parameters
    ----------
    module : Any
        The module to discover functions from
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping function names to their metadata
    """
    functions = {}
    
    # Get all attributes from module
    for name in dir(module):
        # Skip private attributes
        if name.startswith("__"):
            continue
            
        obj = getattr(module, name)
        
        # Check if it's a function
        if not inspect.isfunction(obj):
            # Check if it's a callable (might be a wrapped function)
            if not (callable(obj) and hasattr(obj, "__name__")):
                continue
        
        # Skip if it's imported from elsewhere (check module)
        if hasattr(obj, "__module__"):
            module_name = obj.__module__
            # Only include functions from the gsw/gsw_torch package
            if module_name and (
                module_name.startswith("gsw.") or 
                module_name.startswith("gsw_torch.") or
                module_name == "gsw" or
                module_name == "gsw_torch"
            ):
                try:
                    # Get function signature
                    sig = inspect.signature(obj)
                    params = list(sig.parameters.keys())
                    
                    functions[name] = {
                        "name": name,
                        "module": module_name,
                        "parameters": params,
                        "num_params": len(params),
                        "signature": str(sig),
                    }
                except (ValueError, TypeError):
                    # Some functions might not have inspectable signatures
                    functions[name] = {
                        "name": name,
                        "module": module_name,
                        "parameters": [],
                        "num_params": 0,
                        "signature": "unknown",
                    }
    
    return functions


def categorize_functions(
    ref_functions: Dict[str, Dict[str, Any]],
    torch_functions: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Categorize functions into different groups.
    
    Parameters
    ----------
    ref_functions : Dict[str, Dict[str, Any]]
        Functions from reference implementation
    torch_functions : Dict[str, Dict[str, Any]]
        Functions from PyTorch implementation
        
    Returns
    -------
    Dict[str, Any]
        Categorized function inventory
    """
    ref_names = set(ref_functions.keys())
    torch_names = set(torch_functions.keys())
    
    # Functions in both (to test)
    in_both = ref_names & torch_names
    
    # Functions only in reference (missing in PyTorch)
    only_ref = ref_names - torch_names
    
    # Functions only in PyTorch (extras)
    only_torch = torch_names - ref_names
    
    # Build detailed inventory
    inventory = {
        "summary": {
            "total_reference": len(ref_names),
            "total_pytorch": len(torch_names),
            "in_both": len(in_both),
            "only_reference": len(only_ref),
            "only_pytorch": len(only_torch),
        },
        "functions_in_both": {
            name: {
                "reference": ref_functions[name],
                "pytorch": torch_functions[name],
            }
            for name in sorted(in_both)
        },
        "functions_only_reference": {
            name: ref_functions[name]
            for name in sorted(only_ref)
        },
        "functions_only_pytorch": {
            name: torch_functions[name]
            for name in sorted(only_torch)
        },
    }
    
    return inventory


def main():
    """Main function to discover and inventory functions."""
    print("Discovering functions from reference GSW...")
    gsw_ref, ref_path = get_reference_gsw()
    if gsw_ref is None:
        print("ERROR: Could not import reference GSW")
        sys.exit(1)
    print(f"Found reference GSW at: {ref_path}")
    
    print("Discovering functions from PyTorch GSW...")
    gsw_torch = get_pytorch_gsw()
    if gsw_torch is None:
        print("ERROR: Could not import PyTorch GSW")
        sys.exit(1)
    print("Found PyTorch GSW")
    
    print("Extracting functions from reference implementation...")
    ref_functions = discover_functions(gsw_ref)
    print(f"Found {len(ref_functions)} functions in reference")
    
    print("Extracting functions from PyTorch implementation...")
    torch_functions = discover_functions(gsw_torch)
    print(f"Found {len(torch_functions)} functions in PyTorch")
    
    print("Categorizing functions...")
    inventory = categorize_functions(ref_functions, torch_functions)
    
    # Print summary
    print("\n" + "=" * 60)
    print("FUNCTION INVENTORY SUMMARY")
    print("=" * 60)
    print(f"Total functions in reference: {inventory['summary']['total_reference']}")
    print(f"Total functions in PyTorch: {inventory['summary']['total_pytorch']}")
    print(f"Functions in both (to test): {inventory['summary']['in_both']}")
    print(f"Functions only in reference: {inventory['summary']['only_reference']}")
    print(f"Functions only in PyTorch: {inventory['summary']['only_pytorch']}")
    
    # Save inventory
    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "function_inventory.json"
    
    # Convert to JSON-serializable format
    def make_serializable(obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    serializable_inventory = make_serializable(inventory)
    
    with open(output_file, "w") as f:
        json.dump(serializable_inventory, f, indent=2)
    
    print(f"\nInventory saved to: {output_file}")
    
    # Print some examples
    if inventory['summary']['only_reference'] > 0:
        print("\nSample functions only in reference:")
        for name in list(inventory['functions_only_reference'].keys())[:5]:
            print(f"  - {name}")
    
    if inventory['summary']['only_pytorch'] > 0:
        print("\nSample functions only in PyTorch:")
        for name in list(inventory['functions_only_pytorch'].keys())[:5]:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
