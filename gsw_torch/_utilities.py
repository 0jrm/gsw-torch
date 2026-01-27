"""
Utility functions for PyTorch-based GSW implementation.

These utilities handle tensor conversion, broadcasting, and other
common operations needed across the GSW package.
"""

from functools import wraps
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch


def as_tensor(
    arg: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert input to a torch tensor, handling various input types.

    Parameters
    ----------
    arg : array-like
        Input to convert (numpy array, list, scalar, torch tensor, etc.)
    dtype : torch.dtype, optional
        Desired dtype. If None, uses float64 for numeric types.
    device : torch.device, optional
        Desired device. If None, uses CPU.

    Returns
    -------
    torch.Tensor
        Converted tensor
    """
    if isinstance(arg, torch.Tensor):
        if dtype is not None and arg.dtype != dtype:
            arg = arg.to(dtype=dtype)
        if device is not None and arg.device != device:
            arg = arg.to(device=device)
        return arg

    if isinstance(arg, np.ndarray):
        tensor = torch.from_numpy(arg)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        elif tensor.dtype.is_floating_point:
            tensor = tensor.to(dtype=torch.float64)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor

    # Handle numpy masked arrays by converting to nan
    if hasattr(np.ma, "isMaskedArray") and np.ma.isMaskedArray(arg):
        if arg.dtype.kind == "f":
            arg = arg.filled(np.nan)
        else:
            arg = arg.astype(float).filled(np.nan)
        return as_tensor(arg, dtype=dtype, device=device)

    # Handle scalars and lists
    try:
        tensor = torch.as_tensor(arg, dtype=dtype, device=device)
        if dtype is None and tensor.dtype.is_floating_point:
            tensor = tensor.to(dtype=torch.float64)
        return tensor
    except (TypeError, ValueError):
        # Fallback: convert to numpy first, then to tensor
        np_array = np.asarray(arg, dtype=float)
        return as_tensor(np_array, dtype=dtype, device=device)


def match_args_return(f: Callable) -> Callable:
    """
    Decorator for functions that operate on profile data.

    This decorator handles:
    - Converting inputs to torch tensors
    - Handling numpy arrays and masked arrays
    - Broadcasting arrays
    - Converting scalar outputs appropriately
    - Handling the 'p' keyword argument

    Parameters
    ----------
    f : Callable
        Function to wrap

    Returns
    -------
    Callable
        Wrapped function with tensor conversion and broadcasting
    """
    @wraps(f)
    def wrapper(*args, **kw):
        # Handle 'p' keyword argument that might be passed separately
        p = kw.get("p", None)
        if p is not None:
            args = list(args)
            args.append(p)

        # Check if inputs are arrays/iterables
        isarray = [hasattr(a, "__iter__") and not isinstance(a, (str, bytes)) for a in args]
        ismasked = [
            hasattr(np.ma, "isMaskedArray") and np.ma.isMaskedArray(a) for a in args
        ]
        isduck = [
            hasattr(a, "__array_ufunc__") and not isinstance(a, (torch.Tensor, np.ndarray))
            for a in args
        ]

        hasarray = any(isarray)
        hasmasked = any(ismasked)
        hasduck = any(isduck)

        # Handle leading integer arguments in gibbs and gibbs_ice
        # These functions have integer parameters before the float parameters
        if hasattr(f, "types"):
            # This would be set if we're wrapping a ufunc-like function
            # For now, we'll detect integer args manually
            first_double = 0
            for i, arg in enumerate(args):
                if isinstance(arg, (int, np.integer, torch.Tensor)) and (
                    isinstance(arg, int)
                    or (isinstance(arg, torch.Tensor) and arg.dtype.is_integer)
                    or (isinstance(arg, np.integer))
                ):
                    continue
                first_double = i
                break
            int_return = False  # Most functions return floats
        else:
            first_double = 0
            int_return = False

        def fixup(ret):
            """Fix return value formatting."""
            if hasduck:
                return ret
            if hasmasked and not int_return:
                # Convert nans to masked if input was masked
                # For torch, we just return the tensor with nans
                pass
            if not hasarray and isinstance(ret, torch.Tensor) and ret.numel() == 1:
                # Return scalar-like (0-d tensor) for scalar inputs
                return ret.squeeze()
            return ret

        # Convert arguments to tensors
        newargs = []
        for i, arg in enumerate(args):
            if i < first_double:
                # Integer arguments (for gibbs, gibbs_ice)
                if isinstance(arg, torch.Tensor):
                    newargs.append(arg)
                elif isinstance(arg, np.ndarray):
                    newargs.append(torch.from_numpy(arg))
                else:
                    newargs.append(arg)
            elif ismasked[i]:
                # Handle masked arrays
                if hasattr(np.ma, "isMaskedArray") and np.ma.isMaskedArray(arg):
                    if arg.dtype.kind == "f":
                        arg = arg.filled(np.nan)
                    else:
                        arg = arg.astype(float).filled(np.nan)
                newargs.append(as_tensor(arg))
            elif isduck[i]:
                newargs.append(arg)
            else:
                newargs.append(as_tensor(arg))

        if p is not None:
            kw["p"] = newargs.pop()

        # Call the function
        ret = f(*newargs, **kw)

        # Fix up return value
        if isinstance(ret, tuple):
            retlist = [fixup(arg) for arg in ret]
            ret = tuple(retlist)
        else:
            ret = fixup(ret)
        return ret

    wrapper.__wrapped__ = f
    return wrapper


def axis_slicer(n: int, sl: slice, axis: int) -> Tuple[Union[slice, int], ...]:
    """
    Return an indexing tuple for an array with `n` dimensions,
    with slice `sl` taken on `axis`.

    Parameters
    ----------
    n : int
        Number of dimensions
    sl : slice
        Slice to apply
    axis : int
        Axis to slice

    Returns
    -------
    tuple
        Indexing tuple for torch tensor
    """
    itup = [slice(None)] * n
    itup[axis] = sl
    return tuple(itup)


def indexer(shape: Tuple[int, ...], axis: int, order: str = "C"):
    """
    Generator of indexing tuples for "apply_along_axis" usage.

    The generator cycles through all axes other than `axis`.
    This allows working with functions of more than one array
    along a specific axis.

    Parameters
    ----------
    shape : tuple
        Shape of the array
    axis : int
        Axis along which to iterate
    order : str, optional
        Order of iteration ('C' for C-order, 'F' for Fortran-order)

    Yields
    ------
    tuple
        Indexing tuples for each iteration
    """
    ndim = len(shape)
    ind_shape = list(shape)
    ind_shape[axis] = 1  # "axis" and any dim of 1 will not be incremented
    # List of indices, with a slice at "axis"
    inds = [0] * ndim
    inds[axis] = slice(None)
    kmax = 1
    for s in ind_shape:
        kmax *= s

    if order == "C":
        index_position = list(reversed(range(ndim)))
    else:
        index_position = list(range(ndim))

    for _k in range(kmax):
        yield tuple(inds)

        for i in index_position:
            if ind_shape[i] == 1:
                continue
            inds[i] += 1
            if inds[i] == ind_shape[i]:
                inds[i] = 0
            else:
                break


class Bunch(dict):
    """
    A dictionary that also provides access via attributes.

    Additional methods update_values and update_None provide
    control over whether new keys are added to the dictionary
    when updating, and whether an attempt to add a new key is
    ignored or raises a KeyError.

    The Bunch also prints differently than a normal
    dictionary, using str() instead of repr() for its
    keys and values, and in key-sorted order.  The printing
    format can be customized by subclassing with a different
    str_ftm class attribute.
    """

    str_fmt = "{0!s:<{klen}} : {1!s:>{vlen}}\n"

    def __init__(self, *args, **kwargs):
        """
        *args* can be dictionaries, bunches, or sequences of
        key,value tuples.  *kwargs* can be used to initialize
        or add key, value pairs.
        """
        dict.__init__(self)
        for arg in args:
            self.update(arg)
        self.update(kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(f"'Bunch' object has no attribute {name}. {err}")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __str__(self) -> str:
        return self.formatted()

    def formatted(self, fmt: Optional[str] = None, types: bool = False) -> str:
        """
        Return a string with keys and/or values or types.

        *fmt* is a format string as used in the str.format() method.

        The str.format() method is called with key, value as positional
        arguments, and klen, vlen as kwargs.  The latter are the maxima
        of the string lengths for the keys and values, respectively,
        up to respective maxima of 20 and 40.
        """
        if fmt is None:
            fmt = self.str_fmt

        items = list(self.items())
        items.sort()

        klens = []
        vlens = []
        for i, (k, v) in enumerate(items):
            lenk = len(str(k))
            if types:
                v = type(v).__name__
            lenv = len(str(v))
            items[i] = (k, v)
            klens.append(lenk)
            vlens.append(lenv)

        klen = min(20, max(klens)) if klens else 0
        vlen = min(40, max(vlens)) if vlens else 0
        slist = [fmt.format(k, v, klen=klen, vlen=vlen) for k, v in items]
        return "".join(slist)

    def from_pyfile(self, filename: str) -> "Bunch":
        """
        Read in variables from a python code file.
        """
        d = {}
        lines = ["def _temp_func():\n"]
        with open(filename) as f:
            lines.extend(["    " + line for line in f])
        lines.extend(["\n    return(locals())\n", "_temp_out = _temp_func()\n", "del(_temp_func)\n"])
        codetext = "".join(lines)
        code = compile(codetext, filename, "exec")
        exec(code, globals(), d)
        self.update(d["_temp_out"])
        return self

    def update_values(self, *args: dict, **kw: Any) -> None:
        """
        arguments are dictionary-like; if present, they act as
        additional sources of kwargs, with the actual kwargs
        taking precedence.

        One reserved optional kwarg is "strict".  If present and
        True, then any attempt to update with keys that are not
        already in the Bunch instance will raise a KeyError.
        """
        strict = kw.pop("strict", False)
        newkw = {}
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = {k: v for (k, v) in newkw.items() if k in self}
        self.update(dsub)

    def update_None(self, *args: dict, **kw: Any) -> None:
        """
        Similar to update_values, except that an existing value
        will be updated only if it is None.
        """
        strict = kw.pop("strict", False)
        newkw = {}
        for d in args:
            newkw.update(d)
        newkw.update(kw)
        self._check_strict(strict, newkw)
        dsub = {k: v for (k, v) in newkw.items() if k in self and self[k] is None}
        self.update(dsub)

    def _check_strict(self, strict: bool, kw: dict) -> None:
        if strict:
            bad = set(kw.keys()) - set(self.keys())
            if bad:
                bk = sorted(bad)
                ek = sorted(self.keys())
                raise KeyError(f"Update keys {bk} don't match existing keys {ek}")
