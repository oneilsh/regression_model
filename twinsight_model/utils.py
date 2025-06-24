import numpy as np
import pandas as pd
import sys # For sys.getsizeof to check object size
import logging
from typing import Any, Optional, Set

# Configure logging for this module
logger = logging.getLogger(__name__) # Use logger for output

def check_for_data(
    obj: Any,
    seen: Optional[Set[int]] = None, # Specify Set[int] for type hinting
    depth: int = 0,
    path: str = "root",
    large_threshold_bytes: int = 1024 * 1024, # Default: 1 MB
    max_depth: int = 15,
    max_collection_items_to_check: int = 10 # Only check first N items in large collections
) -> None:
    """
    Recursively scans an object for large or sensitive data-like attributes.
    Flags:
      - pandas DataFrames/Series (shows shape)
      - numpy arrays (shows shape)
      - lists/tuples/sets (shows length if large)
      - dicts (shows number of keys if large)
      - Any object exceeding large_threshold_bytes in size (using sys.getsizeof)

    Parameters:
      obj: The Python object to inspect.
      seen: A set to keep track of objects already visited to prevent infinite recursion.
      depth: Current recursion depth.
      path: String indicating the current path to the object (for better readability of output).
      large_threshold_bytes: Threshold in bytes to flag an object as "large."
      max_depth: Maximum recursion depth to prevent infinite loops on complex objects.
      max_collection_items_to_check: For lists/tuples/sets/dicts, only checks a subset
                                     of elements/keys to prevent excessive output.
    """
    if seen is None:
        seen = set()
    
    obj_id = id(obj)
    if obj_id in seen or depth > max_depth:
        return # Stop if already seen or max depth reached

    seen.add(obj_id)

    # --- Check for known large data structures ---
    if isinstance(obj, pd.DataFrame):
        logger.info(f"[{path}] pandas.DataFrame detected: shape={obj.shape}, size_bytes={sys.getsizeof(obj)}")
        # No further recursion needed for DataFrame internals for this purpose
        return
    elif isinstance(obj, pd.Series):
        logger.info(f"[{path}] pandas.Series detected: shape={obj.shape}, size_bytes={sys.getsizeof(obj)}")
        return
    elif isinstance(obj, np.ndarray):
        logger.info(f"[{path}] numpy.ndarray detected: shape={obj.shape}, size_bytes={obj.nbytes if hasattr(obj, 'nbytes') else sys.getsizeof(obj)}")
        return

    # --- Check for large generic collections ---
    elif isinstance(obj, (list, tuple, set)):
        if len(obj) > max_collection_items_to_check or sys.getsizeof(obj) > large_threshold_bytes:
            logger.info(f"[{path}] {type(obj).__name__} detected: length={len(obj)}, size_bytes={sys.getsizeof(obj)}")
        # Recurse into elements (check a limited number for brevity)
        for idx, item in enumerate(obj[:max_collection_items_to_check]):
            check_for_data(item, seen, depth + 1, f"{path}[{idx}]", large_threshold_bytes, max_depth, max_collection_items_to_check)
        return # Prevent further __dict__ inspection for direct list/tuple/set instances

    elif isinstance(obj, dict):
        if len(obj) > max_collection_items_to_check or sys.getsizeof(obj) > large_threshold_bytes:
            logger.info(f"[{path}] dict detected: {len(obj)} keys, size_bytes={sys.getsizeof(obj)}")
        # Recurse into values (check a limited number of keys)
        for key in list(obj.keys())[:max_collection_items_to_check]:
            check_for_data(obj[key], seen, depth + 1, f"{path}['{key}']", large_threshold_bytes, max_depth, max_collection_items_to_check)
        return # Prevent further __dict__ inspection for direct dict instances

    # --- Check general object size ---
    try:
        obj_size = sys.getsizeof(obj)
        if obj_size > large_threshold_bytes:
            logger.info(f"[{path}] Large object detected: type={type(obj).__name__}, size_bytes={obj_size}")
    except TypeError: # Some objects don't support getsizeof directly
        pass

    # --- Recurse into object attributes (for custom classes) ---
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            check_for_data(v, seen, depth + 1, f"{path}.{k}", large_threshold_bytes, max_depth, max_collection_items_to_check)

    # Note: For efficiency and clarity, specific known model attributes like
    # '._data' in lifelines.CoxPHFitter are targeted manually in model.py.
    # This recursive check is a safety net for unexpected data storage.

