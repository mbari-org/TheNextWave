from dataclasses import fields, is_dataclass
from typing import Any, Dict
import numpy as np
import pandas as pd

from .swift import SWIFTArray, SWIFTData, WaveSpectra, Signature, Uplooking


def _filter_ndarray_by_time_axis(arr: np.ndarray, indices: np.ndarray, time_len: int) -> np.ndarray:
    """If arr's first axis equals time_len, index it; otherwise return arr."""
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.size == 0:
        return arr
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == time_len:
        if indices.size == 0:
            return arr[[]]
        return np.take(arr, indices, axis=0)
    return arr


def _filter_dataclass_by_indices(dc_instance: Any, indices: np.ndarray, time_len: int) -> Any:
    """Recursively filter dataclass fields that are numpy arrays indexed on time axis."""
    if not is_dataclass(dc_instance):
        if isinstance(dc_instance, np.ndarray):
            return _filter_ndarray_by_time_axis(dc_instance, indices, time_len)
        return dc_instance

    dc_type = type(dc_instance)
    new_kwargs: Dict[str, Any] = {}
    for f in fields(dc_instance):
        val = getattr(dc_instance, f.name)
        if is_dataclass(val):
            new_val = _filter_dataclass_by_indices(val, indices, time_len)
        elif isinstance(val, np.ndarray):
            new_val = _filter_ndarray_by_time_axis(val, indices, time_len)
        else:
            new_val = val
        new_kwargs[f.name] = new_val
    return dc_type(**new_kwargs)


def align_swift_bursts_swiftarray(swift_array, ref_buoy_name: str, tol: float = 1.0 / 24.0):
    """
    Align bursts in a SWIFTArray (works with numeric datenum times or datetime64[ns] times).

    Parameters
    ----------
    swift_array : SWIFTArray
        Your dataclass instance (fields: swift22, swift23, ...),
        each field is a SWIFTData dataclass whose .time field is either:
          - numeric MATLAB datenum (float days), or
          - numpy.datetime64 / datetime64[ns] / pandas datetime
    ref_buoy_name : str
        Name of the buoy attribute to use as reference (e.g., 'swift25').
    tol : float
        Tolerance in days (float). Example: 0.25/24 = 15 minutes.
    """
    buoy_names = [f.name for f in fields(type(swift_array))]

    if ref_buoy_name not in buoy_names:
        raise ValueError(f'Reference buoy "{ref_buoy_name}" not found in SWIFTArray fields')

    # reference times (may be numeric or datetime64)
    ref_sw = getattr(swift_array, ref_buoy_name)
    t_ref = np.array(ref_sw.time, copy=False)
    n_ref = t_ref.size

    matched_idx: Dict[str, np.ndarray] = {}

    for name in buoy_names:
        sw = getattr(swift_array, name)
        t_other = np.array(sw.time, copy=False)

        if name == ref_buoy_name:
            matched_idx[name] = np.arange(n_ref, dtype=int)
            continue

        if t_other.size == 0 or n_ref == 0:
            matched_idx[name] = np.full(n_ref, -1, dtype=int)
            continue

        # build diffs matrix and find nearest index in t_other for each t_ref
        # works for numeric floats and for datetime64 (timedelta64 results)
        diffs = np.abs(t_other[:, None] - t_ref[None, :])  # shape (n_other, n_ref)
        idx_per_ref = np.argmin(diffs, axis=0).astype(int)  # length n_ref

        # compute time difference in DAYS as float, regardless of input dtype
        # - if times are datetime-like, use pandas to compute timedelta and convert to days
        # - otherwise (numeric matlab datenum) just use numeric difference
        try:
            is_datetime = np.issubdtype(t_ref.dtype, np.datetime64) or np.issubdtype(t_other.dtype, np.datetime64)
        except TypeError:
            # fallback if dtype inspection fails
            is_datetime = False

        if is_datetime:
            # pandas vectorized subtraction -> TimedeltaIndex; divide by np.timedelta64(1,'D') -> float days
            t_other_sel = t_other[idx_per_ref]
            # ensure pandas accepts the input (works for np.datetime64 and pd.Timestamp)
            dt_days = (np.abs(pd.to_datetime(t_other_sel) - pd.to_datetime(t_ref)) / np.timedelta64(1, 'D')).astype(float)
        else:
            dt_days = np.abs(t_other[idx_per_ref] - t_ref).astype(float)

        # apply tolerance (days); mark invalid matches with -1
        idx_per_ref[dt_days > tol] = -1
        matched_idx[name] = idx_per_ref

    # only keep reference bursts that have a valid match in every buoy
    valid_mask = np.ones(n_ref, dtype=bool)
    for name in buoy_names:
        valid_mask &= (matched_idx[name] >= 0)

    n_aligned = int(np.sum(valid_mask))

    aligned_kwargs: Dict[str, Any] = {}
    for name in buoy_names:
        orig_sw = getattr(swift_array, name)
        if n_aligned == 0:
            indices = np.array([], dtype=int)
            time_len = 0
        else:
            idxs_for_buoy = matched_idx[name]
            indices = idxs_for_buoy[valid_mask]
            time_len = int(np.array(orig_sw.time).size)

        new_sw = _filter_dataclass_by_indices(orig_sw, indices, time_len)
        aligned_kwargs[name] = new_sw

    SWIFTArrayType = type(swift_array)
    aligned_swift_array = SWIFTArrayType(**aligned_kwargs)

    print(f"Aligned {n_aligned} common bursts across all buoys.")
    return aligned_swift_array
