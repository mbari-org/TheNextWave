from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io as spio
import xarray as xr


def _loadmat_struct(path: str):
    # scipy.io.loadmat gives MATLAB structs as objects when struct_as_record=False
    return spio.loadmat(path, struct_as_record=False, squeeze_me=True)


def empty_float64():
    return np.array([], dtype=np.float64)

def empty_int():
    return np.array([], dtype=int)


@dataclass
class WaveSpec:
    Etheta: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    theta: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    f: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    spread: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    spread2: npt.NDArray[np.float64] = field(default_factory=empty_float64)


def _get_field_meta(dc_type) -> Dict[str, Dict[str, Any]]:
    """Return mapping field_name -> metadata dict for a dataclass type."""
    out: Dict[str, Dict[str, Any]] = {}
    for f in fields(dc_type):
        out[f.name] = dict(f.metadata) if f.metadata is not None else {}
    return out


def recursive_metadata(dc_instance_or_type) -> Dict[str, Any]:
    """
    Return nested metadata for a dataclass instance or type.
    If input is a class, returns metadata structure for that class.
    If input is an instance, recurses into nested dataclass attributes.
    """
    if hasattr(dc_instance_or_type, "__dataclass_fields__"):
        # dataclass type or instance
        meta = _get_field_meta(dc_instance_or_type if isinstance(dc_instance_or_type, type) else type(dc_instance_or_type))
        if not isinstance(dc_instance_or_type, type):
            # instance: recurse into nested dataclass values
            out = {}
            for name, m in meta.items():
                val = getattr(dc_instance_or_type, name)
                if hasattr(val, "__dataclass_fields__"):
                    out[name] = {"meta": m, "children": recursive_metadata(val)}
                else:
                    out[name] = {"meta": m}
            return out
        else:
            # class: return flat metadata for fields only
            return {k: {"meta": v} for k, v in meta.items()}
    else:
        raise TypeError("argument must be a dataclass class or instance")


@dataclass
class WaveSpectra:
    freq: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "Hz",
        "desc": "spectral frequencies",
        "shape": "(n,)"
    })
    check: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "TODO but probably unitless",
        "desc": "TODO(andermi) I think this is the ratio of vert/horz motion checking cycle for effect of mooring",
        "shape": "(time, freq)"
    })
    energy_alt: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO(andermi) find out what this is...",
        "shape": "(time, freq)"
    })
    energy: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m^2/Hz",
        "desc": "wave energy spectral density as a function of frequency (from IMU surface elevation)",
        "shape": "(time, freq)"
    })
    a1: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "-",
        "desc": "normalized spectral directional moment (positive east)",
        "shape": "(time, freq)"
    })
    b1: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "-",
        "desc": "normalized spectral directional moment (positive north)",
        "shape": "(time, freq)"
    })
    a2: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "-",
        "desc": "normalized spectral directional moment (east-west)",
        "shape": "(time, freq)"
    })
    b2: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "-",
        "desc": "normalized spectral directional moment (north-south)",
        "shape": "(time, freq)"
    })


@dataclass
class SignatureProfile:
    altimeter: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m",
        "desc": "water depth from altimeter"
    })
    east: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m/s",
        "desc": "vertical profile of zonal (east) velocity (broadband)"
    })
    north: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m/s",
        "desc": "vertical profile of meridional (north) velocity (broadband)"
    })
    w: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m/s",
        "desc": "vertical profile of vertical velocity (broadband)"
    })
    z: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m",
        "desc": "depth bins for the velocity profiles"
    })
    spd_alt: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m/s",
        "desc": "burst-averaged scalar speed (not computed from averaged ENU velocities)"
    })


@dataclass
class SignatureHR:
    w: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m/s",
        "desc": "vertical profile of vertical velocity (HR / pulse-coherent)"
    })
    wvar: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m/s",
        "desc": "vertical velocity standard deviation (HR)"
    })
    tkedissipationrate: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m^2/s^3",
        "desc": "vertical profile of turbulent kinetic energy dissipation rate (HR)"
    })
    z: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m",
        "desc": "depth bins for the TKE dissipation rate profiles (HR)"
    })


@dataclass
class Signature:
    profile: SignatureProfile = field(default_factory=SignatureProfile, metadata={
        "desc": "broadband profile data (downlooking Signature1000 configuration)"
    })
    HRprofile: SignatureHR = field(default_factory=SignatureHR, metadata={
        "desc": "high-resolution (pulse-coherent) profile data"
    })


@dataclass
class Uplooking:
    tkedissipationrate: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m^2/s^3",
        "desc": "vertical profile of turbulent kinetic energy dissipation rate (uplooking ADCP)"
    })
    z: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m",
        "desc": "depth bins for the TKE dissipation rate profiles (uplooking ADCP)"
    })


@dataclass
class SWIFTData:
    name: Optional[str] = ''

    rawtime: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "unix timestamp converted ffrom MATLAB datenum",
        "desc": "unix timestamp"
    })

    u: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "meters per second",
        "desc": "eastings velocity"
    })
    v: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "meters per second",
        "desc": "northings velocity"
    })
    x: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "meters",
        "desc": "TODO"
    })
    y: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "meters",
        "desc": "TODO"
    })
    z: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "meters",
        "desc": "heave"
    })

    wavespectra: WaveSpectra = field(default_factory=WaveSpectra, metadata={
        "desc": "structure containing IMU spectral wave data"
    })

    CTdepth: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    ID: npt.NDArray[int] = field(default=empty_float64, metadata={
        "units": "-",
        "desc": "SWIFT ID"
    })

    airpres: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    airpresstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    airtemp: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    airtempstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    date: Optional[str] = field(default=None, metadata={
        "units": "-",
        "desc": "string giving burst date in format 'ddmmyyyy'"
    })

    driftdirT: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    driftdirTstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    driftspd: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    driftspdstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    lat: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    lon: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    metheight: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    peakwavedirT: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    peakwaveperiod: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    salinity: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    sigwaveheight: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    time: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    watertemp: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    winddirR: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    winddirRstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    winddirT: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    winddirTstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    windspd: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    windspdstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    sigwaveheight_alt: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    peakwaveperiod_alt: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    signature: Signature = field(default_factory=Signature, metadata={
        "desc": "structure containing Nortek Signature1000 HR ADCP data (downlooking configuration)"
    })
    uplooking: Uplooking = field(default_factory=Uplooking, metadata={
        "desc": "structure containing Nortek Aquadopp HR ADCP data (uplooking configuration)"
    })

    #@classmethod
    #def from_dataset(cls, mdat: "MATLAB Data" = None):
    #    if not ds:
    #        return cls()
    #    return cls(

class _SWIFTData:
    pass
class _SBGData:
    pass

@dataclass
class SWIFTArray:
    swift22: _SWIFTData = field(default=None)
    sbg22: _SBGData = field(default=None)
    swift23: _SWIFTData = field(default=None)
    sbg23: _SBGData = field(default=None)
    swift24: _SWIFTData = field(default=None)
    sbg24: _SBGData = field(default=None)
    swift25: _SWIFTData = field(default=None)
    sbg25: _SBGData = field(default=None)

    @classmethod
    def from_mdat(
        cls,
        swiftdat: "MATLAB swift data" = None,
        sbgdat: "MATLAB sbgdata" = None,
        select_idx=None
    ):
        swifts = [None, None, None, None]
        for swift_idx, swiftd in enumerate(swiftdat):
            swift = _loadmat_struct(swiftd)['SWIFT']
            if select_idx is not None:
                try:
                    if swift.size > 1:
                        swift = swift[select_idx]
                except AttributeError:
                    pass
            swifts[swift_idx] = swift
        swift22, swift23, swift24, swift25 = swifts

        sbgs = [None, None, None, None]
        for sbg_idx, sbgd in enumerate(sbgdat):
            sbg = _loadmat_struct(sbgd)['sbgData']
            if select_idx is not None:
                try:
                    if sbg.size > 1:
                        sbg = sbg[select_idx]
                except AttributeError:
                    pass
            sbgs[sbg_idx] = sbg
        sbg22, sbg23, sbg24, sbg25 = sbgs

        return cls(
            swift22=swift22,
            swift23=swift23,
            swift24=swift24,
            swift25=swift25,
            sbg22=sbg22,
            sbg23=sbg23,
            sbg24=sbg24,
            sbg25=sbg25
        )


@dataclass
class LSQWavePropParams:
    """
    Solver output parameters for the least–squares wave–propagation model.

    Shapes:
        - A: (N,), wave amplitude solution vector (concatenated cosine/sine components)
        - Etheta: (Nθ, Nf), directional spectrum reconstructed from solution
        - f: (Nf,), frequency grid [Hz]
        - theta: (Nθ,), direction grid [degrees]
        - kx, ky: (N,), Cartesian wavenumber components [rad/m]
        - omega: (N,), angular frequencies [rad/s]

    Purpose:
        Stores all per–target diagnostic outputs needed for spectrum
        reconstruction and physics-quality verification.
    """
    A: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={
            "units": "m",
            "description": "Wave amplitudes (cosine and sine components concatenated). "
                           "Length = 2000 = 25 directions × 40 frequencies × 2."
        },
    )
    Etheta: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={
            "units": "m^2/Hz/deg",
            "description": "Directional wave energy spectrum. "
                           "Dimensions: direction (25) × frequency (40)."
        },
    )
    f: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={
            "units": "Hz",
            "description": "Logarithmically spaced frequency components (40 elements)."
        },
    )
    theta: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={
            "units": "deg (nautical)",
            "description": "Directional components (25 elements)."
        },
    )
    kx: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={
            "units": "1/m",
            "description": "x-component of wavenumber for each (direction×frequency) = 1000 components."
        },
    )
    ky: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={
            "units": "1/m",
            "description": "y-component of wavenumber for each (direction×frequency) = 1000 components."
        },
    )
    omega: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={
            "units": "rad/s",
            "description": "Angular frequency for each (direction×frequency) = 1000 components."
        },
    )
    use_vel: bool = field(
        default=False,
        metadata={
            "description": "True if velocities were included in inversion."
        },
    )


@dataclass
class Prediction:
    """
    Container for all measurement, reconstruction, and prediction arrays
    produced by the least–squares wave–propagation system.

    MATLAB–consistent shapes:
        Measurement arrays (3 instruments × M samples):
            zm, zc, um, vm, uc, vc, tm:  (M, K)  typically (348, 3)

        Target-point arrays (one target × T times):
            zt, ut, vt: (1, T)
            tp: (T, 1)

        Predictions:
            zp: (T, 1)
            up, vp: optional, same shape as zp

        Solver parameters:
            params: list of LSQWavePropParams (one per predicted time)
            comp_time: (T,), computation time per prediction

    Notes:
        - All arrays are kept 2-D (MATLAB style).
        - zp, up, vp are column vectors for each prediction time.
    """

    tp: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "s", "description": "Prediction times (seconds since t0)"}
    )
    tm: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "s", "description": "Measurement times for each instrument"}
    )

    zm: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m", "description": "Measured vertical displacement"}
    )
    zc: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m", "description": "Reconstructed vertical displacement at sensors"}
    )
    um: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m/s", "description": "Measured eastward velocity"}
    )
    vm: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m/s", "description": "Measured northward velocity"}
    )
    uc: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m/s", "description": "Reconstructed eastward velocity"}
    )
    vc: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m/s", "description": "Reconstructed northward velocity"}
    )

    zp: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m", "description": "Predicted surface elevation at target"}
    )
    up: Optional[npt.NDArray[np.float64]] = field(
        default=None,
        metadata={"units": "m/s", "description": "Predicted eastward velocity at target"}
    )
    vp: Optional[npt.NDArray[np.float64]] = field(
        default=None,
        metadata={"units": "m/s", "description": "Predicted northward velocity at target"}
    )

    zt: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m", "description": "Ground truth elevation at target"}
    )
    ut: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m/s", "description": "Ground truth eastward velocity at target"}
    )
    vt: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "m/s", "description": "Ground truth northward velocity at target"}
    )

    params: List[LSQWavePropParams] = field(
        default_factory=list,
        metadata={"description": "Parameter set per prediction time"}
    )
    comp_time: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={"units": "s", "description": "Computation time for each prediction step"}
    )

    # Optional metadata
    Nlead: Optional[int] = field(
        default=None, metadata={"description": "Wave lead time in samples"}
    )
    Theta: Optional[float] = field(
        default=None, metadata={"units": "deg", "description": "Dominant wave direction"}
    )
    Cp: Optional[float] = field(
        default=None, metadata={"units": "m/s", "description": "Phase speed"}
    )

    def to_netcdf(self, path: str) -> None:
        """
        Save Prediction to NetCDF.
        """
        def _force_1d(arr: np.ndarray) -> np.ndarray:
            # If it is Nx1 or 1xN, flatten to (N,)
            if arr.ndim == 2 and 1 in arr.shape:
                return arr.flatten()
            return arr

        # Fix vector fields
        self.tp = _force_1d(self.tp)
        self.zt = _force_1d(self.zt)
        self.ut = _force_1d(self.ut)
        self.vt = _force_1d(self.vt)
        self.zp = _force_1d(self.zp)
        self.comp_time = _force_1d(self.comp_time)

        # ---------------------------------------------------------
        # 1. Valid prediction indices (skip uninitialized params)
        # ---------------------------------------------------------
        valid = np.array([p.A.size > 0 for p in self.params])
        if not valid.any():
            raise RuntimeError("No valid predictions to save.")

        # prediction-time vectors (Np,)
        tp = self.tp[valid]
        zt = self.zt[valid] if self.zt.size else self.zt
        ut = self.ut[valid] if self.ut.size else self.ut
        vt = self.vt[valid] if self.vt.size else self.vt
        comp_time = self.comp_time[valid]

        # predicted at leave-one-out (Np,)
        zp = self.zp[valid]

        # ---------------------------------------------------------
        # 2. Measurement arrays (M × K)
        # ---------------------------------------------------------
        M, K = self.zm.shape     # K is measurement instruments (typically 3)

        zm = (("measurement_time", "measurement_instrument"), self.zm)
        zc = (("measurement_time", "measurement_instrument"), self.zc)
        um = (("measurement_time", "measurement_instrument"), self.um)
        vm = (("measurement_time", "measurement_instrument"), self.vm)
        uc = (("measurement_time", "measurement_instrument"), self.uc)
        vc = (("measurement_time", "measurement_instrument"), self.vc)

        # ---------------------------------------------------------
        # 3. Stack params across prediction_time
        # ---------------------------------------------------------
        params = np.array(self.params)[valid]

        param_A = np.stack([p.A for p in params], axis=0)
        param_f = np.stack([p.f for p in params], axis=0)
        param_theta = np.stack([p.theta for p in params], axis=0)
        param_Etheta = np.stack([p.Etheta for p in params], axis=0)
        param_kx = np.stack([p.kx for p in params], axis=0)
        param_ky = np.stack([p.ky for p in params], axis=0)
        param_omega = np.stack([p.omega for p in params], axis=0)
        param_use_vel = np.array([int(p.use_vel) for p in params])

        # shapes:
        #   param_Etheta: (Np, F, D)
        #   param_A:      (Np, C)
        #   param_f:      (Np, F)
        #   param_theta:  (Np, D)
        #   param_kx:     (Np, FD)
        # etc.

        # ---------------------------------------------------------
        # 4. Coordinates
        # ---------------------------------------------------------
        coords = {
            "prediction_time": np.arange(tp.size),             # Np
            "measurement_time": np.arange(M),                  # M
            "measurement_instrument": np.arange(K),            # K
            "components": np.arange(param_A.shape[1]),         # C
            "frequency": np.arange(param_f.shape[1]),          # F
            "direction": np.arange(param_theta.shape[1]),      # D
            "frequency_direction": np.arange(param_kx.shape[1])# F*D
        }

        # ---------------------------------------------------------
        # 5. Dataset assembly
        # ---------------------------------------------------------
        vars = {
            "tp": (("prediction_time",), tp),
            "zp": (("prediction_time",), zp),
            "zt": (("prediction_time",), zt) if zt.size else None,
            "ut": (("prediction_time",), ut) if ut.size else None,
            "vt": (("prediction_time",), vt) if vt.size else None,
            "comp_time": (("prediction_time",), comp_time),

            # measurement data
            "zm": zm,
            "zc": zc,
            "um": um,
            "vm": vm,
            "uc": uc,
            "vc": vc,

            # wave parameters
            "param_A": (("prediction_time", "components"), param_A),
            "param_f": (("prediction_time", "frequency"), param_f),
            "param_theta": (("prediction_time", "direction"), param_theta),
            "param_Etheta": (("prediction_time", "frequency", "direction"), param_Etheta),
            "param_kx": (("prediction_time", "frequency_direction"), param_kx),
            "param_ky": (("prediction_time", "frequency_direction"), param_ky),
            "param_omega": (("prediction_time", "frequency_direction"), param_omega),
            "param_use_vel": (("prediction_time",), param_use_vel),
        }

        # remove any None
        vars = {k: v for k, v in vars.items() if v is not None}

        # ---------------------------------------------------------
        # 6. Save
        # ---------------------------------------------------------
        ds = xr.Dataset(vars, coords=coords)
        ds.to_netcdf(path, engine="h5netcdf", invalid_netcdf=False)
        print(f"Saved prediction to {path}")


@dataclass
class WFA:
    x: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    y: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    lon: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    lat: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    x0: np.float64 = field(default=None)
    y0: np.float64 = field(default=None)
    lon0: np.float64 = field(default=None)
    lat0: np.float64 = field(default=None)
