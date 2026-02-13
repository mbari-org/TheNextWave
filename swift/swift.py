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
    '''
    Window-indexed container for measurements, reconstructions, and forecasts.

    Indexing:
        window_start_time: (W,) absolute seconds (same scale as tin)

    Stored per window:
        tm, zm, um, vm, xm, ym: (W, M, K) measurement time-series and sensor positions
        zc, uc, vc:             (W, M, K) reconstructions at sensors

        tp, zp, up, vp:         (W, L) target prediction series, padded to L=max_lead
        n_lead:                 (W,) number of valid lead samples in each row of tp/zp/...

        params:                 list length W, one LSQWavePropParams per window
        comp_time:              (W,) solve time per window
    '''

    window_start_time: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 's', 'description': 'Start time (left edge) of each measurement window'},
    )

    tm: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 's', 'description': 'Measurement times within each window (per buoy)'},
    )
    zm: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm', 'description': 'Measured vertical displacement at sensors'},
    )
    um: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm/s', 'description': 'Measured eastward velocity at sensors'},
    )
    vm: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm/s', 'description': 'Measured northward velocity at sensors'},
    )
    xm: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm', 'description': 'Sensor x positions within each window'},
    )
    ym: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm', 'description': 'Sensor y positions within each window'},
    )

    zc: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm', 'description': 'Reconstructed vertical displacement at sensors'},
    )
    uc: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm/s', 'description': 'Reconstructed eastward velocity at sensors'},
    )
    vc: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm/s', 'description': 'Reconstructed northward velocity at sensors'},
    )

    tp: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 's', 'description': 'Prediction times at target (padded per window)'},
    )
    zp: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm', 'description': 'Predicted surface elevation at target (padded per window)'},
    )
    up: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm/s', 'description': 'Predicted eastward velocity at target (padded per window)'},
    )
    vp: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 'm/s', 'description': 'Predicted northward velocity at target (padded per window)'},
    )

    n_lead: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32),
        metadata={'units': 'samples', 'description': 'Valid lead samples per window (tp/zp/up/vp)'},
    )

    comp_time: npt.NDArray[np.float64] = field(
        default_factory=empty_float64,
        metadata={'units': 's', 'description': 'Solve time per window'},
    )

    params: List['LSQWavePropParams'] = field(
        default_factory=list,
        metadata={'description': 'One LSQWavePropParams per window'},
    )

    # staging (not exported)
    _starts: List[float] = field(default_factory=list, init=False, repr=False)
    _tm: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _zm: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _um: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _vm: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _xm: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _ym: List[np.ndarray] = field(default_factory=list, init=False, repr=False)

    _zc: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _uc: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _vc: List[np.ndarray] = field(default_factory=list, init=False, repr=False)

    _tp: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _zp: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _up: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _vp: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _nlead: List[int] = field(default_factory=list, init=False, repr=False)

    _ct: List[float] = field(default_factory=list, init=False, repr=False)
    _params: List['LSQWavePropParams'] = field(default_factory=list, init=False, repr=False)

    def append_window(
        self,
        window_start_time: float,
        tm: np.ndarray,  # (M, K)
        zm: np.ndarray,  # (M, K)
        um: np.ndarray,  # (M, K)
        vm: np.ndarray,  # (M, K)
        xm: np.ndarray,  # (M, K)
        ym: np.ndarray,  # (M, K)
        zc: np.ndarray,  # (M, K)
        uc: np.ndarray,  # (M, K)
        vc: np.ndarray,  # (M, K)
        tp: np.ndarray,  # (Lw,)
        zp: np.ndarray,  # (Lw,)
        up: Optional[np.ndarray],
        vp: Optional[np.ndarray],
        params: 'LSQWavePropParams',
        comp_time: float,
    ) -> None:
        self._starts.append(float(window_start_time))

        self._tm.append(tm)
        self._zm.append(zm)
        self._um.append(um)
        self._vm.append(vm)
        self._xm.append(xm)
        self._ym.append(ym)

        self._zc.append(zc)
        self._uc.append(uc)
        self._vc.append(vc)

        tp = np.ravel(tp)
        zp = np.ravel(zp)

        self._tp.append(tp)
        self._zp.append(zp)

        self._up.append(np.ravel(up) if up is not None else np.full(tp.shape, np.nan, dtype=float))
        self._vp.append(np.ravel(vp) if vp is not None else np.full(tp.shape, np.nan, dtype=float))

        self._nlead.append(int(tp.size))

        self._params.append(params)
        self._ct.append(float(comp_time))

    def finalize(self) -> None:
        if not self._starts:
            return

        starts = np.array(self._starts, dtype=float)
        order = np.argsort(starts)

        self.window_start_time = starts[order]

        def _stack_3d(lst: List[np.ndarray]) -> np.ndarray:
            return np.stack([lst[i] for i in order], axis=0)

        self.tm = _stack_3d(self._tm)
        self.zm = _stack_3d(self._zm)
        self.um = _stack_3d(self._um)
        self.vm = _stack_3d(self._vm)
        self.xm = _stack_3d(self._xm)
        self.ym = _stack_3d(self._ym)

        self.zc = _stack_3d(self._zc)
        self.uc = _stack_3d(self._uc)
        self.vc = _stack_3d(self._vc)

        nlead = np.array([self._nlead[i] for i in order], dtype=np.int32)
        self.n_lead = nlead

        max_lead = int(np.max(nlead))
        W = self.window_start_time.size

        def _pad_2d(lst_1d: List[np.ndarray]) -> np.ndarray:
            out = np.full((W, max_lead), np.nan, dtype=float)
            for wi, idx in enumerate(order):
                v = np.ravel(lst_1d[idx])
                out[wi, :v.size] = v
            return out

        self.tp = _pad_2d(self._tp)
        self.zp = _pad_2d(self._zp)
        self.up = _pad_2d(self._up)
        self.vp = _pad_2d(self._vp)

        self.comp_time = np.array([self._ct[i] for i in order], dtype=float)
        self.params = [self._params[i] for i in order]

    @staticmethod
    def _apply_dataclass_metadata(ds: xr.Dataset) -> xr.Dataset:
        meta = {f.name: dict(f.metadata) for f in fields(Prediction) if f.metadata}

        for name in list(ds.data_vars):
            md = meta.get(name)
            if md:
                ds[name].attrs.update(md)

        for name in list(ds.coords):
            md = meta.get(name)
            if md:
                ds[name].attrs.update(md)

        return ds

    def to_netcdf(self, path: str) -> None:
        if self.window_start_time.size == 0:
            self.finalize()
        if self.window_start_time.size == 0:
            raise RuntimeError('No windows to save.')

        W, M, K = self.zm.shape
        L = self.tp.shape[1]

        coords = {
            'window_start_time': self.window_start_time,
            'measurement_sample': np.arange(M, dtype=np.int32),
            'buoy': np.arange(K, dtype=np.int32),
            'lead_sample': np.arange(L, dtype=np.int32),
        }

        vars = {
            'tm': (('window_start_time', 'measurement_sample', 'buoy'), self.tm),
            'zm': (('window_start_time', 'measurement_sample', 'buoy'), self.zm),
            'um': (('window_start_time', 'measurement_sample', 'buoy'), self.um),
            'vm': (('window_start_time', 'measurement_sample', 'buoy'), self.vm),
            'xm': (('window_start_time', 'measurement_sample', 'buoy'), self.xm),
            'ym': (('window_start_time', 'measurement_sample', 'buoy'), self.ym),

            'zc': (('window_start_time', 'measurement_sample', 'buoy'), self.zc),
            'uc': (('window_start_time', 'measurement_sample', 'buoy'), self.uc),
            'vc': (('window_start_time', 'measurement_sample', 'buoy'), self.vc),

            'tp': (('window_start_time', 'lead_sample'), self.tp),
            'zp': (('window_start_time', 'lead_sample'), self.zp),
            'up': (('window_start_time', 'lead_sample'), self.up),
            'vp': (('window_start_time', 'lead_sample'), self.vp),
            'n_lead': (('window_start_time',), self.n_lead),

            'comp_time': (('window_start_time',), self.comp_time),
        }

        ds = xr.Dataset(vars, coords=coords)

        # Add param fields (and give them sensible metadata too)
        if self.params:
            A = np.stack([p.A for p in self.params], axis=0)
            ds = ds.assign_coords(components=np.arange(A.shape[1], dtype=np.int32))
            ds['param_A'] = (('window_start_time', 'components'), A)
            ds['param_A'].attrs.update({'description': 'Solver amplitude vector A per window'})

            use_vel = np.array([int(p.use_vel) for p in self.params], dtype=np.int8)
            ds['param_use_vel'] = (('window_start_time',), use_vel)
            ds['param_use_vel'].attrs.update({'description': 'Whether velocity constraints were used (0/1)'})

            f = np.stack([p.f for p in self.params], axis=0)
            theta = np.stack([p.theta for p in self.params], axis=0)
            Etheta = np.stack([p.Etheta for p in self.params], axis=0)
            kx = np.stack([p.kx for p in self.params], axis=0)
            ky = np.stack([p.ky for p in self.params], axis=0)
            omega = np.stack([p.omega for p in self.params], axis=0)

            ds = ds.assign_coords(
                frequency=np.arange(f.shape[1], dtype=np.int32),
                direction=np.arange(theta.shape[1], dtype=np.int32),
                frequency_direction=np.arange(kx.shape[1], dtype=np.int32),
            )

            ds['param_f'] = (('window_start_time', 'frequency'), f)
            ds['param_f'].attrs.update({'units': 'Hz', 'description': 'Frequency bins used in param estimation'})

            ds['param_theta'] = (('window_start_time', 'direction'), theta)
            ds['param_theta'].attrs.update({'units': 'deg', 'description': 'Directional bins used in param estimation'})

            ds['param_Etheta'] = (('window_start_time', 'frequency', 'direction'), Etheta)
            ds['param_Etheta'].attrs.update({'units': 'm^2/Hz/deg', 'description': 'Directional energy density per window'})

            ds['param_kx'] = (('window_start_time', 'frequency_direction'), kx)
            ds['param_kx'].attrs.update({'units': '1/m', 'description': 'kx for each (f,theta) pair used in the solve'})

            ds['param_ky'] = (('window_start_time', 'frequency_direction'), ky)
            ds['param_ky'].attrs.update({'units': '1/m', 'description': 'ky for each (f,theta) pair used in the solve'})

            ds['param_omega'] = (('window_start_time', 'frequency_direction'), omega)
            ds['param_omega'].attrs.update({'units': 'rad/s', 'description': 'omega for each (f,theta) pair used in the solve'})

        ds = self._apply_dataclass_metadata(ds)
        ds.to_netcdf(path) #, engine='h5netcdf', invalid_netcdf=False)
        print(f'Saved windows to {path}')


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
