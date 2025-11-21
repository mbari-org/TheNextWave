from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.io as spio
import xarray as xr


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

    @classmethod
    def from_dataset(cls, ds: xr.Dataset = None):
        if not ds:
            return cls()
        return cls(
            freq=np.array(ds.coords.get('freq', empty_float64)),
            check=np.array(ds.data_vars.get('check', empty_float64)),
            energy=np.array(ds.data_vars.get('energy', empty_float64)),
            energy_alt=np.array(ds.data_vars.get('energy_alt', empty_float64)),
            a1=np.array(ds.data_vars.get('a1', empty_float64)),
            b1=np.array(ds.data_vars.get('b1', empty_float64)),
            a2=np.array(ds.data_vars.get('a2', empty_float64)),
            b2=np.array(ds.data_vars.get('b2', empty_float64))
        )


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

    @classmethod
    def from_dataset(cls, ds: xr.Dataset = None):
        if not ds:
            return cls()
        return cls(
            # TODO signatureProfile_ prefix added to not collide with others since I don't have data with
            # these fields to know how they are named
            altimeter=np.array(ds.data_vars.get('signatureProfile_altimeter', empty_float64())),
            east=np.array(ds.data_vars.get('signatureProfile_east', empty_float64())),
            north=np.array(ds.data_vars.get('signatureProfile_north', empty_float64())),
            w=np.array(ds.data_vars.get('signatureProfile_w', empty_float64())),
            z=np.array(ds.data_vars.get('signatureProfile_z', empty_float64())),
            spd_alt=np.array(ds.data_vars.get('signatureProfile_spd_alt', empty_float64())),
        )


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

    @classmethod
    def from_dataset(cls, ds: xr.Dataset = None):
        if not ds:
            return cls()
        return cls(
            # TODO signatureHR_ prefix added to not collide with others since I don't have data with
            # these fields to know how they are named
            w=np.array(ds.data_vars.get('signatureHR_w', empty_float64())),
            wvar=np.array(ds.data_vars.get('signatureHR_wvar', empty_float64())),
            tkedissipationrate=np.array(ds.data_vars.get('signatureHR_tkedissipationrate', empty_float64())),
            z=np.array(ds.data_vars.get('signatureHR_z', empty_float64())),
        )


@dataclass
class Signature:
    profile: SignatureProfile = field(default_factory=SignatureProfile, metadata={
        "desc": "broadband profile data (downlooking Signature1000 configuration)"
    })
    HRprofile: SignatureHR = field(default_factory=SignatureHR, metadata={
        "desc": "high-resolution (pulse-coherent) profile data"
    })

    @classmethod
    def from_dataset(cls, ds: xr.Dataset = None):
        if not ds:
            return cls()
        return cls(
            profile=SignatureProfile.from_dataset(ds),
            HRprofile=SignatureHR.from_dataset(ds)
        )


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

    @classmethod
    def from_dataset(cls, ds: xr.Dataset = None):
        if not ds:
            return cls()
        return cls(
            # TODO uplooking_ prefix added to not collide with others since I don't have data with
            # these fields to know how they are named
            tkedissipationrate=np.array(ds.data_vars.get('uplooking_tkedissipationrate', empty_float64())),
            z=np.array(ds.data_vars.get('uplooking_z', empty_float64())),
        )


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

    @classmethod
    def from_mat(cls, matfile, name):
        print(f'loading {matfile} ...')
        mat = spio.loadmat(matfile)
        swiftmat = mat['SWIFT']
        swiftdtype = swiftmat.dtype
        swiftdata = {n: np.vstack(swiftmat[n][0]) for n in swiftdtype.names if 'wavespectra' not in n}
        swiftwsmat = swiftmat['wavespectra'][0]
        swiftwsdtype = swiftwsmat[0].dtype
        swiftwsdata = {n: np.hstack([ws[n][0, 0] for ws in swiftwsmat]).T for n in swiftwsdtype.names if 'check' not in n and 'freq' not in n}
        swiftwsdata.update({'check': np.vstack([ws['check'][0, 0] for ws in swiftwsmat])})
        swiftdata.update(**swiftwsdata)
        freq = swiftwsmat[0][0]['freq'][0]
        swiftcolumns = [n for n, v in swiftdata.items() if v.size > 1]
        #swiftattrs = [n for n, v in swiftdata.items() if v.size == 1]

        data_vars={
            'u': (('time', 'raw_idx'), swiftdata['u']),
            'v': (('time', 'raw_idx'), swiftdata['v']),
            'x': (('time', 'raw_idx'), swiftdata['x'].astype(np.float64)),
            'y': (('time', 'raw_idx'), swiftdata['y'].astype(np.float64)),
            'z': (('time', 'raw_idx'), swiftdata['z']),
            'energy': (('time', 'freq'), swiftdata['energy']),
            'a1': (('time', 'freq'), swiftdata['a1']),
            'b1': (('time', 'freq'), swiftdata['b1']),
            'a2': (('time', 'freq'), swiftdata['a2']),
            'b2': (('time', 'freq'), swiftdata['b2']),
            'energy_alt': (('time', 'freq'), swiftdata['energy_alt']),
            'check': (('time', 'check_idx'), swiftdata['check']),
        }
        data_vars.update({n: (('time',), swiftdata[n].ravel(order='F')) for n in swiftcolumns \
            if n not in ['time','rawtime','freq','u','v','x','y','z', 'signature',  # TODO signature has types 'HRprofile','profile',
                         'energy','a1','b1','a2','b2','energy_alt','check']})

        ds = xr.Dataset(
            data_vars=dict(**data_vars),
            coords={
                'rawtime': (('time', 'raw_idx'), pd.to_datetime(swiftdata['rawtime'] - 719529, unit='D').to_numpy()),
                'time': pd.to_datetime(swiftdata['time'].ravel(order='F') - 719529, unit='D').to_numpy(),
                'freq': freq.ravel(order='F'),
            },
            attrs={'name': name}
        )

        pth = Path(matfile)
        ds.to_netcdf(str(Path.joinpath(*[Path(p) for p in list(pth.parts[:-1]) + [pth.stem+'.nc']])))

        return cls.from_dataset(ds)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset = None):
        """Construct a SWIFT from a NetCDF Xarray Dataset."""
        if not ds:
            return cls()
        return cls(
            time=np.array(ds.coords.get('time', empty_float64())),
            rawtime=np.array(ds.coords.get('rawtime', empty_float64())),
            u=np.array(ds.data_vars.get('u', empty_float64())),
            v=np.array(ds.data_vars.get('v', empty_float64())),
            x=np.array(ds.data_vars.get('x', empty_float64())),
            y=np.array(ds.data_vars.get('y', empty_float64())),
            z=np.array(ds.data_vars.get('z', empty_float64())),
            wavespectra=WaveSpectra.from_dataset(ds),
            CTdepth=np.array(ds.data_vars.get('CTdepth', empty_float64())),
            ID=np.array(swiftid).astype(int) if (swiftid:=ds.data_vars.get('ID', empty_int())).size>0 else empty_int(),
            airpres=np.array(ds.data_vars.get('airpres', empty_float64())),
            airpresstddev=np.array(ds.data_vars.get('aipresstdev', empty_float64())),
            airtemp=np.array(ds.data_vars.get('airtemp', empty_float64())),
            airtempstddev=np.array(ds.data_vars.get('airtempstddev', empty_float64())),
            date=np.array(ds.data_vars.get('date', empty_float64())),
            driftdirT=np.array(ds.data_vars.get('driftdirT', empty_float64())),
            driftdirTstddev=np.array(ds.data_vars.get('driftdirTstddev', empty_float64())),
            driftspd=np.array(ds.data_vars.get('driftspd', empty_float64())),
            driftspdstddev=np.array(ds.data_vars.get('driftspdstddev', empty_float64())),
            lat=np.array(ds.data_vars.get('lat', empty_float64())),
            lon=np.array(ds.data_vars.get('lon', empty_float64())),
            metheight=np.array(ds.data_vars.get('metheight', empty_float64())),
            peakwavedirT=np.array(ds.data_vars.get('peakwavedirT', empty_float64())),
            peakwaveperiod=np.array(ds.data_vars.get('peakwaveperiod', empty_float64())),
            salinity=np.array(ds.data_vars.get('salinity', empty_float64())),
            sigwaveheight=np.array(ds.data_vars.get('sigwaveheight', empty_float64())),
            watertemp=np.array(ds.data_vars.get('watertemp', empty_float64())),
            winddirR=np.array(ds.data_vars.get('winddirR', empty_float64())),
            winddirRstddev=np.array(ds.data_vars.get('winddirRstddev', empty_float64())),
            winddirT=np.array(ds.data_vars.get('winddirT', empty_float64())),
            winddirTstddev=np.array(ds.data_vars.get('winddirTstddev', empty_float64())),
            windspd=np.array(ds.data_vars.get('windspd', empty_float64())),
            windspdstddev=np.array(ds.data_vars.get('windspdstddev', empty_float64())),
            sigwaveheight_alt=np.array(ds.data_vars.get('sigwaveheight_alt', empty_float64())),
            peakwaveperiod_alt=np.array(ds.data_vars.get('peakwaveperiod_alt', empty_float64())),
            signature=Signature.from_dataset(ds),  # TODO fix this in the from_mat func
            uplooking=Uplooking.from_dataset(ds),
        )


@dataclass
class SWIFTArray:
    swift22: SWIFTData = field(default_factory=SWIFTData)
    swift23: SWIFTData = field(default_factory=SWIFTData)
    swift24: SWIFTData = field(default_factory=SWIFTData)
    swift25: SWIFTData = field(default_factory=SWIFTData)

    def bursts(self) -> "SWIFTArray":
        idx = 0
        try:
            while True:
                yield self.burst(idx)
                idx += 1
        except IndexError as err:
            # print(err)
            pass


    def burst(self, idx: int) -> "SWIFTArray":
        """
        Return a new SWIFTArray containing only the burst at index `idx` (0-based).
        - Uses `time` to infer burst count.
        - For 1D numpy arrays: returns the single scalar value at index idx.
        - For 2D+ numpy arrays: selects the slice at index idx along axis 0.
        - Recurses into nested dataclasses.
        """
        def _select_value(val, i):
            # nested dataclass -> recurse
            if is_dataclass(val):
                kwargs = {}
                for f in fields(type(val)):
                    v = getattr(val, f.name)
                    if 'freq' in f.name:
                        kwargs[f.name] = v.copy()
                        continue
                    kwargs[f.name] = _select_value(v, i)
                return type(val)(**kwargs)

            # numpy arrays
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    return val.copy()
                if val.ndim == 0:
                    return val.copy()  # scalar array
                # 1D -> return a single value (numpy scalar)
                if val.ndim == 1:
                    return val[int(i)].copy()
                # 2D or higher -> select along first axis and return remaining dims
                return val[int(i)].copy()

            # lists/tuples: try indexing
            if isinstance(val, (list, tuple)):
                try:
                    return val[int(i)]
                except Exception:
                    return val

            # scalars/None/other -> return as-is
            return val

        def _single_swift(sw: "SWIFTData", i: int) -> "SWIFTData":
            kwargs = {}
            for f in fields(SWIFTData):
                v = getattr(sw, f.name)
                kwargs[f.name] = _select_value(v, i)
            return SWIFTData(**kwargs)

        # infer bursts from `time`
        ref = getattr(self.swift22, "time", None)
        if isinstance(ref, np.ndarray) and ref.size > 0:
            n_bursts = int(ref.shape[0])
            if not (-n_bursts <= idx < n_bursts):
                raise IndexError(f"idx {idx} out of bounds for bursts (len={n_bursts})")

        s22 = _single_swift(self.swift22, idx)
        s23 = _single_swift(self.swift23, idx)
        s24 = _single_swift(self.swift24, idx)
        s25 = _single_swift(self.swift25, idx)

        return SWIFTArray(swift22=s22, swift23=s23, swift24=s24, swift25=s25)


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


    '''
    def to_netcdf(self, path: str):
        """
        Write the Prediction object to a NetCDF file.
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

        # -------------------------------------------------------
        # 1. Identify valid prediction times (those with params)
        # -------------------------------------------------------
        valid = np.array([p.A.size > 0 for p in self.params])
        if not valid.any():
            raise RuntimeError("No valid predictions found to save.")

        # Filter all time-based arrays
        tp = self.tp[valid]
        zp = self.zp[valid]
        zt = self.zt[valid]
        ut = self.ut[valid] if self.ut.size else self.ut
        vt = self.vt[valid] if self.vt.size else self.vt

        comp_time = self.comp_time[valid]

        params_valid = [self.params[i] for i in np.where(valid)[0]]

        # -------------------------------------------------------
        # 2. Determine parameter sizes
        # -------------------------------------------------------
        Np = len(params_valid)

        # All params share the same dimensionality (40 freq, 25 dir, 1000 kd)
        F = params_valid[0].f.size
        D = params_valid[0].theta.size
        K = params_valid[0].kx.size
        A_len = params_valid[0].A.size  # 2000

        # -------------------------------------------------------
        # 3. Preallocate parameter arrays
        # -------------------------------------------------------
        A_arr      = np.full((Np, A_len), np.nan)
        Etheta_arr = np.full((Np, F, D), np.nan)
        kx_arr     = np.full((Np, K), np.nan)
        ky_arr     = np.full((Np, K), np.nan)
        omega_arr  = np.full((Np, K), np.nan)

        # -------------------------------------------------------
        # 4. Fill arrays with actual parameter values
        # -------------------------------------------------------
        for i, p in enumerate(params_valid):
            A_arr[i, :]       = p.A
            Etheta_arr[i, :, :] = p.Etheta      # shape (40, 25)
            kx_arr[i, :]      = p.kx
            ky_arr[i, :]      = p.ky
            omega_arr[i, :]   = p.omega

        # -------------------------------------------------------
        # 5. Build single xarray.Dataset
        # -------------------------------------------------------
        ds = xr.Dataset(
            {
                # -------- MAIN PREDICTION OUTPUTS --------
                "tp": ("prediction_time", tp,
                       {"units": "s", "description": "Prediction times (s since t0)"}),

                "zp": ("prediction_time", zp,
                       {"units": "m", "description": "Predicted surface elevation"}),

                "zt": ("prediction_time", zt,
                       {"units": "m", "description": "Truth/verification surface elevation"}),

                "ut": ("prediction_time", ut,
                       {"units": "m/s"}),

                "vt": ("prediction_time", vt,
                       {"units": "m/s"}),

                "comp_time": ("prediction_time", comp_time,
                              {"units": "s", "description": "Prediction compute time"}),

                # -------- PARAMETER ARRAYS --------
                "A": (("prediction_time", "components"), A_arr,
                      {"description": "Solved wave amplitude components"}),

                "Etheta": (("prediction_time", "frequency", "direction"), Etheta_arr,
                           {"units": "m^2/Hz/deg",
                            "description": "Directional energy spectrum"}),

                "kx": (("prediction_time", "frequency_direction"), kx_arr,
                       {"units": "1/m", "description": "kx for each freq×dir component"}),

                "ky": (("prediction_time", "frequency_direction"), ky_arr,
                       {"units": "1/m", "description": "ky for each freq×dir component"}),

                "omega": (("prediction_time", "frequency_direction"), omega_arr,
                          {"units": "rad/s", "description": "Wave angular frequency"}),
            },

            coords={
                "prediction_time": np.arange(Np),
                "frequency": np.arange(F),
                "direction": np.arange(D),
                "instrument": np.arange(self.zp.shape[1]),
                "components": np.arange(A_len),
                "frequency_direction": np.arange(K),
            }
        )

        # -------------------------------------------------------
        # 6. Write to NetCDF
        # -------------------------------------------------------
        print(f"Writing NetCDF → {path}")
        ds.to_netcdf(path)
        print("Done.")
    '''

    '''
    def to_netcdf(self, path: str) -> None:
        """
        Save prediction results and per-prediction LSQ parameters to a NetCDF file.
        params[] is stored using NetCDF groups, one per prediction time,
        because shapes differ (due to removal of zero-energy components).
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

        # -----------------------------
        # 1. Write the main Prediction dataset
        # -----------------------------
        main_vars = {}
        for fname, finfo in self.__dataclass_fields__.items():
            arr = getattr(self, fname)

            if not isinstance(arr, np.ndarray):
                continue

            if fname == "tp":
                dims = ("prediction_time",)
            elif fname == "tm":
                dims = ("measurement_time", "instrument")
            elif fname in ("zm", "zc", "um", "vm", "uc", "vc"):
                dims = ("measurement_time", "instrument")
            elif fname == "zp":
                dims = ("prediction_time")
            elif fname in ("zt", "ut", "vt"):
                dims = ("prediction_time",)
            else:
                # fallback
                dims = tuple(f"{fname}_dim{i}" for i in range(arr.ndim))

            attrs = {
                "units": finfo.metadata.get("units", ""),
                "description": finfo.metadata.get("description", "")
            }

            main_vars[fname] = (dims, arr, attrs)

        ds_main = xr.Dataset(main_vars)
        ds_main.to_netcdf(path)

        # -----------------------------
        # 2. Store per-prediction params as groups
        # -----------------------------
        if self.params:
            for i, p in enumerate(self.params):

                var_dict = {}
                for pname, pinfo in LSQWavePropParams.__dataclass_fields__.items():
                    val = getattr(p, pname)

                    if isinstance(val, np.ndarray):
                        # Build semantic dims
                        if pname == "Etheta":
                            dims = ("frequency", "direction")
                        elif pname == "f":
                            dims = ("frequency",)
                        elif pname == "theta":
                            dims = ("direction",)
                        elif pname in ("kx", "ky", "omega"):
                            dims = ("frequency_direction",)
                        elif pname == "A":
                            dims = ("components",)
                        else:
                            dims = ("scalar",)
                    else:
                        # booleans etc.
                        val = np.array([val])
                        dims = ("scalar",)

                    attrs = {
                        "units": pinfo.metadata.get("units", ""),
                        "description": pinfo.metadata.get("description", "")
                    }

                    var_dict[pname] = (dims, val, attrs)

                ds_p = xr.Dataset(var_dict)
                ds_p.to_netcdf(path, mode="a", group=f"params/prediction_{i:03d}")
    '''

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


"""
if __name__ == "__main__":
    s = SWIFTData()
    s.windspd = 5.2
    s.wavespectra.energy = [0.1, 0.2, 0.05]
    s.signature.profile.z = [0.5, 1.0, 1.5]
    s.signature.HRprofile.tkedissipationrate = [1e-6, 2e-6]

    print("windspd:", b.windspd)
    # access metadata for top-level fields
    print("SWIFT metadata (windspd):", _get_field_meta(SWIFTData).get("windspd"))
    # nested metadata (class-level)
    print("WaveSpectra metadata:", _get_field_meta(WaveSpectra))
    # recursive metadata
    import pprint
    pprint.pprint(recursive_metadata(s))

    # conversion to/from dict
    d = s.to_dict()
    s2 = SWIFT.from_dict(d)
    print("s2.wavespectra.energy:", s2.wavespectra.energy)
"""
