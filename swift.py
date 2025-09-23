from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Dict, Any

import numpy as np
import numpy.typing as npt


def empty_float64():
    return np.array([], dtype=np.float64)


@dataclass
class WaveSpec:
    Etheta: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    f: npt.NDArray[np.float64]
    spread: npt.NDArray[np.float64]
    spread2: npt.NDArray[np.float64]


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
    energy: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "m^2/Hz",
        "desc": "wave energy spectral density as a function of frequency (from IMU surface elevation)"
    })
    freq: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "Hz",
        "desc": "spectral frequencies"
    })
    a1: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "-",
        "desc": "normalized spectral directional moment (positive east)"
    })
    b1: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "-",
        "desc": "normalized spectral directional moment (positive north)"
    })
    a2: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "-",
        "desc": "normalized spectral directional moment (east-west)"
    })
    b2: npt.NDArray[np.float64] = field(default_factory=empty_float64, metadata={
        "units": "-",
        "desc": "normalized spectral directional moment (north-south)"
    })

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]):
        if not d:
            return cls()
        return cls(
            energy=np.array(d.get("energy", []), dtype=np.float64),
            freq=np.array(d.get("freq", []), dtype=np.float64),
            a1=np.array(d.get("a1", []), dtype=np.float64),
            b1=np.array(d.get("b1", []), dtype=np.float64),
            a2=np.array(d.get("a2", []), dtype=np.float64),
            b2=np.array(d.get("b2", []), dtype=np.float64),
        )


@dataclass
class SignatureProfile:
    altimeter: Optional[float] = field(default=None, metadata={
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
    spd_alt: Optional[float] = field(default=None, metadata={
        "units": "m/s",
        "desc": "burst-averaged scalar speed (not computed from averaged ENU velocities)"
    })

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]):
        if not d:
            return cls()
        return cls(
            altimeter=d.get("altimeter"),
            east=np.array((d.get("east", []), dtype=np.float64),
            north=np.array((d.get("north", []), dtype=np.float64),
            w=np.array((d.get("w", []), dtype=np.float64),
            z=np.array((d.get("z", []), dtype=np.float64),
            spd_alt=d.get("spd_alt"),
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
    def from_dict(cls, d: Optional[Dict[str, Any]]):
        if not d:
            return cls()
        return cls(
            w=np.array((d.get("w", []), dtype=np.float64),
            wvar=np.array((d.get("wvar", []), dtype=np.float64),
            tkedissipationrate=np.array((d.get("tkedissipationrate", []), dtype=np.float64),
            z=np.array((d.get("z", []), dtype=np.float64),
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
    def from_dict(cls, d: Optional[Dict[str, Any]]):
        if not d:
            return cls()
        return cls(
            profile=SignatureProfile.from_dict(d.get("profile") or d.get("Profile") or {}),
            HRprofile=SignatureHR.from_dict(d.get("HRprofile") or d.get("HRprofile") or {})
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
    def from_dict(cls, d: Optional[Dict[str, Any]]):
        if not d:
            return cls()
        return cls(
            tkedissipationrate=np.array((d.get("tkedissipationrate", []), dtype=np.float64),
            z=np.array((d.get("z", []), dtype=np.float64),
        )


@dataclass
class SWIFTData:
    # meteorological
    windspd: Optional[float] = field(default=None, metadata={
        "units": "m/s",
        "desc": "wind speed 1 m above the wave-following surface measured by MET sensor"
    })
    windspdstddev: Optional[float] = field(default=None, metadata={
        "units": "m/s",
        "desc": "standard deviation of wind speed"
    })
    winddirT: Optional[float] = field(default=None, metadata={
        "units": "degrees",
        "desc": "true wind direction (from North)"
    })
    winddirTstddev: Optional[float] = field(default=None, metadata={
        "units": "degrees",
        "desc": "standard deviation of true wind direction"
    })
    winddirR: Optional[float] = field(default=None, metadata={
        "units": "degrees",
        "desc": "relative wind direction (from North)"
    })
    winddirRstddev: Optional[float] = field(default=None, metadata={
        "units": "degrees",
        "desc": "standard deviation of relative wind direction"
    })
    airtemp: Optional[float] = field(default=None, metadata={
        "units": "deg C",
        "desc": "air temperature 1 m above the wave-following surface measured by MET sensor"
    })
    airtempstddev: Optional[float] = field(default=None, metadata={
        "units": "deg C",
        "desc": "standard deviation of air temperature"
    })
    airtpres: Optional[float] = field(default=None, metadata={
        "units": "mb",
        "desc": "air pressure 1 m above the wave-following surface measured by MET sensor"
    })
    airtpresstddev: Optional[float] = field(default=None, metadata={
        "units": "mb",
        "desc": "standard deviation of air pressure"
    })
    relhumidity: Optional[float] = field(default=None, metadata={
        "units": "%",
        "desc": "relative humidity 1 m above the wave-following surface measured by MET sensor"
    })
    relhumiditystddev: Optional[float] = field(default=None, metadata={
        "units": "%",
        "desc": "standard deviation of relative humidity"
    })
    radiancemean: Optional[float] = field(default=None, metadata={
        "units": "mV",
        "desc": "radiance measured by radiometer"
    })
    radiancestd: Optional[float] = field(default=None, metadata={
        "units": "mV",
        "desc": "standard deviation of radiance"
    })
    infraredtemp: Optional[float] = field(default=None, metadata={
        "units": "deg C",
        "desc": "(uncalibrated) target temperature inferred from radiance; should be close to true skin temperature"
    })
    infraredtempstd: Optional[float] = field(default=None, metadata={
        "units": "deg C",
        "desc": "standard deviation of target temperature"
    })
    ambienttemp: Optional[float] = field(default=None, metadata={
        "units": "deg C",
        "desc": "ambient temperature measured by radiometer"
    })
    ambienttempstd: Optional[float] = field(default=None, metadata={
        "units": "deg C",
        "desc": "standard deviation of ambient temperature"
    })

    # waves
    sigwaveheight: Optional[float] = field(default=None, metadata={
        "units": "m",
        "desc": "significant wave height estimated from wave energy spectrum"
    })
    peakwaveperiod: Optional[float] = field(default=None, metadata={
        "units": "s",
        "desc": "period corresponding to peak in wave energy spectrum"
    })
    peakwavedirT: Optional[float] = field(default=None, metadata={
        "units": "degrees",
        "desc": "wave direction (from North)"
    })
    wavespectra: WaveSpectra = field(default_factory=WaveSpectra, metadata={
        "desc": "structure containing IMU spectral wave data"
    })

    # CTD / water properties
    watertemp: Optional[float] = field(default=None, metadata={
        "units": "deg C",
        "desc": "water temperature 0.5 m below the surface, measured by CT"
    })
    watertempstddev: Optional[float] = field(default=None, metadata={
        "units": "deg C",
        "desc": "standard deviation of water temperature"
    })
    salinity: Optional[float] = field(default=None, metadata={
        "units": "PSU",
        "desc": "water salinity 0.5 m below the surface, measured by CT"
    })
    salinitystddev: Optional[float] = field(default=None, metadata={
        "units": "PSU",
        "desc": "standard deviation of water salinity"
    })

    # ADCP / signatures
    signature: Signature = field(default_factory=Signature, metadata={
        "desc": "structure containing Nortek Signature1000 HR ADCP data (downlooking configuration)"
    })
    uplooking: Uplooking = field(default_factory=Uplooking, metadata={
        "desc": "structure containing Nortek Aquadopp HR ADCP data (uplooking configuration)"
    })

    # metadata / geolocation
    time: Optional[float] = field(default=None, metadata={
        "units": "days (MATLAB datenum)",
        "desc": "MATLAB datenum time"
    })
    date: Optional[str] = field(default=None, metadata={
        "units": "-",
        "desc": "string giving burst date in format 'ddmmyyyy'"
    })
    lat: Optional[float] = field(default=None, metadata={
        "units": "deg",
        "desc": "latitude"
    })
    lon: Optional[float] = field(default=None, metadata={
        "units": "deg",
        "desc": "longitude"
    })
    driftdirT: Optional[float] = field(default=None, metadata={
        "units": "degrees",
        "desc": "true drift direction TOWARDS (equivalent to 'course over ground')"
    })
    driftspd: Optional[float] = field(default=None, metadata={
        "units": "m/s",
        "desc": "drift speed in m/s (equivalent to 'speed over ground')"
    })
    sbdfile: Optional[str] = field(default=None, metadata={
        "units": "-",
        "desc": "short-burst data file"
    })
    burstID: Optional[str] = field(default=None, metadata={
        "units": "-",
        "desc": "burstID named by burst timestamp, consistent with raw sensor burst files"
    })
    battery: Optional[float] = field(default=None, metadata={
        "units": "V",
        "desc": "battery voltage"
    })
    ID: Optional[str] = field(default=None, metadata={
        "units": "-",
        "desc": "SWIFT ID"
    })
    metheight: Optional[float] = field(default=None, metadata={
        "units": "m",
        "desc": "height of the MET sensor"
    })
    CTdepth: Optional[float] = field(default=None, metadata={
        "units": "m",
        "desc": "depth of the CT sensor"
    })

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]):
        """Construct a SWIFT from a nested dict (e.g. from JSON or MATLAB->dict)."""
        if not d:
            return cls()
        return cls(
            windspd=d.get("windspd"),
            windspdstddev=d.get("windspdstddev"),
            winddirT=d.get("winddirT"),
            winddirTstddev=d.get("winddirTstddev"),
            winddirR=d.get("winddirR"),
            winddirRstddev=d.get("winddirRstddev"),
            airtemp=d.get("airtemp"),
            airtempstddev=d.get("airtempstddev"),
            airtpres=d.get("airtpres"),
            airtpresstddev=d.get("airtpresstddev"),
            relhumidity=d.get("relhumidity"),
            relhumiditystddev=d.get("relhumiditystddev"),
            radiancemean=d.get("radiancemean"),
            radiancestd=d.get("radiancestd"),
            infraredtemp=d.get("infraredtemp"),
            infraredtempstd=d.get("infraredtempstd"),
            ambienttemp=d.get("ambienttemp"),
            ambienttempstd=d.get("ambienttempstd"),
            sigwaveheight=d.get("sigwaveheight"),
            peakwaveperiod=d.get("peakwaveperiod"),
            peakwavedirT=d.get("peakwavedirT"),
            wavespectra=WaveSpectra.from_dict(d.get("wavespectra") or {}),
            watertemp=d.get("watertemp"),
            watertempstddev=d.get("watertempstddev"),
            salinity=d.get("salinity"),
            salinitystddev=d.get("salinitystddev"),
            signature=Signature.from_dict(d.get("signature") or {}),
            uplooking=Uplooking.from_dict(d.get("uplooking") or {}),
            time=d.get("time"),
            date=d.get("date"),
            lat=d.get("lat"),
            lon=d.get("lon"),
            driftdirT=d.get("driftdirT"),
            driftspd=d.get("driftspd"),
            sbdfile=d.get("sbdfile"),
            burstID=d.get("burstID"),
            battery=d.get("battery"),
            ID=d.get("ID"),
            metheight=d.get("metheight"),
            CTdepth=d.get("CTdepth"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert SWIFTData (and nested dataclasses) to a nested plain dict."""
        return asdict(self)


@dataclass
class SWIFTArray:
    swift22: SWIFTData = field(default_factory=SWIFTData)
    swift23: SWIFTData = field(default_factory=SWIFTData)
    swift24: SWIFTData = field(default_factory=SWIFTData)
    swift25: SWIFTData = field(default_factory=SWIFTData)


@dataclass
class LSQWavePropParams:
    A: npt.NDArray[np.float64]
    Etheta: npt.NDArray[np.float64]
    f: npt.NDArray[np.float64]
    theta: npt.NDArray[np.float64]
    kx: npt.NDArray[np.float64]
    ky: npt.NDArray[np.float64]
    omega: npt.NDArray[np.float64]
    use_vel: bool


@dataclass
class Prediction:
    """
    prediction-related data.

    Typical shapes (MATLAB equivalents):
      - tp: (T,)            # prediction times (seconds since t0)
      - tm: (M, K)          # measurement times (M samples, K instruments)
      - zm, zc, um, vm, uc, vc: same shape as tm
      - zp: (T, ...)        # predicted field; 1D, 2D or 3D depending on config
      - up, vp: optional predicted u/v (same shape rules as zp)
      - zt, ut, vt: (T,)    # truth/test arrays (sampled or per-target)
      - params: (T,)        # solver parameters per prediction time (or per-target)
      - comp_time: (T,)     # computation time per prediction
    """

    tp: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    tm: npt.NDArray[np.float64] = field(default_factory=empty_float64)

    zm: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    zc: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    um: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    vm: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    uc: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    vc: npt.NDArray[np.float64] = field(default_factory=empty_float64)

    zp: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    up: Optional[npt.NDArray[np.float64]] = None
    vp: Optional[npt.NDArray[np.float64]] = None

    zt: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    ut: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    vt: npt.NDArray[np.float64] = field(default_factory=empty_float64)

    params: List[LSQWavePropParams] = field(default_factory=list)
    comp_time: npt.NDArray[np.float64] = field(default_factory=empty_float64)

    # Optional metadata
    Nlead: Optional[int] = None
    Theta: Optional[float] = None
    Cp: Optional[float] = None


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
