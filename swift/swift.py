from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional, Dict, Any

import numpy as np
import numpy.typing as npt

import xarray as xr


def empty_float64():
    return np.array([], dtype=np.float64)


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
    # TODO need to understand time units of the sample data I have
    # e.g.
    #    <xarray.DataArray 'time' (time: 549)> Size: 4kB
    #    array([19243.458333, 19243.5     , 19243.541667, ..., 19269.25    ,
    #           19269.291667, 19269.333333])
    # Coordinates:
    #  * time     (time) float64 4kB 1.924e+04 1.924e+04 ... 1.927e+04 1.927e+04
    time: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "days (MATLAB datenum)",
        "desc": "MATLAB datenum time"
    })

    trajectory: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "TODO",
        "desc": "TODO"
    })

    sbg_x: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "meters",
        "desc": "TODO"
    })
    sbg_y: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "meters",
        "desc": "TODO"
    })
    utmzone: Optional[int] = field(default=None, metadata={
        "units": "-",
        "desc": "UTM Zone"
    })

    air_pressure: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "mb",
        "desc": "air pressure 1 m above the wave-following surface measured by MET sensor"
    })
    air_pressure_stddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "mb",
        "desc": "standard deviation of air pressure"
    })

    air_temperature: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg C",
        "desc": "air temperature 1 m above the wave-following surface measured by MET sensor"
    })
    air_temperature_stddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg C",
        "desc": "standard deviation of air temperature"
    })

    drift_direction: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "degrees",
        "desc": "true drift direction TOWARDS (equivalent to 'course over ground')"
    })
    driftdirTstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "degrees",
        "desc": "stddev true drift direction TOWARDS (equivalent to 'course over ground')"
    })
    drift_speed: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m/s",
        "desc": "drift speed in m/s (equivalent to 'speed over ground')"
    })
    driftspdstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m/s",
        "desc": "stddev drift speed in m/s (equivalent to 'speed over ground')"
    })

    lat: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg",
        "desc": "latitude"
    })
    lon: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg",
        "desc": "longitude"
    })

    metheight: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m",
        "desc": "height of the MET sensor"
    })

    peak_wave_direction: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "degrees",
        "desc": "wave direction (from North)"
    })
    peak_wave_period: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "s",
        "desc": "period corresponding to peak in wave energy spectrum"
    })

    significant_wave_height: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m",
        "desc": "significant wave height estimated from wave energy spectrum"
    })

    wavespectra: WaveSpectra = field(default_factory=WaveSpectra, metadata={
        "desc": "structure containing IMU spectral wave data"
    })

    winddirR: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "degrees",
        "desc": "relative wind direction (from North)"
    })
    winddirRstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "degrees",
        "desc": "standard deviation of relative wind direction"
    })
    wind_direction: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "degrees",
        "desc": "true wind direction (from North)"
    })
    wind_direction_stddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "degrees",
        "desc": "standard deviation of true wind direction"
    })
    wind_speed: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m/s",
        "desc": "wind speed 1 m above the wave-following surface measured by MET sensor"
    })
    wind_speed_stddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m/s",
        "desc": "standard deviation of wind speed"
    })

    sigwaveheight_alt: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "m",
        "desc": "significant wave height estimated from wave energy spectrum"
    })

    peakwaveperiod_alt: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "s",
        "desc": "period corresponding to peak in wave energy spectrum"
    })

    sea_water_temperature: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg C",
        "desc": "water temperature 0.5 m below the surface, measured by CT"
    })

    sea_water_salinity: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "PSU",
        "desc": "water salinity 0.5 m below the surface, measured by CT"
    })

    ID: Optional[int] = field(default=None, metadata={
        "units": "-",
        "desc": "SWIFT ID"
    })

    ### TODO the rest of these are not in the sample data I have
    relhumidity: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "%",
        "desc": "relative humidity 1 m above the wave-following surface measured by MET sensor"
    })
    relhumiditystddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "%",
        "desc": "standard deviation of relative humidity"
    })
    radiancemean: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "mV",
        "desc": "radiance measured by radiometer"
    })
    radiancestd: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "mV",
        "desc": "standard deviation of radiance"
    })
    infraredtemp: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg C",
        "desc": "(uncalibrated) target temperature inferred from radiance; should be close to true skin temperature"
    })
    infraredtempstd: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg C",
        "desc": "standard deviation of target temperature"
    })
    ambienttemp: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg C",
        "desc": "ambient temperature measured by radiometer"
    })
    ambienttempstd: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg C",
        "desc": "standard deviation of ambient temperature"
    })

    watertempstddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "deg C",
        "desc": "standard deviation of water temperature"
    })
    salinitystddev: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "PSU",
        "desc": "standard deviation of water salinity"
    })

    signature: Signature = field(default_factory=Signature, metadata={
        "desc": "structure containing Nortek Signature1000 HR ADCP data (downlooking configuration)"
    })
    uplooking: Uplooking = field(default_factory=Uplooking, metadata={
        "desc": "structure containing Nortek Aquadopp HR ADCP data (uplooking configuration)"
    })

    date: Optional[str] = field(default=None, metadata={
        "units": "-",
        "desc": "string giving burst date in format 'ddmmyyyy'"
    })
    sbdfile: Optional[str] = field(default=None, metadata={
        "units": "-",
        "desc": "short-burst data file"
    })
    burstID: Optional[str] = field(default=None, metadata={
        "units": "-",
        "desc": "burstID named by burst timestamp, consistent with raw sensor burst files"
    })
    battery: npt.NDArray[np.float64] = field(default=empty_float64, metadata={
        "units": "V",
        "desc": "battery voltage"
    })
    CTdepth: npt.NDArray[np.float64] = field(default=None, metadata={
        "units": "m",
        "desc": "depth of the CT sensor"
    })

    @classmethod
    def from_dataset(cls, ds: xr.Dataset = None):
        """Construct a SWIFT from a NetCDF Xarray Dataset."""
        if not ds:
            return cls()
        return cls(
            time=np.array(ds.coords.get("time", empty_float64())),
            trajectory=np.array(ds.data_vars.get("trajectory", empty_float64())),
            sbg_x=np.array(ds.data_vars.get("sbg_x", empty_float64())),
            sbg_y=np.array(ds.data_vars.get("sbg_y", empty_float64())),
            utmzone=int(utmzone) if (utmzone:=ds.data_vars.get("utmzone", None)) else None,
            air_pressure=np.array(ds.data_vars.get("air_pressure", empty_float64())),
            air_pressure_stddev=np.array(ds.data_vars.get("air_pressure_stddev", empty_float64())),
            drift_direction=np.array(ds.data_vars.get("drift_direction", empty_float64())),
            driftdirTstddev=np.array(ds.data_vars.get("driftdirTstddev", empty_float64())),
            drift_speed=np.array(ds.data_vars.get("drift_speed", empty_float64())),
            driftspdstddev=np.array(ds.data_vars.get("driftspdstddev", empty_float64())),
            lat=np.array(ds.data_vars.get("lat", empty_float64())),
            lon=np.array(ds.data_vars.get("lon", empty_float64())),
            metheight=np.array(ds.data_vars.get("metheight", empty_float64())),
            peak_wave_direction=np.array(ds.data_vars.get("peak_wave_direction", empty_float64())),
            peak_wave_period=np.array(ds.data_vars.get("peak_wave_period", empty_float64())),
            significant_wave_height=np.array(ds.data_vars.get("significant_wave_height", empty_float64())),
            wavespectra=WaveSpectra.from_dataset(ds),
            #wavespectra.freq
            #wavespectra.energy
            #wavespectra.a1
            #wavespectra.b1
            #wavespectra.a2
            #wavespectra.b2
            #wavespectra.check
            #wavespectra.energy_alt
            winddirR=np.array(ds.data_vars.get("winddirR", empty_float64())),
            winddirRstddev=np.array(ds.data_vars.get("winddirRstddev", empty_float64())),
            wind_direction=np.array(ds.data_vars.get("wind_direction", empty_float64())),
            wind_direction_stddev=np.array(ds.data_vars.get("wind_direction_stddev", empty_float64())),
            wind_speed=np.array(ds.data_vars.get("wind_speed", empty_float64())),
            wind_speed_stddev=np.array(ds.data_vars.get("wind_speed_stddev", empty_float64())),
            sigwaveheight_alt=np.array(ds.data_vars.get("sigwaveheight_alt", empty_float64())),
            peakwaveperiod_alt=np.array(ds.data_vars.get("peakwaveperiod_alt", empty_float64())),
            sea_water_salinity=np.array(ds.data_vars.get("sea_water_salinity", empty_float64())),
            sea_water_temperature=np.array(ds.data_vars.get("sea_water_temperature", empty_float64())),
            ID=int(id) if (id:=ds.attrs.get("id", None)) else None,
            ### TODO the rest of these fields aren't in sample data
            air_temperature=np.array(ds.data_vars.get("airtemp", empty_float64())),
            air_temperature_stddev=np.array(ds.data_vars.get("airtempstddev", empty_float64())),
            relhumidity=np.array(ds.data_vars.get("relhumidity", empty_float64())),
            relhumiditystddev=np.array(ds.data_vars.get("relhumiditystddev", empty_float64())),
            radiancemean=np.array(ds.data_vars.get("radiancemean", empty_float64())),
            radiancestd=np.array(ds.data_vars.get("radiancestd", empty_float64())),
            infraredtemp=np.array(ds.data_vars.get("infraredtemp", empty_float64())),
            infraredtempstd=np.array(ds.data_vars.get("infraredtempstd", empty_float64())),
            ambienttemp=np.array(ds.data_vars.get("ambienttemp", empty_float64())),
            ambienttempstd=np.array(ds.data_vars.get("ambienttempstd", empty_float64())),
            watertempstddev=np.array(ds.data_vars.get("watertempstddev", empty_float64())),
            salinitystddev=np.array(ds.data_vars.get("salinitystddev", empty_float64())),
            signature=Signature.from_dataset(ds),
            uplooking=Uplooking.from_dataset(ds),
            date=np.array(ds.data_vars.get("date", empty_float64())),
            sbdfile=np.array(ds.data_vars.get("sbdfile", empty_float64())),
            burstID=np.array(ds.data_vars.get("burstID", empty_float64())),
            battery=np.array(ds.data_vars.get("battery", empty_float64())),
            CTdepth=np.array(ds.data_vars.get("CTdepth", empty_float64())),
        )

@dataclass
class SWIFTArray:
    swift22: SWIFTData = field(default_factory=SWIFTData)
    swift23: SWIFTData = field(default_factory=SWIFTData)
    swift24: SWIFTData = field(default_factory=SWIFTData)
    swift25: SWIFTData = field(default_factory=SWIFTData)


@dataclass
class LSQWavePropParams:
    A: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    Etheta: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    f: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    theta: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    kx: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    ky: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    omega: npt.NDArray[np.float64] = field(default_factory=empty_float64)
    use_vel: bool = False


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
