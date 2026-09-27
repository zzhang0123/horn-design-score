"""Frozen site, sky, signal and observing schedule for comparable scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
from types import MappingProxyType
from typing import Protocol as TypingProtocol

import healpy as hp
import numpy as np


class _BeamFrequencySource(TypingProtocol):
    freqs_mhz: np.ndarray


_DEFAULT_ID = "jbo-default-v2"
_DEFAULT_SKIES = frozenset({"cnn_pl_gleam", "gsm2008_gleam"})
_ANALYTIC_ID = "analytic-default-v1"
_ANALYTIC_SKIES = frozenset({"analytic_a", "analytic_b"})
_DEFAULT_SETTINGS = {
    "latitude_deg": 53.23625,
    "receiver_k": 100.0,
    "ground_k": 300.0,
    "loss_k": 300.0,
    "integration_s": 240.0,
    "bandwidth_hz": 1e6,
    "foreground_order": 5,
}


@dataclass(frozen=True, eq=False)
class Protocol:
    freqs_mhz: np.ndarray
    sky_maps_k: dict[str, np.ndarray]  # equatorial HEALPix RING, (frequency, pixel)
    signal_k: np.ndarray
    visits: np.ndarray  # visits per 1-degree LST bin
    latitude_deg: float = 53.23625
    receiver_k: float = 100.0
    ground_k: float = 300.0
    loss_k: float = 300.0
    integration_s: float = 240.0
    bandwidth_hz: float = 1e6
    foreground_order: int = 5
    protocol_id: str = "custom-v1"
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        f = np.array(self.freqs_mhz, dtype=float, copy=True)
        signal = np.array(self.signal_k, dtype=float, copy=True)
        visits = np.array(self.visits, dtype=float, copy=True)
        if f.ndim != 1 or f.size < 8 or not np.all(np.isfinite(f)) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
            raise ValueError("frequencies must be positive, increasing and have at least 8 channels")
        if signal.shape != f.shape or not np.all(np.isfinite(signal)):
            raise ValueError("signal_k must have one finite value per channel")
        if visits.shape != (360,) or not np.all(np.isfinite(visits)) or np.any(visits <= 0):
            raise ValueError("visits must contain 360 strictly positive LST-bin counts")
        physical = np.asarray((self.latitude_deg, self.receiver_k, self.ground_k,
                               self.loss_k, self.integration_s, self.bandwidth_hz), dtype=float)
        if (not np.all(np.isfinite(physical)) or not -90 <= self.latitude_deg <= 90
                or min(self.receiver_k, self.ground_k, self.loss_k) < 0):
            raise ValueError("invalid site or temperature")
        if (self.integration_s <= 0 or self.bandwidth_hz <= 0
                or not isinstance(self.foreground_order, (int, np.integer))
                or not 1 <= self.foreground_order <= f.size - 3):
            raise ValueError("invalid integration, bandwidth or foreground order")
        if not isinstance(self.protocol_id, str) or not self.protocol_id:
            raise ValueError("protocol_id must be a nonempty string")
        if len(self.sky_maps_k) < 1:
            raise ValueError("at least one foreground sky is required")
        maps: dict[str, np.ndarray] = {}
        nside: int | None = None
        for name, sky in self.sky_maps_k.items():
            if not name or not isinstance(name, str):
                raise ValueError("scenario names must be nonempty strings")
            a = np.array(sky, dtype=float, copy=True)
            if a.ndim != 2 or a.shape[0] != f.size or not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"sky {name}: expected nonnegative (frequency, pixel) Kelvin array")
            sky_nside = hp.npix2nside(a.shape[1])
            if nside is not None and sky_nside != nside:
                raise ValueError("all sky maps must have the same nside")
            nside = sky_nside
            maps[name] = a
        if self.protocol_id == "jbo-band-only-v1" and not np.array_equal(f, np.arange(55., 121.)):
            raise ValueError("jbo-band-only-v1 requires 55..120 MHz at 1 MHz spacing")
        if self.protocol_id == "jbo-default-v1" and not np.array_equal(f, np.arange(55., 121.)):
            raise ValueError("jbo-default-v1 requires 55..120 MHz at 1 MHz spacing")
        if (self.protocol_id in ("jbo-default-v1", _DEFAULT_ID)
                and (set(maps) != _DEFAULT_SKIES
                     or any(getattr(self, key) != value for key, value in _DEFAULT_SETTINGS.items()))):
            raise ValueError(f"{self.protocol_id} has inconsistent sky labels or observing settings")
        if (self.protocol_id == _ANALYTIC_ID
                and (set(maps) != _ANALYTIC_SKIES
                     or any(getattr(self, key) != value for key, value in _DEFAULT_SETTINGS.items()))):
            raise ValueError(f"{_ANALYTIC_ID} has inconsistent sky labels or observing settings")
        object.__setattr__(self, "freqs_mhz", f)
        object.__setattr__(self, "signal_k", signal)
        object.__setattr__(self, "visits", visits)
        object.__setattr__(self, "sky_maps_k", MappingProxyType(maps))
        digest = hashlib.sha256()
        for a in (f, signal, visits):
            digest.update(np.ascontiguousarray(a).tobytes())
        for name in sorted(maps):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(maps[name]).tobytes())
        settings = (self.latitude_deg, self.receiver_k, self.ground_k, self.loss_k,
                    self.integration_s, self.bandwidth_hz, self.foreground_order, self.protocol_id)
        digest.update(json.dumps(settings).encode())
        object.__setattr__(self, "_fingerprint", digest.hexdigest())
        for a in (f, signal, visits, *maps.values()):
            a.flags.writeable = False

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def sky_nside(self) -> int:
        return hp.npix2nside(next(iter(self.sky_maps_k.values())).shape[1])

    @classmethod
    def default(cls, beam: _BeamFrequencySource,
                sky_maps_k: dict[str, np.ndarray] | None = None,
                signal_k: np.ndarray | None = None,
                visits: np.ndarray | None = None) -> "Protocol":
        """Analytic design benchmark on the input beam's frequency grid.

        Without overrides, generate two reproducible analytic skies, a
        Gaussian absorption template, and 180 visits per LST bin. These are
        illustrative inputs, not the real Jodrell Bank 2025 reference data.
        Caller-supplied arrays must be sampled on beam.freqs_mhz.
        """
        try:
            freqs_mhz = beam.freqs_mhz
        except AttributeError as exc:
            raise TypeError("beam must provide freqs_mhz") from exc
        analytic_skies = sky_maps_k is None
        analytic_signal = signal_k is None
        if analytic_skies:
            try:
                nside = beam.nside
            except AttributeError as exc:
                raise TypeError("beam must provide nside to generate default foregrounds") from exc
            from .defaults import default_foreground_scenarios
            sky_maps_k = default_foreground_scenarios(
                freqs_mhz, nside, _DEFAULT_SETTINGS["bandwidth_hz"]
            )
        if analytic_signal:
            from .defaults import default_signal_k
            signal_k = default_signal_k(freqs_mhz, _DEFAULT_SETTINGS["bandwidth_hz"])
        if visits is None:
            visits = np.full(360, 180.)
        if analytic_skies and analytic_signal:
            protocol_id = _ANALYTIC_ID
        elif set(sky_maps_k) == _DEFAULT_SKIES and not analytic_signal:
            protocol_id = _DEFAULT_ID
        else:
            protocol_id = "custom-v1"
        return cls(freqs_mhz, sky_maps_k, signal_k, visits,
                   **_DEFAULT_SETTINGS, protocol_id=protocol_id)

    def save_npz(self, path: str | Path) -> None:
        """Save all arrays and observing settings needed to reproduce a score."""
        np.savez_compressed(
            path, freqs_mhz=self.freqs_mhz, signal_k=self.signal_k, visits=self.visits,
            latitude_deg=self.latitude_deg, receiver_k=self.receiver_k,
            ground_k=self.ground_k, loss_k=self.loss_k,
            integration_s=self.integration_s, bandwidth_hz=self.bandwidth_hz,
            foreground_order=self.foreground_order, protocol_id=self.protocol_id,
            **{f"sky_{name}": sky for name, sky in self.sky_maps_k.items()},
        )

    @classmethod
    def load_npz(cls, path: str | Path, *, require_reference_band: bool = False) -> "Protocol":
        """Load a protocol without pickle; older array-only NPZs use defaults."""
        with np.load(path, allow_pickle=False) as data:
            skies = {k[4:]: data[k] for k in data.files if k.startswith("sky_")}
            freqs = data["freqs_mhz"]
            if require_reference_band and not np.array_equal(freqs, np.arange(55., 121.)):
                raise ValueError("reference band requires 55..120 MHz at 1 MHz spacing")
            float_fields = ("latitude_deg", "receiver_k", "ground_k", "loss_k",
                            "integration_s", "bandwidth_hz")
            settings = {key: float(data[key]) for key in float_fields if key in data}
            if "foreground_order" in data:
                settings["foreground_order"] = int(data["foreground_order"])
            if "protocol_id" in data:
                settings["protocol_id"] = str(data["protocol_id"])
            return cls(freqs, skies, data["signal_k"], data["visits"], **settings)
