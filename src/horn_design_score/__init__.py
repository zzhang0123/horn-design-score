"""Global 21 cm horn design scoring for a fixed drift-scan protocol."""

from .beam import BeamPattern
from .modes import BeamEfficiencyError, BeamModes, beam_modes_from_grid
from .protocol import Protocol
from .defaults import default_foreground_scenarios, default_signal_k
from .kernel import ZonalKernel
from .harmonic import HarmonicScorer
from .score import FittedSpectrum, ScoreResult, edges_beam_factor, score_beam
from .reference import reference_score, reference_forward, reference_modes_forward, reference_modes_score

__all__ = ["BeamPattern", "BeamEfficiencyError", "BeamModes", "beam_modes_from_grid", "Protocol",
           "default_foreground_scenarios", "default_signal_k", "ZonalKernel",
           "HarmonicScorer", "FittedSpectrum", "ScoreResult", "edges_beam_factor", "score_beam", "reference_score", "reference_forward",
           "reference_modes_forward", "reference_modes_score"]
__version__ = "0.6.1"
