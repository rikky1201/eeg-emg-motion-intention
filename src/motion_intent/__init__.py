"""motion_intent - movement-intention estimation from EEG / EMG / motion capture.

Modules
-------
config         constants: paths, channels, bands, window sizes
io_xdf         load LSL streams from an XDF recording into a merged DataFrame
labeling       derive per-sample movement labels and smooth them
preprocessing  band-pass filtering, EEG artefact cleaning, marker-axis merge
windowing      time-block train/val/test split and sliding-window extraction
features       Hjorth / CSP / RMS / kinematic / environment feature extractors
datasets       torch Dataset wrappers for the extracted feature matrices
models         fully-connected intention classifiers (EEG / non-EEG / fusion)
training       train / evaluate loops with early stopping
"""

from . import config  # noqa: F401

__all__ = ["config"]
__version__ = "0.1.0"
