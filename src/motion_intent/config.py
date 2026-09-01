"""Central configuration: paths, channel groups, frequency bands, window sizes.

Every constant that used to be copy-pasted across notebook cells lives here so the
pipeline has a single source of truth.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# <repo>/src/motion_intent/config.py  ->  parents[2] == <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"          # *.xdf recordings (one folder per subject)
CSV_DIR = DATA_DIR / "csv"          # per-session CSV exported from XDF
FORMATTED_DIR = DATA_DIR / "formatted"  # per-session CSV with a `label` column
CLEAN_DIR = DATA_DIR / "clean"      # per-subject CSV, artefact-cleaned, modalities merged

# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
FS = 1000              # common working sampling rate [Hz] after resampling
FS_EMG_RAW = 2000      # EMG amplifier native rate [Hz]

# --------------------------------------------------------------------------- #
# Classes (7-class movement intention)
# --------------------------------------------------------------------------- #
CLASS_NAMES = [
    "stand",       # 0  standing (static)
    "sit",         # 1  sitting  (static)
    "descending",  # 2  stair descent
    "ascending",   # 3  stair ascent
    "walk",        # 4  level walking
    "stand_up",    # 5  sit-to-stand transition
    "sit_down",    # 6  stand-to-sit transition
]
N_CLASSES = len(CLASS_NAMES)

STATIC_CLASSES = [0, 1]
DYNAMIC_CLASSES = [2, 3, 4, 5, 6]

# --------------------------------------------------------------------------- #
# EEG channels
# --------------------------------------------------------------------------- #
# Full 32-channel cap (Emotiv / 10-20 layout) as exported by the amplifier.
EEG_CH_ALL = [
    "EEG_Cz", "EEG_Fz", "EEG_Fp1", "EEG_F7", "EEG_F3", "EEG_FC1", "EEG_C3",
    "EEG_FC5", "EEG_FT9", "EEG_T7", "EEG_CP5", "EEG_CP1", "EEG_P3", "EEG_P7",
    "EEG_PO9", "EEG_O1", "EEG_Pz", "EEG_Oz", "EEG_O2", "EEG_PO10", "EEG_P8",
    "EEG_P4", "EEG_CP2", "EEG_CP6", "EEG_T8", "EEG_FT10", "EEG_FC6", "EEG_C4",
    "EEG_FC2", "EEG_F4", "EEG_F8", "EEG_Fp2",
]

# Sensorimotor-cortex subset actually used for feature extraction.
EEG_CH_FEATURE = [
    "EEG_Cz", "EEG_FC1", "EEG_C3", "EEG_FC5", "EEG_CP1", "EEG_P3",
    "EEG_Pz", "EEG_P4", "EEG_CP2", "EEG_C4", "EEG_FC2",
]

# --------------------------------------------------------------------------- #
# EMG channels
# --------------------------------------------------------------------------- #
# The raw CSV headers are Japanese muscle names with a " (uV)" suffix. This map
# renames them to short romaji identifiers used everywhere downstream.
#   RF = rectus femoris, TA = tibialis anterior,
#   BF = biceps femoris, GM = gastrocnemius (medial head); _L / _R = side
EMG_RENAME = {
    "大腿直筋 左 (uV)": "emg_RF_L",
    "大腿直筋 右 (uV)": "emg_RF_R",
    "前脛骨筋 左 (uV)": "emg_TA_L",
    "前脛骨筋 右 (uV)": "emg_TA_R",
    "大腿二頭筋 右 (uV)": "emg_BF_R",
    "大腿二頭筋 左 (uV)": "emg_BF_L",
    "腓腹筋(内側) 右 (uV)": "emg_GM_R",
    "腓腹筋(内側) 左 (uV)": "emg_GM_L",
}
EMG_COLS = list(EMG_RENAME.values())

# --------------------------------------------------------------------------- #
# Motion-capture markers (OptiTrack biomechanics marker set)
# --------------------------------------------------------------------------- #
# Left/right: CAJ shoulder, FTC hip, FLE knee, FAL ankle, FCC heel, DP1 toe.
# The X and Z ground-plane axes are merged into a single "_XZ" column upstream
# (see preprocessing.merge_xz_all).
MOTION_MARKERS = [
    "RCAJ", "LCAJ", "RFTC", "LFTC", "RDP1", "LDP1",
    "RFCC", "LFCC", "RFLE", "LFLE", "RFAL", "LFAL",
]
MOTION_COLS = [f"Markers_{m}_XZ" for m in MOTION_MARKERS]

# --------------------------------------------------------------------------- #
# Frequency bands [Hz]
# --------------------------------------------------------------------------- #
EEG_BANDS = {
    "delta": (0.1, 4.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
EEG_BROADBAND = (8.0, 30.0)   # mu + beta, used for the CSP branch
EMG_BAND = (20.0, 450.0)

# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
WIN_SEC_EEG = 0.5
WIN_SEC_EMG = 0.2
WIN_SEC_MOTION = 0.2
STEP_SEC = 0.01

WIN_EEG = int(WIN_SEC_EEG * FS)
WIN_EMG = int(WIN_SEC_EMG * FS)
WIN_MOTION = int(WIN_SEC_MOTION * FS)
STEP = int(STEP_SEC * FS)

# --------------------------------------------------------------------------- #
# CSP
# --------------------------------------------------------------------------- #
CSP_N_COMPONENTS = 4

# --------------------------------------------------------------------------- #
# Environment features (task geometry, constant within a session)
# --------------------------------------------------------------------------- #
DEFAULT_CHAIR_HEIGHT_M = 0.60
DEFAULT_STAIR_HEIGHT_M = 0.30

# --------------------------------------------------------------------------- #
# Transition-offset analysis (evaluation)
# --------------------------------------------------------------------------- #
OFFSET_RANGE_MS = 400
OFFSET_STEP_MS = 10
