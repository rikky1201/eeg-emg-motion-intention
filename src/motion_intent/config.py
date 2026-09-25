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

# Session-ID number (in `ses-<id><speed>`, e.g. `ses-1431f`) -> the repeating
# movement cycle recorded in that session, as hysteresis-thresholding roles
# for `motion_intent.labeling.hysteresis_phases` (see there for the meaning
# of low/high/rising/falling). Speed suffix (f/n/s = fast/normal/slow) only
# changes pace, not the sequence of classes or the rep count (5, for all of
# these session numbers).
SESSION_PATTERNS = {
    "19": {"feature": "heel_activity", "low": "stand", "high": "walk"},
    "1431": {"feature": "hip_height", "low": "sit", "rising": "stand_up", "high": "stand", "falling": "sit_down"},
    # 3-step staircase: stand at the bottom, ascend, stand at the top while
    # turning around, descend, stand at the bottom again (next cycle).
    "176": {"feature": "hip_height", "low": "stand", "rising": "ascending", "high": "stand", "falling": "descending"},
}

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
#   RF = rectus femoris, TA = tibialis anterior,
#   BF = biceps femoris, GM = gastrocnemius (medial head); _L / _R = side
#
# The XDF-merged CSV's EMG_Stream channels are generic ("EMG_EMG0".."EMG_EMG7"),
# not muscle names, so the mapping to a physical sensor has to come from each
# session's own raw device CSV (data/raw/<subject>/*.csv), whose header lists
# the 8 muscles in wiring order. For subject haru this order is stable across
# the whole recording day for channels 0-5, but the last two (gastrocnemius
# medial L/R) come out swapped from ses-176n onward - the sensors were
# apparently re-plugged in the opposite order partway through the session.
EMG_CHANNEL_ORDER = ["RF_R", "RF_L", "TA_R", "TA_L", "BF_L", "BF_R", "GM_L", "GM_R"]
EMG_GM_SWAP_SESSIONS = {"176n", "176f", "19n"}  # ses-<id><speed> recorded after the gastroc rewiring
EMG_COLS = [f"emg_{name}" for name in EMG_CHANNEL_ORDER]


def emg_rename_for(session_id: str) -> dict[str, str]:
    """``{"EMG_EMG0": "emg_RF_R", ...}`` for one session, accounting for the
    gastrocnemius L/R swap in :data:`EMG_GM_SWAP_SESSIONS`.

    ``session_id`` is the full ``<number><speed>`` id, e.g. ``"176n"``
    (unlike :data:`SESSION_PATTERNS`, which only needs the number).
    """
    order = list(EMG_CHANNEL_ORDER)
    if session_id in EMG_GM_SWAP_SESSIONS:
        order[-2], order[-1] = order[-1], order[-2]
    return {f"EMG_EMG{i}": f"emg_{name}" for i, name in enumerate(order)}

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
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
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
# Environment features (task geometry, constant within a session)
# --------------------------------------------------------------------------- #
DEFAULT_CHAIR_HEIGHT_M = 0.60
DEFAULT_STAIR_HEIGHT_M = 0.30

# --------------------------------------------------------------------------- #
# Transition-offset analysis (evaluation)
# --------------------------------------------------------------------------- #
OFFSET_RANGE_MS = 400
OFFSET_STEP_MS = 10
