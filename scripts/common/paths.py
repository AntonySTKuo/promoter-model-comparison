"""Repository path helpers used by analysis scripts."""

import os



COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(COMMON_DIR)
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def repo_path(*parts):
    return os.path.join(REPO_ROOT, *parts)


def display_path(path):
    return os.path.relpath(path, REPO_ROOT)


# ── Fitted model checkpoints ─────────────────────────────────────────────────
# One directory per model variant under results/models/. Directory names carry
# a numeric prefix so they sort by model complexity; the tags used in code stay
# unprefixed. The figure labels GM* and GM** are the cGM and cGM_a1 tags -- the
# asterisks are kept out of directory names so shell globs stay predictable.

MODEL_CACHE_ROOT = repo_path("results", "models")

FIGURE_DIR = repo_path("results", "figures")

MODEL_CACHE_DIRNAMES = {
    "RM":     "0_RM",
    "AM":     "0_AM",
    "GM":     "1_GM",
    "cGM":    "2_cGM",
    "cGM_a1": "3_cGM_a1",
    }

MODEL_TAGS = tuple(MODEL_CACHE_DIRNAMES)


def model_cache_dir(tag):
    """Checkpoint directory for one model variant (GM* -> cGM, GM** -> cGM_a1)."""
    if tag not in MODEL_CACHE_DIRNAMES:
        raise ValueError(f"Unknown model tag: {tag!r} (expected one of {MODEL_TAGS})")
    return os.path.join(MODEL_CACHE_ROOT, MODEL_CACHE_DIRNAMES[tag])
