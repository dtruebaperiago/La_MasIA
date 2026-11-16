"""Smart Tagging loader and the priors derived from it (agent D, see solution/CONTRACT.md).

Smart Tagging is Metrica's *automatic* tactical annotation (SportsCode XML): intervals with a
code (phase of play or set piece) and labels (Team, Half, Type, ...). It is not ground truth, so
everything here is a *prior*: weak, documented, and meant to be combined with tracking evidence.

Time base (verified in starter/load_smart_tagging.py): <start>/<end> are seconds from the first
frame of the tactical video, so video_frame = round(seconds * fps).

Only the standard library XML parser is used (kloppy is not a dependency).
"""

from __future__ import annotations

import os
import re
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

COLUMNS = ["code", "team", "type", "half", "start_frame", "end_frame", "start_s", "end_s"]

SET_PIECE_CODE = "set pieces"
KICKOFF_START_TYPE = "kick off (start)"

# --------------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------------

def _norm(s):
    """Lower-case, strip, collapse internal whitespace. None stays None."""
    if s is None:
        return None
    s = " ".join(str(s).split()).lower()
    return s or None


def _team_code(value):
    """'Barcelona' -> 'H', 'Atletic Club' (sic) / 'Athletic Club' / 'Athletic Bilbao' -> 'A',
    'N/A' / empty / unknown -> None. Tolerant of case, accents-free spelling variants and spaces."""
    v = _norm(value)
    if v is None:
        return None
    letters = re.sub(r"[^a-z]", "", v)
    if letters in ("na", "none", "null", "unknown", ""):
        return None
    if "barcelona" in letters or letters.startswith("barca") or letters in ("fcb", "bar"):
        return "H"
    if "atletic" in letters or "athletic" in letters or "bilbao" in letters or letters == "ath":
        return "A"
    warnings.warn("tags: unknown Team value %r mapped to None" % (value,))
    return None


def _half_code(value):
    """'1st Half' -> 1, '2nd Half' -> 2, anything else -> None."""
    v = _norm(value)
    if v is None:
        return None
    if v.startswith("1") or "first" in v:
        return 1
    if v.startswith("2") or "second" in v:
        return 2
    return None


def _default_path():
    from games import require
    return require(["smart_tagging"])[0]


def load_tags(path=None, fps=25) -> pd.DataFrame:
    """Parse the SportsCode XML into one row per <instance>.

    Columns: code (lower str), team ('H'|'A'|None), type (lower str|None), half (1|2|None),
    start_frame (int), end_frame (int, *inclusive*), plus start_s / end_s (float seconds).
    Rows sorted by (start_frame, end_frame), file order as tie-break.
    Label groups are matched case-insensitively ('Team', 'team', 'TEAM' all work).
    """
    path = path or _default_path()
    rows = []
    for inst in ET.parse(path).iter("instance"):
        start_txt, end_txt = inst.findtext("start"), inst.findtext("end")
        if start_txt is None or end_txt is None:
            continue
        start_s, end_s = float(start_txt), float(end_txt)
        labels = {}
        for lab in inst.findall("label"):
            g = _norm(lab.findtext("group"))
            if g is not None:
                labels[g] = lab.findtext("text")  # last one wins (single-valued groups)
        start_f = int(round(start_s * fps))
        # Starter treats round(end*fps) as the first frame AFTER the phase -> inclusive end is -1.
        end_f = max(start_f, int(round(end_s * fps)) - 1)
        rows.append({
            "code": _norm(inst.findtext("code")),
            "team": _team_code(labels.get("team")),
            "type": _norm(labels.get("type")),
            "half": _half_code(labels.get("half")),
            "start_frame": start_f,
            "end_frame": end_f,
            "start_s": start_s,
            "end_s": end_s,
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["start_frame", "end_frame"], kind="mergesort").reset_index(drop=True)
    df["start_frame"] = df["start_frame"].astype(np.int64)
    df["end_frame"] = df["end_frame"].astype(np.int64)
    df["half"] = pd.Series([int(h) if h in (1, 2) else None for h in df["half"]],
                           index=df.index, dtype=object)
    df["team"] = df["team"].astype(object).where(df["team"].notna(), None)
    df["type"] = df["type"].astype(object).where(df["type"].notna(), None)
    return df


# --------------------------------------------------------------------------------------------
# Set pieces
# --------------------------------------------------------------------------------------------

def set_piece_windows(tags: pd.DataFrame) -> pd.DataFrame:
    """SET PIECES subset: start_frame, end_frame (inclusive), type, team, half."""
    m = tags["code"].fillna("").str.lower() == SET_PIECE_CODE
    out = tags.loc[m, ["start_frame", "end_frame", "type", "team", "half"]]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# Kickoffs
# --------------------------------------------------------------------------------------------

# Without tracking: set-piece tags open ~3.5 s before the actual restart (DATA.md / CLAUDE.md,
# a property of the tagger, not of this match), so the kickoff is estimated at start + 3.5 s,
# clipped into the tag window.
KICKOFF_TAG_LEAD_S = 3.5

# With tracking, refine within the tag window (+ a 1 s margin either side for tag jitter):
CENTRE_RADIUS_M = 3.0   # ball "on the centre spot": tolerance for tracker calibration error
STILL_TOL_M = 0.5       # ball at rest: within 0.5 m of its resting position (tracker jitter)
MOVE_TOL_M = 1.0        # ball has been played: > 1 m from rest (a kickoff pass travels metres)
MOVE_CONFIRM = (3, 5)   # departure confirmed if >=3 of the next 5 valid frames are > MOVE_TOL
MAX_GAP_S = 1.0         # a still run may bridge missing-ball gaps up to 1 s
WINDOW_MARGIN_S = 1.0


def _infer_half(frame):
    """Half for a frame from games.py block boundaries (allowed: they are published boundaries)."""
    try:
        from games import GAME, half_video_blocks
        blocks = half_video_blocks(GAME)
    except Exception:  # pragma: no cover
        return None
    best, best_d = None, None
    for half, lo, hi in blocks:
        d = 0 if lo <= frame <= hi else min(abs(frame - lo), abs(frame - hi))
        if best_d is None or d < best_d:
            best, best_d = half, d
    return best


def _refine_kickoff(tracking, start, end, pitch):
    """Return the first frame the ball leaves the centre spot inside [start, end], or None."""
    fps = int(getattr(tracking, "fps", 25) or 25)
    xy = np.asarray(tracking.ball_xy, dtype=float)
    n = len(xy)
    margin = int(round(WINDOW_MARGIN_S * fps))
    lo, hi = max(0, start - margin), min(n - 1, end + margin)
    if hi <= lo:
        return None
    w = xy[lo:hi + 1]
    valid = np.isfinite(w).all(axis=1)
    if valid.sum() < 3:
        return None
    centre = np.array([pitch[0] / 2.0, pitch[1] / 2.0])
    d_centre = np.full(len(w), np.inf)
    d_centre[valid] = np.hypot(*(w[valid] - centre).T)
    near = d_centre < CENTRE_RADIUS_M
    if not near.any():
        return None
    # Resting position: median of on-spot samples (the ball sits there for seconds, while the
    # kickoff pass spends only a few frames inside the radius, so the median is the spot).
    rest = np.median(w[near], axis=0)
    disp = np.full(len(w), np.nan)
    disp[valid] = np.hypot(*(w[valid] - rest).T)
    still = valid & (disp < STILL_TOL_M)
    moved = valid & (disp > MOVE_TOL_M)
    valid_idx = np.flatnonzero(valid)
    max_gap = int(round(MAX_GAP_S * fps))
    k_need, k_of = MOVE_CONFIRM

    last_still = None
    for pos, i in enumerate(valid_idx):
        if still[i]:
            last_still = i
            continue
        if last_still is None or not moved[i]:
            continue
        if i - last_still > max_gap + 1:
            last_still = None  # rest run too far back to explain this movement
            continue
        nxt = valid_idx[pos:pos + k_of]
        if moved[nxt].sum() >= min(k_need, len(nxt)):
            kick = lo + last_still + 1
            return int(min(max(kick, start), end))
    return None


def kickoff_frames(tags: pd.DataFrame, tracking=None) -> dict:
    """{half: estimated video frame of the kickoff that starts that half}.

    Source: SET PIECES tags with Type 'Kick Off (Start)'; earliest one per half. The tag window
    contains the restart but opens ~3.5 s before it.
    * tracking given (duck-typed: .ball_xy (N,2) metres, .fps, optional .pitch): the kickoff is
      the first frame the ball, resting near the centre spot, starts to move (displacement from
      its resting position > 1 m, confirmed over the next frames). Falls back if not found.
    * otherwise / fallback: tag start + KICKOFF_TAG_LEAD_S, clipped into the tag window.
    """
    fps = int(getattr(tracking, "fps", 25) or 25) if tracking is not None else 25
    pitch = getattr(tracking, "pitch", None) if tracking is not None else None
    if pitch is None:
        try:
            from games import GAME, GAMES
            pitch = GAMES[GAME]["pitch"]
        except Exception:  # pragma: no cover
            pitch = (105.3, 68.0)
    m = ((tags["code"].fillna("").str.lower() == SET_PIECE_CODE)
         & (tags["type"].fillna("").str.lower() == KICKOFF_START_TYPE))
    out = {}
    for _, r in tags.loc[m].sort_values("start_frame", kind="mergesort").iterrows():
        s, e = int(r["start_frame"]), int(r["end_frame"])
        half = r["half"] if r["half"] in (1, 2) else _infer_half(s)
        if half is None or half in out:
            continue
        est = None
        if tracking is not None:
            est = _refine_kickoff(tracking, s, e, pitch)
        if est is None:
            est = min(s + int(round(KICKOFF_TAG_LEAD_S * fps)), e)
        out[int(half)] = int(est)
    return out


# --------------------------------------------------------------------------------------------
# Team prior
# --------------------------------------------------------------------------------------------

# Signed weight per code, multiplied by the tagged team (+1 home, -1 away). Positive weight =
# the tagged team has the ball; negative = the tagged team is defending, so the OTHER team has
# it. Weights come from football reasoning about how strictly each phase implies possession over
# its whole interval (NOT tuned on any ground truth).
#
# Mirror-pair assumption: the published code counts pair up exactly (BUILD UP 22 / DEFENDING IN
# ATTACKING THIRD 22, PROGRESSION 66 / DEFENDING IN MIDDLE THIRD 66, CREATING CHANCES 30 /
# DEFENDING IN DEFENSIVE THIRD 30, ATTACKING / DEFENSIVE TRANSITION 24 / 24), which suggests the
# tagger emits each attacking phase together with the opponent's defending phase over the same
# interval. That is one piece of evidence, not two, so each pair is weighted to SUM to the
# intended confidence for that phase (~0.5-0.65), leaving headroom below the +-1 clip. Final-
# third detail tags (final third / box / players in box) stack on top of creating chances; only
# that full stack (a sustained attack with box entry) is allowed to approach saturation.
PRIOR_WEIGHTS = {
    # Build up (own half, controlled possession) + opponent pressing high. The press is where
    # regains happen, so the defending half of the pair gets less. Pair sum 0.60.
    "build up": 0.35,
    "defending in attacking third": -0.25,
    # Progression through the middle third + opponent's mid block: the opponent is settled out
    # of possession, the clearest possession phase. Pair sum 0.65.
    "progression": 0.35,
    "defending in middle third": -0.30,
    # Creating chances + opponent's low block: still the attacker's ball, but these end in shots,
    # crosses and clearances, so more of the interval is contested. Pair sum 0.60.
    "creating chances": 0.30,
    "defending in defensive third": -0.30,
    # Right after a regain / right after a loss: the onset can precede the turnover, and counter-
    # pressing often wins the ball straight back. Pair sum 0.50.
    "attacking transition": 0.30,
    "defensive transition": -0.20,
    # Final-third detail tags (overlap creating chances): the team maintains the ball near the
    # rival goal; box entries are often crosses (contested); players-in-box says who attacks,
    # not who touches the ball (defenders clear in the same window).
    "ball in final third": 0.25,
    "ball in the box": 0.20,
    "players in the box": 0.15,
    # Long ball: in the air most of the interval, second ball is a duel.
    "long ball": 0.20,
    # Goal interval runs into the celebration / dead ball; only its head is live possession.
    "goals": 0.10,
    # Restarts: tag covers mostly dead time before the restart; the taker's team gets the ball
    # when it goes live. Weak. N/A team (every throw-in) contributes nothing.
    "set pieces": 0.20,
}


def team_prior(tags: pd.DataFrame, n_frames: int) -> np.ndarray:
    """(n_frames,) float in [-1, 1]; positive = home in possession, 0 = no information.

    Each tag adds team_sign * PRIOR_WEIGHTS[code] over its (inclusive) frame span; overlapping
    tags sum; the result is clipped to [-1, 1]. Tags with team None or unknown code add nothing.
    """
    n = int(n_frames)
    acc = np.zeros(n + 1, dtype=np.float64)
    for code, team, s, e in zip(tags["code"], tags["team"], tags["start_frame"], tags["end_frame"]):
        w = PRIOR_WEIGHTS.get(_norm(code) or "", 0.0)
        if team not in ("H", "A") or w == 0.0:
            continue
        s, e = max(int(s), 0), min(int(e), n - 1)
        if e < s:
            continue
        v = w if team == "H" else -w
        acc[s] += v
        acc[e + 1] -= v
    prior = np.cumsum(acc[:n])
    prior[np.abs(prior) < 1e-12] = 0.0
    return np.clip(prior, -1.0, 1.0)


if __name__ == "__main__":
    # Aggregates only (data must never be printed row by row).
    t = load_tags()
    print("instances:", len(t))
    print(t.groupby("code").size().sort_values(ascending=False).to_string())
    print("team counts:", t["team"].value_counts(dropna=False).to_dict())
    print("kickoff halves found:", sorted(kickoff_frames(t)))
    from games import source_frames
    p = team_prior(t, source_frames("tactical"))
    print("prior: frac>0 %.3f  frac<0 %.3f  frac=0 %.3f  frac|p|==1 %.3f"
          % ((p > 0).mean(), (p < 0).mean(), (p == 0).mean(), (np.abs(p) >= 1).mean()))
    cover = np.zeros(len(p) + 1, dtype=np.int64)
    for s_, e_ in zip(t["start_frame"], t["end_frame"]):
        cover[max(s_, 0)] += 1
        cover[min(e_, len(p) - 1) + 1] -= 1
    cover = np.cumsum(cover[:-1])
    print("frames covered by >=1 tag %.3f, >=2 tags %.3f" % ((cover >= 1).mean(), (cover >= 2).mean()))
