"""Fast loader for the raw ATD (FIFA/EPTS) tracking export. See solution/CONTRACT.md.

Raw line format: ``<frame_id>:<slot>;<slot>;...:<ball>``; each group is ``x,y,speed`` (extra comma
fields are ignored), normalised 0-1 coordinates, ``NaN`` (bare or one per field) when missing.

Speed strategy: one Python pass splits each line into frame id / players / ball (cheap: 3 parts per
line). The player sections of all well-formed lines are joined into one buffer and parsed by the
pandas C CSV parser. Lines whose player section is irregular (odd field counts, empty groups, ...)
go through a slow but tolerant per-group parser. Results are cached as .npz keyed on the raw
file's size and mtime.

Data rule: error messages carry line numbers and counts only, never line contents.
"""

import io
import os
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

HOME, AWAY = 0, 1
FPS = 25
CACHE_VERSION = 2   # 2: round-trip float parsing + per-group field-count routing

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIELD_SEP = b":"
_GROUP_SEP = b";"
_VAL_SEP = b","


@dataclass
class Tracking:
    ball_xy: np.ndarray        # (N, 2) float32, metres, NaN where ball missing
    ball_speed: np.ndarray     # (N,)   float32, m/s as given by the file (NaN if missing)
    players_xy: np.ndarray     # (N, S, 2) float32, metres, NaN where slot absent
    players_speed: np.ndarray  # (N, S) float32, m/s
    team_of_slot: np.ndarray   # (S,) int8, HOME=0 / AWAY=1
    fps: int = FPS


# --------------------------------------------------------------------------------------- metadata

def team_of_slot_from_meta(meta_path):
    """(S,) int8: HOME for slots of the first <Team> in the metadata XML, AWAY otherwise."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    unknown = sum(1 for t in slots if t not in team_ids)
    if unknown:
        raise ValueError("{}: {} Player entries have a teamId matching no Team".format(
            meta_path, unknown))
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


# ----------------------------------------------------------------------------------------- parsing

def _to_float(token, line_no):
    token = token.strip()
    if not token:
        return np.nan
    try:
        return float(token)
    except ValueError:
        raise ValueError("ATD line {}: unparsable numeric field".format(line_no)) from None


def _parse_group(group, line_no):
    """First three fields (x, y, speed) of one comma group; missing ones -> NaN."""
    fields = group.split(_VAL_SEP)
    out = [np.nan, np.nan, np.nan]
    for i in range(min(3, len(fields))):
        out[i] = _to_float(fields[i], line_no)
    return out


def _split_lines(data, n_slots):
    """One pass over the file. Returns (frame_ids, player_sections, ball_sections, line_numbers)."""
    fids, players, balls, line_nos = [], [], [], []
    for line_no, line in enumerate(data.split(b"\n"), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(_FIELD_SEP)
        if len(parts) != 3:
            raise ValueError("ATD line {}: expected 3 ':'-separated parts, found {}".format(
                line_no, len(parts)))
        fid, pl, ball = parts
        try:
            fids.append(int(fid))
        except ValueError:
            raise ValueError("ATD line {}: unparsable frame id".format(line_no)) from None
        n_groups = pl.count(_GROUP_SEP) + 1
        if n_groups == n_slots + 1 and pl.endswith(_GROUP_SEP):   # tolerate a trailing ';'
            pl = pl[:-1]
            n_groups -= 1
        if n_groups != n_slots:
            raise ValueError(
                "ATD line {}: {} slot groups, but the metadata XML has {} Player entries".format(
                    line_no, n_groups, n_slots))
        players.append(pl)
        balls.append(ball)
        line_nos.append(line_no)
    return np.array(fids, dtype=np.int64), players, balls, line_nos


def _fields_per_group(players):
    """k = number of comma fields in a fully observed group (first one found), default 3."""
    for pl in players:
        for group in pl.split(_GROUP_SEP, 4):
            if _VAL_SEP in group:
                return group.count(_VAL_SEP) + 1
    return 3


def _expand_bare_nan(players, k):
    """Replace bare 'NaN' groups by k comma-separated NaNs, so every group has k fields."""
    full = _VAL_SEP.join([b"NaN"] * k)
    buf = b";" + b";\n;".join(players) + b";"
    for _ in range(2):  # adjacent bare groups overlap; two passes catch the ones skipped
        buf = buf.replace(b";NaN;", b";" + full + b";")
    return buf[1:-1].split(b";\n;")


def _parse_players_fast(players, n_slots, k):
    """C-parser path for player sections that all have exactly k fields per group."""
    if not players:
        return np.empty((0, n_slots, 3), dtype=np.float32)
    buf = b"\n".join(players).replace(_GROUP_SEP, _VAL_SEP)
    try:
        # round_trip: exact decimal->double conversion (same as Python float() in the slow
        # path), so the fast and slow paths agree bit-for-bit and pandas versions cannot drift.
        df = pd.read_csv(io.BytesIO(buf), header=None, dtype=np.float32, engine="c",
                         na_filter=True, skip_blank_lines=False, float_precision="round_trip")
    except (ValueError, pd.errors.ParserError):
        raise ValueError("ATD: player fields failed to parse as numbers") from None
    arr = df.to_numpy(dtype=np.float32)
    if arr.shape != (len(players), n_slots * k):
        raise ValueError("ATD: fast parser produced shape {}, expected {}".format(
            arr.shape, (len(players), n_slots * k)))
    arr = arr.reshape(len(players), n_slots, k)
    out = np.full((len(players), n_slots, 3), np.nan, dtype=np.float32)
    out[:, :, :min(k, 3)] = arr[:, :, :3]
    return out


def _parse_players_slow(players, line_nos, n_slots):
    out = np.full((len(players), n_slots, 3), np.nan, dtype=np.float32)
    for r, (pl, line_no) in enumerate(zip(players, line_nos)):
        for s, group in enumerate(pl.split(_GROUP_SEP)):
            out[r, s] = _parse_group(group, line_no)
    return out


def _parse_raw(raw_path, n_slots, force_slow=False):
    """(frame_ids, players (R,S,3) [x,y,speed normalised], ball (R,3)) for every line in the file."""
    with open(raw_path, "rb") as handle:
        data = handle.read()
    fids, players, balls, line_nos = _split_lines(data, n_slots)
    del data

    ball = np.array([_parse_group(b, n) for b, n in zip(balls, line_nos)],
                    dtype=np.float32).reshape(-1, 3)

    k = _fields_per_group(players)
    players = _expand_bare_nan(players, k)
    # A line takes the fast path only if EVERY group has exactly k fields. Checking the line's total
    # comma count is not enough: a short group and a long group on one line cancel out and would
    # silently shift values between slots.
    group = rb"[^,;]*(?:,[^,;]*){%d}" % (k - 1)
    exact = re.compile(group + rb"(?:;" + group + rb"){%d}" % (n_slots - 1))
    regular = np.array([exact.fullmatch(pl) is not None for pl in players], dtype=bool)
    if force_slow:
        regular[:] = False

    out = np.empty((len(players), n_slots, 3), dtype=np.float32)
    idx_fast = np.flatnonzero(regular)
    idx_slow = np.flatnonzero(~regular)
    out[idx_fast] = _parse_players_fast([players[i] for i in idx_fast], n_slots, k)
    if len(idx_slow):
        out[idx_slow] = _parse_players_slow([players[i] for i in idx_slow],
                                            [line_nos[i] for i in idx_slow], n_slots)
    return fids, out, ball


def _build(raw_path, team_of_slot, n_frames, pitch, force_slow=False):
    n_slots = len(team_of_slot)
    fids, pl, ball = _parse_raw(raw_path, n_slots, force_slow=force_slow)

    bad = (fids < 1) | (fids > n_frames)
    if bad.any():
        raise ValueError("ATD: {} lines have frame_id outside 1..{}".format(int(bad.sum()),
                                                                           n_frames))
    rows = fids - 1
    # keep the last occurrence of any duplicated frame id
    _, last = np.unique(rows[::-1], return_index=True)
    keep = len(rows) - 1 - last
    if len(keep) != len(rows):
        warnings.warn("ATD: {} duplicated frame ids; keeping the last line of each".format(
            len(rows) - len(keep)))
    rows, pl, ball = rows[keep], pl[keep], ball[keep]

    scale = np.array(pitch, dtype=np.float32)
    players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    players_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    players_xy[rows] = pl[:, :, :2] * scale
    players_speed[rows] = pl[:, :, 2]
    ball_xy[rows] = ball[:, :2] * scale
    ball_speed[rows] = ball[:, 2]
    return ball_xy, ball_speed, players_xy, players_speed


# ------------------------------------------------------------------------------------------- cache

def _cache_key(raw_path, n_frames, pitch, n_slots):
    st = os.stat(raw_path)
    return np.array([CACHE_VERSION, st.st_size, st.st_mtime_ns, n_frames, n_slots,
                     int(round(pitch[0] * 1e6)), int(round(pitch[1] * 1e6))], dtype=np.int64)


def _cache_file(cache_dir, raw_path):
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(_ROOT, cache_dir)
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    return os.path.join(cache_dir, stem + ".npz")


_ARRAYS = ("ball_xy", "ball_speed", "players_xy", "players_speed")


def _load_cache(path, key):
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as z:
            if not np.array_equal(z["key"], key):
                return None
            return tuple(np.array(z[name]) for name in _ARRAYS)
    except Exception:  # corrupt / partial cache: just rebuild
        return None


def _save_cache(path, key, arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        np.savez(handle, key=key, **dict(zip(_ARRAYS, arrays)))
    os.replace(tmp, path)


# -------------------------------------------------------------------------------------------- API

def load_tracking(meta_path=None, raw_path=None, n_frames=None, pitch=None,
                  cache_dir="solution/cache", _force_slow=False) -> Tracking:
    """Load the ATD into per-video-frame arrays (row = frame_id - 1), metres, NaN = missing.

    cache_dir=None disables caching. Relative cache_dir is resolved against the repo root.
    """
    if meta_path is None or raw_path is None:
        import games
        meta_default, raw_default = games.require(["atd_meta", "atd_raw"])
        meta_path = meta_path or meta_default
        raw_path = raw_path or raw_default
    if n_frames is None or pitch is None:
        import games
        n_frames = n_frames or games.source_frames("tactical")
        pitch = pitch or games.GAMES[games.GAME]["pitch"]
    n_frames = int(n_frames)
    pitch = (float(pitch[0]), float(pitch[1]))

    team_of_slot = team_of_slot_from_meta(meta_path)
    n_slots = len(team_of_slot)

    arrays = None
    if cache_dir is not None and not _force_slow:
        path = _cache_file(cache_dir, raw_path)
        key = _cache_key(raw_path, n_frames, pitch, n_slots)
        arrays = _load_cache(path, key)
    if arrays is None:
        arrays = _build(raw_path, team_of_slot, n_frames, pitch, force_slow=_force_slow)
        if cache_dir is not None and not _force_slow:
            _save_cache(path, key, arrays)

    ball_xy, ball_speed, players_xy, players_speed = arrays
    return Tracking(ball_xy=ball_xy, ball_speed=ball_speed, players_xy=players_xy,
                    players_speed=players_speed, team_of_slot=team_of_slot, fps=FPS)
