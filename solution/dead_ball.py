"""Dead-ball detector: per-frame score that the ball is NOT in play, plus a cleaned-up mask.

Design (no labels from this match are used anywhere; every number below is a football prior):

* Balanced errors. A false D inside a possession splits it in two; a missed D merges possessions
  or glues dead time onto one. spellF1 charges both about equally. So no single *soft* cue (ball
  missing, ball still, players slow) can start OR sustain a stoppage on its own; it takes two soft
  cues, or one near-certain cue (ball clearly off the pitch, ball lost/parked at a line, a Smart
  Tagging set-piece window, pre-kickoff). P1 (see the P1_* switches) raised recall within that rule.
* Sharp edges. Duration-qualified evidence (a long missing gap, a long stationary run) is stamped
  on the whole run rather than ramped in, and all filters are centred.
* Everything temporal runs inside one graded half (``games.half_video_blocks``); outside the
  halves the score is 0 and the mask is False.

Public API (see solution/CONTRACT.md):
    dead_ball_score(tracking, tags, blocks=None) -> (N,) float in [0, 1]
    dead_ball_mask(score, tracking, tags, blocks=None) -> (N,) bool
``blocks`` is an optional extension (list of (half, first, last) inclusive) used by the tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# --------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------
PITCH = (105.3, 68.0)          # metres; fallback when the tracking object carries no pitch size
OUT_MARGIN_M = 0.5             # ball centre this far beyond a line = whole ball over it, beyond tracking jitter
NEAR_LINE_M = 1.0              # throw-ins, corners and goal kicks out of the side are taken on/at the line
CORNER_R_M = 2.5               # corner kicks are placed inside the 1 m arc; 2.5 m allows for tracking error
CENTRE_R_M = 3.0               # kickoff: ball on the centre spot, allow tracking error
OWN_HALF_TOL_M = 1.0           # kickoff: players may stand on the halfway line itself
OWN_HALF_FRAC = 0.9            # kickoff: (almost) every tracked player of a team in its own half

# --------------------------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------------------------
VEL_HALF_WIN_S = 0.2           # centred +-0.2 s displacement for speed; smooths 25 Hz jitter, keeps kicks sharp
BALL_STILL_MPS = 0.6           # a placed ball only jitters; a dribbled ball moves > 1 m/s
NEAR_LINE_STILL_MIN_S = 1.0    # ball parked at a line for 1 s is a throw-in / corner being prepared
KICK_SPEED_MPS = 4.0           # restart kick/throw launches the ball well above walking pace
PLAYER_SMOOTH_S = 2.0          # centred smoothing of the collective speed; one sprint should not count
BASELINE_WIN_S = 300.0         # 5-minute window for the "normal" collective speed; longer than any stoppage
BASELINE_Q = 0.6               # upper-middle quantile: stoppages pull the median down, not the 60th pct
SLOW_RATIO_HI = 0.80           # collective speed below 80% of normal starts to look like a stoppage
SLOW_RATIO_LO = 0.55           # about half of normal = everyone walking (saturates here)
WALK_ABS_HI = 2.5              # m/s median player speed; above this nobody is walking (loose: units unverified)
WALK_ABS_LO = 1.5              # m/s; brisk walk, typical of players repositioning at a stoppage
KICK_PRE_GAP_S = 0.2           # restart onset: skip the last 0.2 s before the kick (centred speed already rising)
KICK_PRE_WIN_S = 0.4           # ... and require the 0.4 s before that to be ball missing or slow
KICK_PRE_MAX_MPS = 2.0         # a ball waiting for a restart moves slower than this

# --------------------------------------------------------------------------------------------
# Ball "not seen moving" runs (missing OR stationary, merged)
# --------------------------------------------------------------------------------------------
NM_BRIDGE_S = 0.5              # tracker blips (a stray detection, a lost frame) <= 0.5 s don't break a run
NM_SHORT_S = 3.0               # runs < 3 s are aerial balls / occlusions / a trap in live play
NM_LONG_S = 8.0                # an 8 s+ run with no ball movement is mostly a stoppage
LINE_LOST_MIN_S = 2.0          # ball last seen at a line then gone for 2 s = it went out

# --------------------------------------------------------------------------------------------
# Smart Tagging
# --------------------------------------------------------------------------------------------
SET_PIECE_LEAD_S = 3.5         # kickoff tags open ~3.5 s before the kick (starter/01_smart_tagging.ipynb,
                               # measured on the two Kick Off (Start) tags); see P1 part 2 for the others
RESTART_SEARCH_LO_S = 2.0      # refine the restart: search for the kick from 2 s after the tag opens ...
RESTART_SEARCH_HI_S = 6.0      # ... to 6 s after it (tag timing is automatic, so it is noisy)
TAG_LIVE_S = 2.0               # just after a restart the ball is live even if it is in the air / missing
GOAL_TO_KICKOFF_MAX_S = 180.0  # celebrations + VAR rarely exceed 3 min; beyond that do not bridge
KICKOFF_SEARCH_S = 600.0       # a half's kickoff lies within 10 min of the start of its graded block
KICKOFF_FORM_MIN_S = 2.0       # players line up for a kickoff for at least 2 s
KICKOFF_PRE_DEAD_FRAC = 0.8    # physical fallback: >= 80% of pre-kickoff frames must look dead

# --------------------------------------------------------------------------------------------
# Score: logistic over evidence (weights in logit units)
# --------------------------------------------------------------------------------------------
BIAS = -5.0                    # prior: the ball is in play ~60% of graded time; stay live by default
W_OUT = 6.0                    # ball clearly over a line: near-certain alone
W_LINE_LOST = 5.5              # lost at a line for 2 s+: enough alone (just over the enter threshold)
W_CORNER_STILL = 6.0           # ball parked in a corner arc: corner kick
W_LINE_STILL = 4.5             # ball parked on a line: needs one more cue (a winger can shield at the line)
W_NM_MIN = 2.0                 # 3 s without ball movement: weak alone, needs players clearly walking
W_NM_MAX = 4.5                 # 8 s+: just below STAY alone; moderate slowdown (>~0.3) makes it dead
W_SLOW = 3.0                   # players walking: confirms, never starts, a stoppage alone
W_KICKOFF_FORM = 6.0           # teams in own halves, ball on the spot: kickoff about to happen
W_TAG_DEAD = 6.0               # inside a set-piece tag before the restart
W_TAG_LIVE = -4.0              # just after a tagged restart
W_GOAL = 5.5                   # between a goal and the following kickoff

# --------------------------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------------------------
ENTER_THR = 0.6                # hysteresis: a dead run needs one frame above 0.6 ...
STAY_THR = 0.4                 # ... and lasts while the score stays above 0.4
MIN_LIVE_S = 1.0               # live gaps < 1 s between dead runs are tracker flicker, not play
MIN_DEAD_S = 2.0               # real stoppages last >= 2 s (public sample: 5-20 s); shorter = flicker
HARD_DEAD = 0.999              # score at/above this is forced dead after cleanup (pre-kickoff)

# --------------------------------------------------------------------------------------------
# P1: dead-ball recall package. Each part has its own switch so it can be measured on its own.
# With every switch False the module behaves exactly as the P0 version.
# --------------------------------------------------------------------------------------------
P1_OUT_GUARD = True            # part 6: a single out-of-pitch detection does not make a whole gap dead
P1_RESTART_LEAD = True         # part 2: fallback restart lead measured from the data, per set-piece kind
P1_SLIVERS = True              # part 1: short live runs between stoppages with no team control -> dead
P1_SOFT_RECAL = True           # part 3: mid-pitch stoppages (fouls, offsides) reachable from soft cues
P1_BACKFILL = True             # part 4: start each stoppage where play actually stopped

# part 6 -- out-carry guard
OUT_CARRY_MIN_FRAMES = 3       # 3 consecutive out frames (0.12 s) = a real ball over the line, not one jittery point
OUT_CARRY_LOOKBACK_S = 0.4     # ... or the ball was moving outwards over the last 0.4 s before it was lost
OUT_CARRY_MIN_STEP_M = 0.2     # outward by at least 0.2 m: a resting spare ball by the touchline does not qualify

# part 2 -- restart lead
MEASURED_LEAD_MIN_N = 5        # median of at least 5 detected kicks, else keep the documented 3.5 s
# The documented 3.5 s (starter/01_smart_tagging.ipynb) was measured on the two kickoff tags only, so
# kickoffs keep it; other set pieces use the median tag-to-kick delay this module detects itself.

# part 1 -- slivers
SLIVER_MAX_S = 3.0             # a live run this short between two stoppages is a tossed-back ball or a blip ...
SLIVER_MIN_CONTROL_S = 0.55    # ... unless one team shows >= 0.55 s of clear control in it (= possession.py's
                               #     MIN_INSERT_SECONDS: the same bar a possession must clear anywhere else)

# part 3 -- soft-cue recalibration. Mid-pitch stoppages (fouls, offsides) have no line or tag cue;
# only "ball not seen moving" + "players walking" can find them. Enter needs z > 5.41 (score 0.6);
# staying dead needs z > 4.59 (score 0.4).
W_NM_MIN_P1 = 2.5              # "not moving" weight at a 3 s run (P0: 2.0)
W_NM_MAX_P1 = 4.5              # ... at 8 s+ (= P0). Alone: 0.378 < 0.4, so it can neither START nor
                               #     SUSTAIN a stoppage -- a one-frame strong blip cannot turn a long
                               #     ball-tracking gap in live play into D
W_SLOW_P1 = 3.5                # players walking (P0: 3.0). Alone: 0.18, never a stoppage by itself
# 4 s + walking (slow 1): 2.9 + 3.5 = 6.4 enters; 8 s + moderate slowing (0.3): 4.5 + 1.05 = 5.55
# enters. P0 needed >= 3.8 s with everyone walking, or slow > 0.30 at 8 s.

# part 4 -- backward extension
BACKFILL_MAX_S = 3.0           # extend a stoppage back by at most 3 s
BACKFILL_BALL_MPS = 1.0        # ... over frames where the ball is not moving (missing or < 1 m/s)
BACKFILL_SLOW_MIN = 0.5        # ... and players are clearly walking (slow >= 0.5)


# ============================================================================================
# helpers
# ============================================================================================

def _blocks(n, blocks=None):
    if blocks is None:
        import games
        blocks = games.half_video_blocks("BAR-ATH")
    out = []
    for half, a, b in blocks:
        a, b = max(int(a), 0), min(int(b), n - 1)
        if b >= a:
            out.append((half, a, b))
    return out


def _runs(mask):
    """Start (inclusive) and end (exclusive) indices of True runs in a 1-D bool array."""
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _keep_runs(mask, min_len):
    """Drop True runs shorter than min_len frames."""
    out = np.zeros_like(mask, dtype=bool)
    s, e = _runs(mask)
    for a, b in zip(s[(e - s) >= min_len], e[(e - s) >= min_len]):
        out[a:b] = True
    return out


def _stamp_runs(mask, min_len):
    """Same as _keep_runs (named for readability where evidence is stamped on whole runs)."""
    return _keep_runs(mask, min_len)


def _ffill_nan(x):
    """Forward-fill then back-fill NaNs of a 1-D float array (for filters only)."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    y = x[idx]
    first = np.argmax(ok)
    y[:first] = x[first]
    return y


def _speed(xy, h):
    """Centred displacement speed (units/frame*2h) for (n,2) or (n,S,2) arrays; NaN at edges."""
    n = xy.shape[0]
    v = np.full(xy.shape[:-1], np.nan, dtype=np.float64)
    if n > 2 * h:
        d = xy[2 * h:] - xy[:-2 * h]
        v[h:n - h] = np.sqrt((d.astype(np.float64) ** 2).sum(axis=-1))
    return v


def _pitch(tracking):
    p = getattr(tracking, "pitch", None)
    if p is not None:
        return float(p[0]), float(p[1])
    try:
        import games
        p = games.GAMES["BAR-ATH"]["pitch"]
        return float(p[0]), float(p[1])
    except Exception:
        return PITCH


def _ramp(x, lo, hi):
    """0 at/below lo, 1 at/above hi (works for lo > hi as a falling ramp)."""
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _tag_rows(tags, a, b):
    """Tags overlapping [a, b] as a DataFrame with lower-case code/type strings."""
    if tags is None or len(tags) == 0:
        return None
    t = tags
    t = t[(t["end_frame"] >= a) & (t["start_frame"] <= b)]
    if len(t) == 0:
        return None
    t = t.copy()
    t["code"] = t["code"].astype(str).str.lower()
    t["type"] = t["type"].where(t["type"].notna(), "").astype(str).str.lower() \
        if "type" in t.columns else ""
    return t.sort_values("start_frame")


def _out_confirmed(i, out_run, present, dist_line, lookback):
    """Is the out-of-pitch detection at frame i (the last one before a gap) trustworthy?

    Yes if the ball was out for several consecutive frames, or if it was travelling outwards just
    before (a real ball crossing the line). A single stray point -- a high ball's parallax, a spare
    ball resting by the touchline -- is neither.
    """
    if out_run[i] >= OUT_CARRY_MIN_FRAMES:
        return True
    prev = np.flatnonzero(present[max(0, i - lookback):i])
    if not len(prev):
        return False
    j = max(0, i - lookback) + int(prev[0])        # earliest sighting in the look-back window
    return bool(dist_line[j] - dist_line[i] >= OUT_CARRY_MIN_STEP_M)


# ============================================================================================
# per-half features
# ============================================================================================

def _half_features(tr, a, b, fps, L, W):
    """Return dict of per-frame evidence arrays for frames a..b (inclusive)."""
    sl = slice(a, b + 1)
    n = b - a + 1
    h = max(1, int(round(VEL_HALF_WIN_S * fps)))
    dt2 = 2 * h / fps

    bxy = np.asarray(tr.ball_xy[sl], dtype=np.float64)
    present = np.isfinite(bxy).all(axis=1)
    bx, by = bxy[:, 0], bxy[:, 1]

    # --- ball location ---------------------------------------------------------------------
    with np.errstate(invalid="ignore"):
        dist_line = np.minimum.reduce([bx, L - bx, by, W - by])   # negative = outside
        out = present & (dist_line < -OUT_MARGIN_M)
        near_line = present & (dist_line <= NEAR_LINE_M)
        dcorner = np.minimum.reduce([np.hypot(bx - cx, by - cy)
                                     for cx in (0.0, L) for cy in (0.0, W)])
        corner = present & (dcorner <= CORNER_R_M)
        centre = present & (np.hypot(bx - L / 2, by - W / 2) <= CENTRE_R_M)

    # --- ball motion -----------------------------------------------------------------------
    bspeed = _speed(bxy, h) / dt2
    with np.errstate(invalid="ignore"):
        still_raw = present & (bspeed < BALL_STILL_MPS)
    line_still = _stamp_runs(still_raw & near_line, int(round(NEAR_LINE_STILL_MIN_S * fps)))
    corner_still = _stamp_runs(still_raw & corner, int(round(NEAR_LINE_STILL_MIN_S * fps)))

    # --- "ball not seen moving" runs (missing or still), stamped on the whole run ---------
    nm_raw = (~present) | still_raw
    nm = nm_raw.copy()
    bridge = int(round(NM_BRIDGE_S * fps))
    hs, he = _runs(~nm)
    for x, y in zip(hs, he):
        if (y - x) <= bridge and x > 0 and y < n:
            nm[x:y] = True
    nm_w = np.zeros(n)
    short, long_ = NM_SHORT_S * fps, NM_LONG_S * fps
    w_lo, w_hi = (W_NM_MIN_P1, W_NM_MAX_P1) if P1_SOFT_RECAL else (W_NM_MIN, W_NM_MAX)
    s, e = _runs(nm)
    for g0, g1 in zip(s, e):
        ln = g1 - g0
        if ln >= short:
            frac = min(1.0, (ln - short) / max(long_ - short, 1.0))
            nm_w[g0:g1] = w_lo + (w_hi - w_lo) * frac

    # --- missing gaps: where was the ball last seen? ----------------------------------------
    line_lost = np.zeros(n, dtype=bool)
    out_carry = np.zeros(n, dtype=bool)
    out_run = np.zeros(n, dtype=np.int64)          # length of the out run ending at each frame
    os_, oe_ = _runs(out)
    for x, y in zip(os_, oe_):
        out_run[x:y] = np.arange(1, y - x + 1)
    lookback = max(1, int(round(OUT_CARRY_LOOKBACK_S * fps)))
    s, e = _runs(~present)
    for g0, g1 in zip(s, e):
        ln = g1 - g0
        if g0 > 0:                                  # last seen position before the gap
            if out[g0 - 1]:
                # an unconfirmed out point is not trusted for "lost at a line" either -- falling
                # through to that rule would let the same stray point mark the gap dead
                if not P1_OUT_GUARD or _out_confirmed(g0 - 1, out_run, present, dist_line, lookback):
                    out_carry[g0:g1] = True
            elif near_line[g0 - 1] and ln >= LINE_LOST_MIN_S * fps:
                line_lost[g0:g1] = True

    # --- collective slowdown ---------------------------------------------------------------
    pxy = np.asarray(tr.players_xy[sl], dtype=np.float32)
    pspeed = _speed(pxy, h) / dt2                                     # (n, S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(pspeed, axis=1)
    if not np.isfinite(med).any():                                     # fall back to file speeds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.asarray(tr.players_speed[sl], dtype=np.float64), axis=1)
    have_players = np.isfinite(med)
    sm = uniform_filter1d(_ffill_nan(med), max(1, int(round(PLAYER_SMOOTH_S * fps))),
                          mode="nearest")
    # baseline at 1 Hz with a centred rolling upper-middle quantile, then back to frames
    step = max(1, int(fps))
    sec = sm[::step]
    win = max(3, int(round(BASELINE_WIN_S * fps / step)))
    base_sec = pd.Series(sec).rolling(win, center=True, min_periods=1).quantile(BASELINE_Q).to_numpy()
    base = np.repeat(base_sec, step)[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(base > 1e-6, sm / base, 1.0)
    slow = np.minimum(_ramp(ratio, SLOW_RATIO_HI, SLOW_RATIO_LO),
                      _ramp(sm, WALK_ABS_HI, WALK_ABS_LO))
    slow = np.where(have_players | (~present), slow, 0.0)
    if not have_players.any():
        slow = np.zeros(n)

    # --- kickoff formation -----------------------------------------------------------------
    team = np.asarray(tr.team_of_slot)
    px = pxy[:, :, 0].astype(np.float64)
    form = np.zeros(n, dtype=bool)
    if (team == 0).any() and (team == 1).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hx = px[:, team == 0]
            ax = px[:, team == 1]
            hmean = np.nanmean(hx, axis=1)
            amean = np.nanmean(ax, axis=1)
            home_left = hmean < amean
            mid = L / 2
            sgn = np.where(home_left, 1.0, -1.0)[:, None]
            # home "own half" is x <= mid (+tol) when home attacks right, etc.
            h_ok = (sgn * (hx - mid) <= OWN_HALF_TOL_M)
            a_ok = (sgn * (ax - mid) >= -OWN_HALF_TOL_M)
            h_cnt = np.isfinite(hx).sum(axis=1)
            a_cnt = np.isfinite(ax).sum(axis=1)
            h_frac = np.where(h_cnt > 0, (h_ok & np.isfinite(hx)).sum(axis=1) / np.maximum(h_cnt, 1), 0)
            a_frac = np.where(a_cnt > 0, (a_ok & np.isfinite(ax)).sum(axis=1) / np.maximum(a_cnt, 1), 0)
        split = (h_frac >= OWN_HALF_FRAC) & (a_frac >= OWN_HALF_FRAC) & (h_cnt >= 7) & (a_cnt >= 7)
        ball_ok = (centre & still_raw) | (~present & (slow > 0.5))   # just after kickoff teams are still split
        form = _stamp_runs(split & ball_ok, int(round(KICKOFF_FORM_MIN_S * fps)))

    return dict(present=present, out=out | out_carry, line_lost=line_lost,
                corner_still=corner_still, line_still=line_still, nm_w=nm_w, slow=slow, form=form, bspeed=bspeed)


def _detect_restart(start, end, bspeed_half, a, fps):
    """Restart frame (absolute) where the kick/throw is seen in the tracking, or None."""
    lo = start + int(round(RESTART_SEARCH_LO_S * fps))
    hi = min(end, start + int(round(RESTART_SEARCH_HI_S * fps)))
    lo = max(lo, a)
    seg = bspeed_half[max(lo - a, 0):max(hi - a + 1, 0)]
    gap, win = int(round(KICK_PRE_GAP_S * fps)), max(1, int(round(KICK_PRE_WIN_S * fps)))
    with np.errstate(invalid="ignore"):
        for c in np.flatnonzero(seg > KICK_SPEED_MPS):
            f = lo + int(c) - a                       # index into the half arrays
            pre = bspeed_half[max(f - gap - win, 0):max(f - gap, 0)]
            # onset = ball was waiting (missing or slow) just before; otherwise keep looking
            if len(pre) and not (pre >= KICK_PRE_MAX_MPS).any():
                return a + f
    return None


def _restart_frame(start, end, bspeed_half, a, fps, lead_s=None):
    """Estimated restart frame (absolute) for a set-piece tag opening at `start`.

    The detected kick if there is one, else `lead_s` seconds after the tag opens (default: the
    documented SET_PIECE_LEAD_S).
    """
    k = _detect_restart(start, end, bspeed_half, a, fps)
    if k is not None:
        return k
    lead = SET_PIECE_LEAD_S if lead_s is None else lead_s
    return min(end, start + int(round(lead * fps)))


def _is_kickoff(set_piece_type):
    return "kick off" in str(set_piece_type)


def _measured_restart_lead(blk, feats, ttags, fps):
    """Median tag-to-kick delay of the non-kickoff set pieces whose kick is detected, or the
    documented SET_PIECE_LEAD_S if fewer than MEASURED_LEAD_MIN_N kicks were found."""
    offsets = []
    for half, a, _ in blk:
        t = ttags[half]
        if t is None:
            continue
        for r in t[t["code"] == "set pieces"].itertuples(index=False):
            if _is_kickoff(r.type):
                continue
            k = _detect_restart(int(r.start_frame), int(r.end_frame), feats[half]["bspeed"], a, fps)
            if k is not None:
                offsets.append((k - int(r.start_frame)) / fps)
    if len(offsets) >= MEASURED_LEAD_MIN_N:
        return float(np.median(offsets))
    return SET_PIECE_LEAD_S


def _kickoff_from_tags_module(tags, tracking):
    try:
        from solution.tags import kickoff_frames
        k = kickoff_frames(tags, tracking)
        return {int(h): int(f) for h, f in dict(k).items() if f is not None and np.isfinite(f)}
    except Exception:
        return {}


def _kickoff_for_half(half, a, b, fps, feats, ttag, ko_mod):
    """Kickoff frame (absolute) of this half, or None. Never a hard-coded constant."""
    lim = a + int(round(KICKOFF_SEARCH_S * fps))
    k = ko_mod.get(half)
    if k is not None and k <= a:
        return None                                    # block starts at/after kickoff: no pre-kickoff dead
    if k is not None and k <= min(b, lim):
        return k
    if ttag is not None:
        sp = ttag[(ttag["code"] == "set pieces") & ttag["type"].str.contains("kick off")
                  & ttag["type"].str.contains("start") & (ttag["start_frame"] <= lim)]
        if len(sp):
            r = sp.iloc[0]
            k = _restart_frame(int(r.start_frame), int(r.end_frame), feats["bspeed"], a, fps)
            if a <= k <= b:
                return k
    # physical fallback: end of the first kickoff-formation run, if before it the ball looked dead
    s, e = _runs(feats["form"][: lim - a + 1])
    if len(s):
        k_rel = int(e[0])
        if k_rel > 0:
            pre = (~feats["present"][:k_rel]) | (feats["slow"][:k_rel] > 0.5) | feats["form"][:k_rel]
            if pre.mean() >= KICKOFF_PRE_DEAD_FRAC:
                return a + k_rel
    return None


# ============================================================================================
# public API
# ============================================================================================

CUE_NAMES = ("out", "line_lost", "still_at_line", "not_moving", "slow", "kickoff_form",
             "tag_dead", "goal", "pre_kickoff")


def dead_ball_score(tracking, tags, blocks=None, return_cues=False):
    """(N,) float in [0, 1]: probability-like score that the ball is NOT in play.

    return_cues=True returns (score, info) instead: info["cues"] maps each name in CUE_NAMES to a
    full-length bool array of where that cue was active, info["restarts"] lists
    (set_piece_type, restart_minus_tag_start_s, kick_found) per set-piece tag, and
    info["restart_lead_s"] is the fallback lead used for non-kickoff set pieces. For diagnostics.
    """
    n = int(np.asarray(tracking.ball_xy).shape[0])
    fps = int(getattr(tracking, "fps", 25) or 25)
    L, W = _pitch(tracking)
    score = np.zeros(n, dtype=np.float64)
    ko_mod = None
    blk = _blocks(n, blocks)

    feats = {half: _half_features(tracking, a, b, fps, L, W) for half, a, b in blk}
    ttags = {half: _tag_rows(tags, a, b) for half, a, b in blk}
    lead_other = (_measured_restart_lead(blk, feats, ttags, fps) if P1_RESTART_LEAD
                  else SET_PIECE_LEAD_S)
    cues = {name: np.zeros(n, dtype=bool) for name in CUE_NAMES} if return_cues else None
    restart_info = []

    for half, a, b in blk:
        f = feats[half]
        m = b - a + 1
        z = np.full(m, BIAS)
        z += W_OUT * f["out"] + W_LINE_LOST * f["line_lost"]
        z += np.maximum(W_CORNER_STILL * f["corner_still"], W_LINE_STILL * f["line_still"])
        z += f["nm_w"]
        z += (W_SLOW_P1 if P1_SOFT_RECAL else W_SLOW) * f["slow"] + W_KICKOFF_FORM * f["form"]

        ttag = ttags[half]
        tag_dead = np.zeros(m, dtype=bool)
        tag_live = np.zeros(m, dtype=bool)
        goal = np.zeros(m, dtype=bool)
        if ttag is not None:
            sp = ttag[ttag["code"] == "set pieces"]
            restarts = []
            for r in sp.itertuples(index=False):
                st, en = int(r.start_frame), int(r.end_frame)
                rs = _detect_restart(st, en, f["bspeed"], a, fps)
                found = rs is not None
                if not found:
                    lead = SET_PIECE_LEAD_S if _is_kickoff(r.type) else lead_other
                    rs = min(en, st + int(round(lead * fps)))
                restarts.append((rs, str(r.type)))
                restart_info.append((str(r.type), (rs - st) / fps, found))
                tag_dead[max(st - a, 0):max(rs - a, 0)] = True
                # "ball live just after the restart" only when the kick was actually seen: on a
                # guessed restart it would cut the stoppage short with nothing to back it.
                if found or not P1_RESTART_LEAD:
                    tag_live[max(rs - a, 0):max(rs + int(round(TAG_LIVE_S * fps)) - a, 0)] = True
            ko_restarts = sorted(rs for rs, ty in restarts if "kick off" in ty)
            for r in ttag[ttag["code"] == "goals"].itertuples(index=False):
                g = int(r.end_frame)
                nxt = [k for k in ko_restarts if g < k <= g + GOAL_TO_KICKOFF_MAX_S * fps]
                if nxt:
                    goal[max(g - a, 0):max(nxt[0] - a, 0)] = True
        z += W_TAG_DEAD * tag_dead + W_TAG_LIVE * (tag_live & ~tag_dead & ~f["out"]) + W_GOAL * goal

        s = 1.0 / (1.0 + np.exp(-z))

        if ko_mod is None:
            ko_mod = _kickoff_from_tags_module(tags, tracking) if tags is not None else {}
        k = _kickoff_for_half(half, a, b, fps, f, ttag, ko_mod)
        if k is not None and k > a:
            s[: k - a] = 1.0
        score[a:b + 1] = s

        if cues is not None:
            sl = slice(a, b + 1)
            cues["out"][sl] = f["out"]
            cues["line_lost"][sl] = f["line_lost"]
            cues["still_at_line"][sl] = f["corner_still"] | f["line_still"]
            cues["not_moving"][sl] = f["nm_w"] > 0
            cues["slow"][sl] = f["slow"] > 0.3
            cues["kickoff_form"][sl] = f["form"]
            cues["tag_dead"][sl] = tag_dead
            cues["goal"][sl] = goal
            if k is not None and k > a:
                cues["pre_kickoff"][a:k] = True
    if return_cues:
        return score, {"cues": cues, "restarts": restart_info, "restart_lead_s": lead_other}
    return score


def _hysteresis(s, hi, lo):
    """True runs of s > lo that contain at least one frame with s > hi."""
    low = s > lo
    st, en = _runs(low)
    out = np.zeros(len(s), dtype=bool)
    high = s > hi
    if len(st):
        csum = np.concatenate(([0], np.cumsum(high)))
        keep = (csum[en] - csum[st]) > 0
        for a, b in zip(st[keep], en[keep]):
            out[a:b] = True
    return out


def _backfill(d, f, fps):
    """Extend each dead run backwards over frames where play had visibly already stopped.

    The score only crosses the enter level once evidence has piled up, often seconds after the ball
    went out or the whistle went. Walk back from each run start while the ball is not moving
    (missing or under BACKFILL_BALL_MPS) and players are clearly walking, for at most
    BACKFILL_MAX_S, never into the previous dead run.
    """
    d = d.copy()
    with np.errstate(invalid="ignore"):
        moving = f["present"] & (f["bspeed"] >= BACKFILL_BALL_MPS)   # NaN speed is not "moving"
    quiet = ~moving & (f["slow"] >= BACKFILL_SLOW_MIN)
    cap = int(round(BACKFILL_MAX_S * fps))
    starts, ends = _runs(d)
    prev_end = 0
    for x0, x1 in zip(starts, ends):
        t = x0
        floor = max(prev_end, x0 - cap)
        while t > floor and quiet[t - 1]:
            t -= 1
        d[t:x0] = True
        prev_end = x1
    return d


def merge_slivers(dead, evidence, fps=25, blocks=None) -> np.ndarray:
    """(N,) bool: dead mask with short, uncontrolled live runs between two stoppages made dead.

    A live run shorter than SLIVER_MAX_S that sits between two dead runs becomes a predicted
    possession of its own. If no team shows SLIVER_MIN_CONTROL_S of clear control in it -- the same
    bar possession.py sets for any possession -- it is a tossed-back ball or a tracking blip inside
    one stoppage, and is merged into it. Within each graded half only.
    """
    dead = np.asarray(dead, dtype=bool).copy()
    if not P1_SLIVERS:
        return dead
    ev = np.nan_to_num(np.asarray(evidence, dtype=np.float64), nan=0.0)
    max_len = int(round(SLIVER_MAX_S * fps))
    min_control = SLIVER_MIN_CONTROL_S * fps
    for _, a, b in _blocks(len(dead), blocks):
        d = dead[a:b + 1]
        ls, le = _runs(~d)
        for x, y in zip(ls, le):
            if x > 0 and y < len(d) and (y - x) < max_len \
                    and abs(ev[a + x:a + y].sum()) < min_control:
                d[x:y] = True
        dead[a:b + 1] = d
    return dead


def dead_ball_mask(score, tracking=None, tags=None, blocks=None) -> np.ndarray:
    """(N,) bool: hysteresis + minimum durations, computed independently within each half."""
    score = np.nan_to_num(np.asarray(score, dtype=np.float64), nan=0.0)
    n = len(score)
    fps = int(getattr(tracking, "fps", 25) or 25) if tracking is not None else 25
    min_live = int(round(MIN_LIVE_S * fps))
    min_dead = int(round(MIN_DEAD_S * fps))
    mask = np.zeros(n, dtype=bool)
    L, W = _pitch(tracking) if tracking is not None else PITCH
    for _, a, b in _blocks(n, blocks):
        s = score[a:b + 1]
        d = _hysteresis(s, ENTER_THR, STAY_THR)
        # 0) start each stoppage where play actually stopped (needs the tracking)
        if P1_BACKFILL and tracking is not None:
            d = _backfill(d, _half_features(tracking, a, b, fps, L, W), fps)
        # 1) fill short live gaps that sit between two dead runs
        ls, le = _runs(~d)
        for x, y in zip(ls, le):
            if (y - x) < min_live and x > 0 and y < len(d):
                d[x:y] = True
        # 2) drop short dead runs
        d = _keep_runs(d, min_dead)
        d |= s >= HARD_DEAD
        mask[a:b + 1] = d
    return mask
