"""Team evidence from tracking and possession segmentation (contract: solution/CONTRACT.md).

Two stages:

  team_evidence(tracking)  per-frame signed evidence in [-1, 1] (+ home, - away), NaN when no ball.
  segment(evidence, dead)  2-state (H/A) Viterbi per live stretch with a switch cost, so the ball
                           changes team only on sustained *controlled* evidence for the opponent
                           (CHALLENGE.md), plus D on dead frames and X outside the graded halves.

Football rationale lives next to each constant. No frame constants of any match appear here.
"""

import numpy as np

HOME, AWAY = 0, 1

# --------------------------------------------------------------------------------------------
# team_evidence parameters
# --------------------------------------------------------------------------------------------

# A player "has" the ball when it is within roughly a stride of his feet. 1.5 m covers a close
# dribble plus tracking noise of the ball/player centroids.
CONTROL_RADIUS_M = 1.5

# Scale of the home-vs-away distance margin: tanh((d_away - d_home) / scale). One metre is the
# gap at which a challenger is clearly second best (a duel inside 1 m is genuinely contested and
# should give weak evidence).
MARGIN_SCALE_M = 1.0

# Proximity gate: if even the nearest player is far, the ball is loose or in flight and says
# nothing new about possession. Logistic, 0.5 at 2.5 m, ~0.1 at 3.6 m, ~0.9 at 1.4 m.
PROX_HALF_M = 2.5
PROX_WIDTH_M = 0.5

# Speed gate: a dribbled or received ball moves at running pace (< ~9 m/s even at a sprint);
# passes and shots travel at 10-30 m/s. Logistic, 0.5 at 11 m/s.
FLIGHT_SPEED_MS = 11.0
FLIGHT_WIDTH_MS = 1.5
# Ball speed is derived from ball_xy as displacement over a centred +/-2 frame baseline (0.16 s),
# not as a mean of single-frame steps, so centroid jitter does not inflate it (0.2 m of jitter is
# ~5 m/s per single step but ~1.2 m/s over the baseline). Falls back to +/-1 frame, then a
# one-sided step, then the file's ball_speed, then no gate.
SPEED_BASELINE_FRAMES = 2

# Dwell (control proxy): mean over the surrounding +/-5 frames (0.4 s window) of the margin on
# frames where the nearest player is within CONTROL_RADIUS_M. A reception or dribble keeps the
# ball near one team for several frames; a touch/deflection lasts one or two. Short on purpose
# so it does not delay onsets.
DWELL_HALF = 5
# Weight of the dwell term versus the instantaneous (margin x gates) term.
DWELL_WEIGHT = 0.5

# --------------------------------------------------------------------------------------------
# segment parameters
# --------------------------------------------------------------------------------------------

# Per-frame log-likelihood ratio H:A = 2 * EVIDENCE_GAIN * evidence. Only the ratio
# SWITCH_COST / EVIDENCE_GAIN matters, so the gain is fixed at 1 and the cost is expressed in
# "frames of full-strength evidence".
EVIDENCE_GAIN = 1.0

# An opposing possession inserted inside a spell (H-A-H) pays two switches and needs
# sum|evidence| > SWITCH_COST / EVIDENCE_GAIN; a one-way hand-over (H-A then dead/end) needs half
# of that. 0.55 s of clear control (~14 frames at 25 fps) is just above the 0.5 s limit for a
# deflection/aerial touch without control, and well below the ~0.8-1.3 s shortest real
# possessions, so it kills flicker without merging genuine short spells.
MIN_INSERT_SECONDS = 0.55

# Boundary snapping: Viterbi cannot place a switch inside a stretch of zero/NaN evidence (ball
# in flight or missing): all positions on that plateau score the same. "Loose ball stays with the
# last team", so each switch goes to the LATEST position whose score is within this tolerance
# (in frames of full-strength evidence) of the optimum, i.e. the reception. Well under one clear
# frame, so the snap never contradicts evidence the decoder weighed.
SNAP_TOLERANCE_FRAMES = 0.25

# Short ball-missing gaps (<= 0.2 s) between two frames that agree on the team are bridged with
# the weaker neighbour's evidence before decoding. Tracking dropouts during a carry should not
# shrink a possession's integrated evidence; the span is what separates a gappy ~1 s possession
# from a solid 0.5 s deflection.
BRIDGE_MAX_FRAMES = 5

# Tag prior. Used mainly as the initial-state bias of each live stretch (the restarting team):
# mean prior over the first 2 s of the stretch, worth ~3 frames of clear evidence, so it decides
# only when the early evidence is ambiguous.
PRIOR_INIT_SECONDS = 2.0
PRIOR_INIT_LLR = 6.0
# Per-frame prior gain, tiny: with zero evidence a prior of 1 needs SWITCH_COST/0.02 frames
# (~28 s) to insert a spell alone, and it can never outweigh real controlled evidence.
PRIOR_FRAME_GAIN = 0.02

# Tie-break for a live stretch with no evidence and no prior: keep the team that had the ball
# at the end of the previous live stretch of the same half (home at the head of a half).
FALLBACK_LLR = 1e-3


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------

def _logistic_down(x, half, width):
    """1 for x << half, 0 for x >> half."""
    z = np.clip((x - half) / width, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(z))


def _centred_nanmean(values, half):
    """Centred moving mean over +/-half frames ignoring NaN; NaN where the window is all NaN."""
    finite = np.isfinite(values)
    v = np.where(finite, values, 0.0).astype(np.float64)
    c_sum = np.concatenate(([0.0], np.cumsum(v)))
    c_cnt = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    n = len(values)
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + half + 1)
    cnt = c_cnt[hi] - c_cnt[lo]
    out = np.full(n, np.nan)
    ok = cnt > 0
    out[ok] = (c_sum[hi] - c_sum[lo])[ok] / cnt[ok]
    return out


def _ball_speed(tracking):
    """Ball speed in m/s from position differences, smoothed; file speed as fallback."""
    xy = np.asarray(tracking.ball_xy, dtype=np.float64)
    fps = float(tracking.fps)
    n = len(xy)
    speed = np.full(n, np.nan)
    for k in range(SPEED_BASELINE_FRAMES, 0, -1):
        if n <= 2 * k:
            continue
        d = np.hypot(xy[2 * k:, 0] - xy[:-2 * k, 0], xy[2 * k:, 1] - xy[:-2 * k, 1])
        cand = np.full(n, np.nan)
        cand[k:n - k] = d * fps / (2.0 * k)
        use = ~np.isfinite(speed) & np.isfinite(cand)
        speed[use] = cand[use]
    if n > 1:  # one-sided single step at the edges of visible stretches
        d = np.hypot(xy[1:, 0] - xy[:-1, 0], xy[1:, 1] - xy[:-1, 1]) * fps
        for cand in (np.concatenate((d, [np.nan])), np.concatenate(([np.nan], d))):
            use = ~np.isfinite(speed) & np.isfinite(cand)
            speed[use] = cand[use]
    speed[~np.isfinite(xy).all(axis=1)] = np.nan
    file_speed = getattr(tracking, "ball_speed", None)
    if file_speed is not None:
        file_speed = np.asarray(file_speed, dtype=np.float64)
        use = ~np.isfinite(speed) & np.isfinite(file_speed)
        speed[use] = file_speed[use]
    return speed


def _nearest_team_distances(tracking, chunk=20000):
    """(d_home, d_away) nearest distances in metres; inf where that team has nobody tracked."""
    ball = np.asarray(tracking.ball_xy, dtype=np.float32)
    players = np.asarray(tracking.players_xy, dtype=np.float32)
    team = np.asarray(tracking.team_of_slot)
    home_slots = team == HOME
    away_slots = team == AWAY
    n = len(ball)
    d_home = np.full(n, np.inf, dtype=np.float32)
    d_away = np.full(n, np.inf, dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        diff = players[a:b] - ball[a:b, None, :]
        dist = np.sqrt((diff * diff).sum(axis=2))
        dist = np.where(np.isfinite(dist), dist, np.inf)
        if home_slots.any():
            d_home[a:b] = dist[:, home_slots].min(axis=1)
        if away_slots.any():
            d_away[a:b] = dist[:, away_slots].min(axis=1)
    return d_home.astype(np.float64), d_away.astype(np.float64)


# --------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------

def team_evidence(tracking):
    """(N,) signed team evidence in [-1, 1] (+ home), NaN where there is no usable ball.

    evidence = (1 - DWELL_WEIGHT) * margin * w_prox * w_speed + DWELL_WEIGHT * dwell
      margin  = tanh((d_away - d_home) / MARGIN_SCALE_M), per team (slots are tracklets)
      w_prox  = logistic gate on the nearest-player distance (loose ball / in flight -> 0)
      w_speed = logistic gate on ball speed (pass or shot in flight -> 0)
      dwell   = centred mean over +/-DWELL_HALF frames of margin * [nearest within CONTROL_RADIUS_M],
                over frames with a ball (control proxy: sustained closeness of one team)
    """
    ball = np.asarray(tracking.ball_xy, dtype=np.float64)
    ball_ok = np.isfinite(ball).all(axis=1)

    d_home, d_away = _nearest_team_distances(tracking)
    d_min = np.minimum(d_home, d_away)
    usable = ball_ok & np.isfinite(d_min)

    with np.errstate(invalid="ignore"):
        gap = d_away - d_home  # +inf when only home is tracked, -inf when only away
        gap = np.where(np.isfinite(gap), gap, np.sign(np.nan_to_num(gap, nan=0.0)) * 1e3)
    margin = np.tanh(gap / MARGIN_SCALE_M)

    w_prox = _logistic_down(np.where(np.isfinite(d_min), d_min, 1e3), PROX_HALF_M, PROX_WIDTH_M)
    speed = _ball_speed(tracking)
    w_speed = np.where(np.isfinite(speed),
                       _logistic_down(np.nan_to_num(speed, nan=0.0),
                                      FLIGHT_SPEED_MS, FLIGHT_WIDTH_MS),
                       1.0)  # no speed information at all: do not gate

    instant = margin * w_prox * w_speed

    # Margin-weighted, so a contested duel (both within a metre) contributes little dwell.
    close = margin * (d_min <= CONTROL_RADIUS_M)
    close = np.where(usable, close, np.nan)
    dwell = _centred_nanmean(close, DWELL_HALF)

    ev = (1.0 - DWELL_WEIGHT) * instant + DWELL_WEIGHT * np.nan_to_num(dwell, nan=0.0)
    ev = np.clip(ev, -1.0, 1.0)
    ev[~usable] = np.nan
    return ev.astype(np.float32)


def _viterbi(llr, init_llr, switch_cost):
    """Most likely H/A path. llr[t] = log P(obs|H) - log P(obs|A). Returns +1 (H) / -1 (A)."""
    n = len(llr)
    half = [0.5 * float(v) for v in llr.tolist()]
    bp_h = bytearray(n)  # 1 = arrived in H by switching from A
    bp_a = bytearray(n)
    dh = 0.5 * init_llr + half[0]
    da = -0.5 * init_llr - half[0]
    for t in range(1, n):
        h = half[t]
        sw_h = da - switch_cost
        sw_a = dh - switch_cost
        if dh >= sw_h:
            nh = dh
        else:
            nh = sw_h
            bp_h[t] = 1
        if da >= sw_a:
            na = da
        else:
            na = sw_a
            bp_a[t] = 1
        dh = nh + h
        da = na - h
    out = np.empty(n, dtype=np.int8)
    state = 1 if dh >= da else -1
    for t in range(n - 1, -1, -1):
        out[t] = state
        if state == 1:
            if bp_h[t]:
                state = -1
        elif bp_a[t]:
            state = 1
    return out


def _snap_switches(path, llr, tol):
    """Place each switch at the latest position on the decoder's tie plateau (the reception).

    For a switch old->new inside (previous snapped switch, next switch), putting the boundary at
    b scores const - 2 * sum_{k<b} new*llr[k]; the optimum minimises that prefix sum. Among
    positions within `tol` of the minimum the latest is taken, so a pass in flight or a missing
    ball stays with the passer, but evidence the decoder weighed is never overridden.
    """
    path = path.copy()
    n = len(path)
    switches = (np.flatnonzero(path[1:] != path[:-1]) + 1).tolist()
    if not switches:
        return path
    bounds = switches + [n]
    spell_start = 0
    new_switches = []
    for k, t in enumerate(switches):
        new = float(path[t])
        lo, hi = spell_start + 1, bounds[k + 1] - 1  # candidates keep both spells non-empty
        # prefix[j] = sum of new*llr over [spell_start, lo + j), for boundary lo + j
        prefix = np.cumsum(new * llr[spell_start:hi])[lo - spell_start - 1:]
        t_new = lo + int(np.flatnonzero(prefix <= prefix.min() + tol)[-1])
        new_switches.append(t_new)
        spell_start = t_new
    edges = [0] + new_switches + [n]
    state = path[0]
    for a, b in zip(edges[:-1], edges[1:]):
        path[a:b] = state
        state = -state
    return path


def _merge_weak_tail(path, llr, switch_cost):
    """Drop a last spell whose own evidence would not pay for an inserted possession.

    A one-way switch costs only half an insertion, so a last touch before the ball goes dead
    (deflection out of play) would otherwise become a possession. The first spell of a stretch
    is left alone: the restart taker legitimately has short spells.
    """
    switches = np.flatnonzero(path[1:] != path[:-1]) + 1
    if not len(switches):
        return path
    t = int(switches[-1])
    own = float(path[t]) * 0.5 * float(llr[t:].sum())
    if own < switch_cost:
        path = path.copy()
        path[t:] = path[t - 1]
    return path


def _runs(mask):
    """[start, stop) runs of True in a boolean array."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _bridge_gaps(ev, dead):
    """Fill NaN runs of <= BRIDGE_MAX_FRAMES whose finite neighbours agree in sign."""
    ev = ev.copy()
    n = len(ev)
    for a, b in _runs(np.isnan(ev)):
        if b - a > BRIDGE_MAX_FRAMES or a == 0 or b >= n:
            continue
        left, right = ev[a - 1], ev[b]
        if left * right <= 0 or dead[a - 1:b + 1].any():
            continue
        ev[a:b] = np.sign(left) * min(abs(left), abs(right))
    return ev


def segment(evidence, dead_mask, prior=None, blocks=None, fps=25):
    """(N,) labels in {'H','A','D','X'}.

    Within each graded block (inclusive bounds) every live stretch between dead runs is decoded
    on its own with a 2-state Viterbi: per-frame LLR from evidence (0 where NaN, so a missing
    ball keeps the current team) plus a tiny prior term, a switch cost equal to
    MIN_INSERT_SECONDS of clear opposing control, and an initial bias from the early prior (the
    restarting team). Short NaN gaps inside a carry are bridged first; a weak last spell (last
    touch before the ball goes dead) is merged back; switches are placed at the latest position
    of the decoder's tie plateau (reception: a loose ball stays with the last team).
    Only H/A/D are emitted inside blocks (X would cost macroF1); X outside them.
    """
    ev = np.asarray(evidence, dtype=np.float64)
    n = len(ev)
    dead = np.asarray(dead_mask, dtype=bool)
    if len(dead) != n:
        raise ValueError("evidence and dead_mask lengths differ")
    pr = None if prior is None else np.nan_to_num(np.asarray(prior, dtype=np.float64), nan=0.0)
    if blocks is None:
        import games
        blocks = games.half_video_blocks("BAR-ATH")

    ev0 = np.nan_to_num(_bridge_gaps(ev, dead), nan=0.0)
    llr_all = 2.0 * EVIDENCE_GAIN * ev0
    if pr is not None:
        llr_all = llr_all + 2.0 * PRIOR_FRAME_GAIN * pr
    switch_cost = EVIDENCE_GAIN * MIN_INSERT_SECONDS * fps
    snap_tol = 2.0 * EVIDENCE_GAIN * SNAP_TOLERANCE_FRAMES
    n_init = max(1, int(round(PRIOR_INIT_SECONDS * fps)))

    labels = np.full(n, "X", dtype="<U1")
    for block in blocks:
        first, last = int(block[-2]), int(block[-1])
        first, stop = max(0, first), min(n, last + 1)
        if stop <= first:
            continue
        labels[first:stop][dead[first:stop]] = "D"
        fallback = 1  # home at the head of a half; reset per block
        for a, b in _runs(~dead[first:stop]):
            a += first
            b += first
            init = fallback * FALLBACK_LLR
            if pr is not None:
                init += PRIOR_INIT_LLR * float(np.mean(pr[a:min(b, a + n_init)]))
            llr = llr_all[a:b]
            path = _viterbi(llr, init, switch_cost)
            path = _merge_weak_tail(path, llr, switch_cost)
            path = _snap_switches(path, llr, snap_tol)
            labels[a:b] = np.where(path == 1, "H", "A")
            fallback = int(path[-1])
    return labels
