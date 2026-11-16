"""The whole pipeline in one function, shared by make_submission.py and diagnostics.py.

Keeping a single call path means the diagnostics always describe exactly the file that gets
submitted -- a step added here (like sliver merging) cannot be missing from one of them.
"""

import numpy as np

from solution.dead_ball import dead_ball_mask, dead_ball_score, merge_slivers
from solution.possession import segment, team_evidence
from solution.tags import team_prior


def run_pipeline(tracking, tags, use_tags=True, blocks=None):
    """Tracking + tags -> dict with every intermediate and the final labels.

    keys: score (N,) dead-ball score; cue_info (see dead_ball_score return_cues); dead_raw (N,) mask
    before sliver merging; dead (N,) final mask; evidence (N,); prior (N,) or None;
    labels (N,) array of 'H'/'A'/'D'/'X'.
    """
    n = int(np.asarray(tracking.ball_xy).shape[0])
    fps = int(getattr(tracking, "fps", 25) or 25)
    score, cue_info = dead_ball_score(tracking, tags, blocks=blocks, return_cues=True)
    dead_raw = dead_ball_mask(score, tracking, tags, blocks=blocks)
    evidence = team_evidence(tracking)
    dead = merge_slivers(dead_raw, evidence, fps=fps, blocks=blocks)
    prior = team_prior(tags, n) if use_tags else None
    labels = np.asarray(segment(evidence, dead, prior=prior, blocks=blocks))
    return {"score": score, "cue_info": cue_info, "dead_raw": dead_raw, "dead": dead,
            "evidence": evidence, "prior": prior, "labels": labels}
