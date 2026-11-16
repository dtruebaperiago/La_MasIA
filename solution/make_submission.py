"""Generate the possession submission from the provided match data.

    python solution/make_submission.py --name <team_name>
    python solution/make_submission.py --name <team_name> --no-tags     # ablation: tracking only

Writes evaluation/results/<team_name>.csv (gitignored) and prints aggregate statistics only --
never per-frame rows (see CLAUDE.md: the match data must not leave this machine).

Pipeline (see solution/CONTRACT.md):
    ATD -> Tracking (metres, video-frame index)
    Smart Tagging -> tags DataFrame, team prior
    dead_ball_score / dead_ball_mask / merge_slivers   -> D
    team_evidence + segment (temporal model) -> H / A, X outside graded halves
All of it runs through solution.pipeline.run_pipeline, the same call diagnostics.py uses.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from solution.atd import load_tracking  # noqa: E402
from solution.pipeline import run_pipeline  # noqa: E402
from solution.tags import load_tags  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def write_submission(path, labels):
    """frame,label CSV with '\\n' line endings, byte-identical across runs and platforms."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = "".join("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.write(body)
    return path


def summarize(labels):
    """Aggregate statistics a coach would read -- counts, possessions, durations."""
    codes = pm.label_codes(np.asarray(labels, dtype=object))
    blocks = half_video_blocks(GAME)
    runs = [(first, last + 1) for _, first, last in blocks]
    graded = np.zeros(len(codes), dtype=bool)
    for start, stop in runs:
        graded[start:stop] = True
    filled = pm.fill_abstentions(codes, runs)
    stats = pm.possession_stats(filled, runs)
    print("graded-frame label counts:")
    for index, name in enumerate(pm.LABELS):
        n = int(np.sum(codes[graded] == index))
        print("  {}  {:>7}  {:5.1f}%".format(name, n, 100.0 * n / graded.sum()))
    dead_runs = [s for s in pm.segments(filled, runs) if s[2] == pm.D]
    print("possessions (X filled as the scorer does): {}  mean {:.1f} s  home share {:.3f}".format(
        stats["n_spells"], stats["mean_duration_s"], stats["home_share"]))
    print("dead-ball stretches: {}  mean {:.1f} s".format(
        len(dead_runs),
        np.mean([b - a for a, b, _ in dead_runs]) / GAMES[GAME]["fps"] if dead_runs else 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="team name; output is <name>.csv")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--no-tags", action="store_true", help="ignore the Smart Tagging prior")
    args = parser.parse_args()

    n_frames = source_frames("tactical", GAME)
    t0 = time.time()

    tracking = load_tracking()
    print("tracking loaded: {} frames, {} slots  ({:.1f} s)".format(
        tracking.ball_xy.shape[0], tracking.players_xy.shape[1], time.time() - t0))
    assert tracking.ball_xy.shape[0] == n_frames

    tags = load_tags()
    print("smart tagging: {} instances".format(len(tags)))

    labels = run_pipeline(tracking, tags, use_tags=not args.no_tags)["labels"]
    assert labels.shape == (n_frames,)

    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    print("wrote {}  ({:.1f} s total)\n".format(os.path.relpath(path, ROOT), time.time() - t0))
    summarize(labels)
    print("\nNext: python evaluation/validate_submission.py {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
