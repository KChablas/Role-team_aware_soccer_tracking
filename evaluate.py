"""
Evaluation pipeline — three stages:
  Stage 1: Standard MOT metrics (HOTA / CLEAR / Identity) via TrackEval
  Stage 2: Role classification accuracy (player / goalkeeper / referee)
  Stage 3: Team assignment accuracy (left / right)
  Stage 4: Per-sequence breakdown saved to CSV
"""

import argparse
import configparser
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def discover_sequences(config, pred_dir):
    explicit = config.get('sequences', [])
    if explicit:
        return sorted(explicit)
    return sorted(p.stem for p in Path(pred_dir).glob('SNMOT-*.txt'))


# ─────────────────────────────────────────────────────────────────────────────
# MOT file I/O
# ─────────────────────────────────────────────────────────────────────────────

def read_mot_file(path):
    """Returns list of rows as float arrays."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            rows.append([float(p) for p in parts])
    return rows


def filter_mot_rows(rows, min_frame, exclude_track_ids=None, exclude_cls_col=None,
                    exclude_cls_val=None):
    """Filter rows by frame, optional track id set, optional class column value."""
    out = []
    for r in rows:
        if int(r[0]) < min_frame:
            continue
        if exclude_track_ids and int(r[1]) in exclude_track_ids:
            continue
        if exclude_cls_col is not None and len(r) > exclude_cls_col:
            if int(r[exclude_cls_col]) == exclude_cls_val:
                continue
        out.append(r)
    return out


def rows_to_file(rows, path):
    with open(path, 'w') as f:
        for r in rows:
            f.write(','.join(str(v) if isinstance(v, int) else
                             (f"{v:.2f}" if isinstance(v, float) else str(v))
                             for v in r) + '\n')


# ─────────────────────────────────────────────────────────────────────────────
# gameinfo.ini parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_gameinfo(gameinfo_path):
    """
    Returns dict: track_id (int) -> {"role": str, "team": str|None}
    Roles: "player", "goalkeeper", "referee", "ball"
    Teams: "left", "right", None

    Expected format (section [Sequence], keys like trackletID_N):
        [Sequence]
        trackletID_1= player team left;10
        trackletID_22= goalkeepers team left;y
        trackletID_14= referee;main
    configparser lowercases keys, so we match "trackletid_N".
    """
    cfg = configparser.ConfigParser(strict=False)
    cfg.read(str(gameinfo_path))

    mapping = {}
    if not cfg.has_section('Sequence'):
        return mapping

    for key, value in cfg.items('Sequence'):
        # key is lowercased by configparser, e.g. "trackletid_1"
        if not key.startswith('trackletid_'):
            continue
        try:
            track_id = int(key.split('_', 1)[1])
        except (IndexError, ValueError):
            continue

        descriptor = value.split(';')[0].strip().lower()

        if 'goalkeeper' in descriptor or 'goalkeepers' in descriptor:
            role = 'goalkeeper'
        elif 'referee' in descriptor:
            role = 'referee'
        elif 'ball' in descriptor:
            role = 'ball'
        elif 'player' in descriptor:
            role = 'player'
        else:
            role = 'unknown'

        if 'team left' in descriptor:
            team = 'left'
        elif 'team right' in descriptor:
            team = 'right'
        else:
            team = None

        mapping[track_id] = {'role': role, 'team': team}

    return mapping


def ball_track_ids(gameinfo_map):
    return {tid for tid, info in gameinfo_map.items() if info['role'] == 'ball'}


# ─────────────────────────────────────────────────────────────────────────────
# IoU matching
# ─────────────────────────────────────────────────────────────────────────────

def xywh_to_xyxy(row):
    x, y, w, h = row[2], row[3], row[4], row[5]
    return x, y, x + w, y + h


def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def match_iou(gt_boxes, pred_boxes, threshold=0.5):
    """
    Hungarian matching on IoU matrix.
    Returns list of (gt_idx, pred_idx) matched pairs with IoU >= threshold.
    """
    if not gt_boxes or not pred_boxes:
        return []
    iou_mat = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            iou_mat[i, j] = compute_iou(g, p)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    return [(r, c) for r, c in zip(row_ind, col_ind) if iou_mat[r, c] >= threshold]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – TrackEval workspace setup and run
# ─────────────────────────────────────────────────────────────────────────────

def build_trackeval_workspace(workspace, sequences, pred_dir, gt_dir, min_frame, gameinfo_maps):
    # SKIP_SPLIT_FOL=True in run_trackeval means TrackEval uses GT_FOLDER and
    # TRACKERS_FOLDER directly, with no benchmark-split subdirectory inserted.
    gt_base = workspace / 'gt'
    trk_base = workspace / 'trackers' / 'soccer_tracker' / 'data'
    gt_base.mkdir(parents=True, exist_ok=True)
    trk_base.mkdir(parents=True, exist_ok=True)

    valid = []
    for seq in sequences:
        pred_file = Path(pred_dir) / f"{seq}.txt"
        gt_file = Path(gt_dir) / seq / 'gt' / 'gt.txt'
        seqinfo_src = Path(gt_dir) / seq / 'seqinfo.ini'

        if not pred_file.exists():
            print(f"  [skip] no prediction file for {seq}")
            continue
        if not gt_file.exists():
            print(f"  [skip] no gt file for {seq}")
            continue
        if not seqinfo_src.exists():
            print(f"  [skip] no seqinfo.ini for {seq}")
            continue

        ginfo = gameinfo_maps.get(seq, {})
        ball_ids = ball_track_ids(ginfo)

        gt_rows = read_mot_file(gt_file)
        gt_filtered = filter_mot_rows(gt_rows, min_frame, exclude_track_ids=ball_ids)

        pred_rows = read_mot_file(pred_file)
        # Exclude ball (class_id==0 is at column index 7, 0-based)
        pred_filtered = filter_mot_rows(pred_rows, min_frame,
                                        exclude_cls_col=7, exclude_cls_val=0)

        seq_gt_dir = gt_base / seq / 'gt'
        seq_gt_dir.mkdir(parents=True, exist_ok=True)
        rows_to_file(gt_filtered, seq_gt_dir / 'gt.txt')

        # Copy seqinfo.ini — TrackEval reads seqLength from it (mandatory)
        shutil.copy2(str(seqinfo_src), str(gt_base / seq / 'seqinfo.ini'))

        # Write tracker file with only 7 columns (frame,id,x,y,w,h,conf).
        # TrackEval reads column 7 (0-idx) as class and rejects values > 1
        # (pedestrian=1 is the only valid class). Truncating avoids this check.
        rows_to_file([r[:7] for r in pred_filtered], trk_base / f"{seq}.txt")
        valid.append(seq)

    # Seqmap: GT_FOLDER/seqmaps/BENCHMARK-SPLIT.txt = gt/seqmaps/train-train.txt
    seqmap_dir = workspace / 'gt' / 'seqmaps'
    seqmap_dir.mkdir(parents=True, exist_ok=True)
    with open(seqmap_dir / 'train-train.txt', 'w') as f:
        f.write('name\n')
        for seq in valid:
            f.write(f"{seq}\n")

    return valid


def run_trackeval(workspace):
    try:
        import trackeval
    except ImportError:
        print("  trackeval not found — skipping Stage 1. Install: pip install sn-trackeval")
        return None

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg['DISPLAY_LESS_PROGRESS'] = True

    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg['GT_FOLDER'] = str(workspace / 'gt')
    ds_cfg['TRACKERS_FOLDER'] = str(workspace / 'trackers')
    ds_cfg['BENCHMARK'] = 'train'       # seqmap filename = BENCHMARK-SPLIT = train-train.txt
    ds_cfg['SPLIT_TO_EVAL'] = 'train'
    ds_cfg['SKIP_SPLIT_FOL'] = True     # sequences live directly in GT_FOLDER/seq/, no split subdir
    ds_cfg['DO_PREPROC'] = False        # skip pedestrian-class validation (SoccerNet GT format)
    ds_cfg['TRACKER_SUB_FOLDER'] = 'data'
    ds_cfg['OUTPUT_FOLDER'] = str(workspace / 'results')
    ds_cfg['TRACKERS_TO_EVAL'] = ['soccer_tracker']

    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]

    evaluator = trackeval.Evaluator(eval_cfg)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(ds_cfg)]
    results, _ = evaluator.evaluate(dataset_list, metrics_list)
    return results


def extract_mot_summary(results):
    """Pull scalar metrics out of TrackEval results dict."""
    summary = {}
    try:
        tracker_results = results['MotChallenge2DBox']['soccer_tracker']
        combined = tracker_results.get('COMBINED_SEQ', {})

        hota = combined.get('pedestrian', {}).get('HOTA', {})
        clear = combined.get('pedestrian', {}).get('CLEAR', {})
        identity = combined.get('pedestrian', {}).get('Identity', {})

        summary['HOTA'] = float(np.mean(hota.get('HOTA', [0])))
        summary['DetA'] = float(np.mean(hota.get('DetA', [0])))
        summary['AssA'] = float(np.mean(hota.get('AssA', [0])))
        summary['MOTA'] = float(clear.get('MOTA', 0))
        summary['MOTP'] = float(clear.get('MOTP', 0))
        summary['IDF1'] = float(identity.get('IDF1', 0))
        summary['IDP'] = float(identity.get('IDP', 0))
        summary['IDR'] = float(identity.get('IDR', 0))
    except Exception:
        pass
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Stages 2 & 3 – Role and team evaluation per sequence
# ─────────────────────────────────────────────────────────────────────────────

ROLE_STR_TO_ID = {'player': 2, 'goalkeeper': 1, 'referee': 3}
ROLE_ID_TO_STR = {2: 'player', 1: 'goalkeeper', 3: 'referee'}


def group_by_frame(rows):
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[int(r[0])].append(r)
    return by_frame


def evaluate_sequence(pred_file, gt_file, gameinfo_map, min_frame):
    """
    Returns:
        role_pairs: list of (gt_role_id, pred_role_id)
        team_pairs_with_roles: list of (gt_team_str, pred_team_str, gt_role_id, pred_role_id)
    """
    gt_rows = read_mot_file(gt_file)
    pred_rows = read_mot_file(pred_file)

    ball_ids = ball_track_ids(gameinfo_map)

    gt_filtered = filter_mot_rows(gt_rows, min_frame, exclude_track_ids=ball_ids)
    pred_filtered = filter_mot_rows(pred_rows, min_frame,
                                    exclude_cls_col=7, exclude_cls_val=0)

    gt_by_frame = group_by_frame(gt_filtered)
    pred_by_frame = group_by_frame(pred_filtered)

    role_pairs = []
    team_pairs_with_roles = []

    all_frames = sorted(set(gt_by_frame) | set(pred_by_frame))

    for fid in all_frames:
        gt_frame = gt_by_frame.get(fid, [])
        pred_frame = pred_by_frame.get(fid, [])

        if not gt_frame or not pred_frame:
            continue

        gt_boxes_xyxy = [xywh_to_xyxy(r) for r in gt_frame]
        pred_boxes_xyxy = [xywh_to_xyxy(r) for r in pred_frame]

        matches = match_iou(gt_boxes_xyxy, pred_boxes_xyxy, threshold=0.5)

        for gi, pi in matches:
            gt_row = gt_frame[gi]
            pred_row = pred_frame[pi]

            gt_tid = int(gt_row[1])
            gt_info = gameinfo_map.get(gt_tid)
            if gt_info is None:
                continue

            gt_role = gt_info['role']
            if gt_role not in ROLE_STR_TO_ID:
                continue

            gt_role_id = ROLE_STR_TO_ID[gt_role]

            # pred class_id is column index 7 (0-based)
            pred_cls = int(pred_row[7]) if len(pred_row) > 7 else -1
            if pred_cls not in ROLE_ID_TO_STR:
                continue

            role_pairs.append((gt_role_id, pred_cls))

            # Team evaluation: only player/GK with known GT team and predicted team
            if gt_role in ('player', 'goalkeeper') and gt_info['team'] is not None:
                pred_team_label = int(pred_row[8]) if len(pred_row) > 8 else -1
                if pred_team_label in (1, 2, 3, 4):
                    # Map predicted label → side
                    pred_side = 'left' if pred_team_label in (1, 3) else 'right'
                    team_pairs_with_roles.append((gt_info['team'], pred_side, gt_role_id, pred_cls))

    return role_pairs, team_pairs_with_roles


def compute_role_metrics(role_pairs):
    """Compute per-class P/R/F1 and overall accuracy for roles 1, 2, 3."""
    if not role_pairs:
        return {}, 0.0

    roles = [1, 2, 3]
    tp = {r: 0 for r in roles}
    fp = {r: 0 for r in roles}
    fn = {r: 0 for r in roles}

    correct = 0
    for gt_r, pred_r in role_pairs:
        if gt_r == pred_r:
            correct += 1
            tp[gt_r] += 1
        else:
            if gt_r in fn:
                fn[gt_r] += 1
            if pred_r in fp:
                fp[pred_r] += 1

    accuracy = correct / len(role_pairs) if role_pairs else 0.0

    per_class = {}
    for r in roles:
        p = tp[r] / (tp[r] + fp[r]) if (tp[r] + fp[r]) > 0 else 0.0
        rec = tp[r] / (tp[r] + fn[r]) if (tp[r] + fn[r]) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        per_class[r] = {'P': p, 'R': rec, 'F1': f1}

    return per_class, accuracy


def compute_team_metrics(team_pairs):
    """Find best permutation and compute team accuracy."""
    if not team_pairs:
        return 0.0, 0.0, 0.0, 'unknown'

    agree_a = sum(1 for gt, pred in team_pairs if gt == pred)
    agree_b = sum(1 for gt, pred in team_pairs
                  if (gt == 'left' and pred == 'right') or
                     (gt == 'right' and pred == 'left'))

    if agree_a >= agree_b:
        perm = 'ours_left=GT_left'
        agreements = agree_a
        # Per-side under this perm
        left_correct = sum(1 for gt, pred in team_pairs if gt == 'left' and pred == 'left')
        left_total = sum(1 for gt, _ in team_pairs if gt == 'left')
        right_correct = sum(1 for gt, pred in team_pairs if gt == 'right' and pred == 'right')
        right_total = sum(1 for gt, _ in team_pairs if gt == 'right')
    else:
        perm = 'ours_left=GT_right'
        agreements = agree_b
        left_correct = sum(1 for gt, pred in team_pairs if gt == 'right' and pred == 'left')
        left_total = sum(1 for gt, _ in team_pairs if gt == 'right')
        right_correct = sum(1 for gt, pred in team_pairs if gt == 'left' and pred == 'right')
        right_total = sum(1 for gt, _ in team_pairs if gt == 'left')

    team_acc = agreements / len(team_pairs)
    acc_left = left_correct / left_total if left_total else 0.0
    acc_right = right_correct / right_total if right_total else 0.0

    return team_acc, acc_left, acc_right, perm


def compute_combined_accuracy(role_pairs, team_pairs):
    """% of player/GK detections where BOTH role AND team are correct."""
    # role_pairs and team_pairs are not directly linkable after aggregation,
    # so this is approximated as role_acc * team_acc for player/GK class only.
    gk_player_role_pairs = [(g, p) for g, p in role_pairs if g in (1, 2)]
    if not gk_player_role_pairs:
        return 0.0
    role_correct = sum(1 for g, p in gk_player_role_pairs if g == p)
    role_acc_sub = role_correct / len(gk_player_role_pairs)
    team_acc, _, _, _ = compute_team_metrics(team_pairs)
    return role_acc_sub * team_acc


def compute_combined_accuracy_exact(role_pairs, team_pairs_with_roles):
    """% of detections where BOTH role AND team are correct, using the optimal team permutation."""
    if not team_pairs_with_roles:
        return 0.0
    team_only = [(gt_t, pred_t) for gt_t, pred_t, _, _ in team_pairs_with_roles]
    _, _, _, perm = compute_team_metrics(team_only)
    correct = 0
    for gt_team, pred_team, gt_role, pred_role in team_pairs_with_roles:
        role_ok = (gt_role == pred_role)
        team_ok = (gt_team == pred_team) if perm == 'ours_left=GT_left' else (gt_team != pred_team)
        if role_ok and team_ok:
            correct += 1
    return correct / len(team_pairs_with_roles)


def weighted_mean(per_seq_results, key, weight_key):
    total_w = sum(r[weight_key] for r in per_seq_results)
    if total_w == 0:
        return 0.0
    return sum(r[key] * r[weight_key] for r in per_seq_results) / total_w


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Per-sequence CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_per_sequence_csv(per_seq_results, output_path):
    if not per_seq_results:
        return
    fieldnames = ['sequence', 'role_accuracy', 'gk_f1', 'player_f1', 'referee_f1',
                  'team_accuracy', 'combined_accuracy', 'n_role_pairs', 'n_team_pairs']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in per_seq_results:
            writer.writerow(row)
    print(f"Per-sequence CSV saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(mot_summary, per_class, role_acc, team_acc, acc_left, acc_right,
                  combined_acc, perm):
    lines = []
    lines.append('=' * 44)
    lines.append('         MOT Evaluation Results')
    lines.append('=' * 44)

    if mot_summary:
        lines.append(f"HOTA:  {mot_summary.get('HOTA', 0)*100:5.2f}   "
                     f"(DetA: {mot_summary.get('DetA', 0)*100:5.2f}, "
                     f"AssA: {mot_summary.get('AssA', 0)*100:5.2f})")
        lines.append(f"MOTA:  {mot_summary.get('MOTA', 0)*100:5.2f}   "
                     f"MOTP:  {mot_summary.get('MOTP', 0)*100:5.2f}")
        lines.append(f"IDF1:  {mot_summary.get('IDF1', 0)*100:5.2f}   "
                     f"IDP:   {mot_summary.get('IDP', 0)*100:5.2f}   "
                     f"IDR: {mot_summary.get('IDR', 0)*100:5.2f}")
    else:
        lines.append('(TrackEval not available — Stage 1 skipped)')

    lines.append('=' * 44)
    lines.append('      Role/Team Classification Results')
    lines.append('=' * 44)
    lines.append(f"Role Accuracy:          {role_acc*100:5.2f}%")

    for rid, name in [(2, 'Player '), (1, 'GK     '), (3, 'Referee')]:
        m = per_class.get(rid, {'P': 0, 'R': 0, 'F1': 0})
        lines.append(f"  {name} P/R/F1:       "
                     f"{m['P']*100:4.1f} / {m['R']*100:4.1f} / {m['F1']*100:4.1f}")

    lines.append(f"Team Accuracy:          {team_acc*100:5.2f}%")
    lines.append(f"  Side A accuracy:      {acc_left*100:5.2f}%")
    lines.append(f"  Side B accuracy:      {acc_right*100:5.2f}%")
    lines.append(f"Combined (Role+Team):   {combined_acc*100:5.2f}%")
    lines.append(f"Permutation used:       {perm}")
    lines.append('=' * 44)
    return '\n'.join(lines)


def build_confusion_matrix_text(role_pairs):
    roles = [1, 2, 3]
    names = {1: 'GK', 2: 'Player', 3: 'Referee'}
    mat = np.zeros((3, 3), dtype=int)
    for gt_r, pred_r in role_pairs:
        if gt_r in roles and pred_r in roles:
            mat[roles.index(gt_r), roles.index(pred_r)] += 1

    gt_pred_label = 'GT \\ Pred'
    header = f"{gt_pred_label:>10}" + ''.join(f"{names[r]:>10}" for r in roles)
    rows = [header]
    for i, r in enumerate(roles):
        row = f"{names[r]:>10}" + ''.join(f"{mat[i, j]:>10}" for j in range(3))
        rows.append(row)
    return '\n'.join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate tracking predictions')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    pred_dir = config['output_dir']
    gt_dir = config.get('gt_dir', config['dataset_dir'])
    min_frame = config.get('eval_start_frame', 101)
    out_dir = Path(config['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(config, pred_dir)
    if not sequences:
        print("No prediction files found in", pred_dir)
        return

    print(f"Evaluating {len(sequences)} sequence(s), frames >= {min_frame}\n")

    # Load gameinfo for all sequences
    gameinfo_maps = {}
    for seq in sequences:
        ginfo_path = Path(gt_dir) / seq / 'gameinfo.ini'
        if ginfo_path.exists():
            gameinfo_maps[seq] = parse_gameinfo(ginfo_path)
        else:
            gameinfo_maps[seq] = {}

    # ── Stage 1: TrackEval ───────────────────────────────────────────────────
    print("Stage 1: Building TrackEval workspace...")
    workspace = Path('eval_workspace')
    workspace.mkdir(exist_ok=True)

    valid_seqs = build_trackeval_workspace(
        workspace, sequences, pred_dir, gt_dir, min_frame, gameinfo_maps
    )
    print(f"  {len(valid_seqs)} sequences ready for TrackEval\n")

    print("Stage 1: Running TrackEval...")
    trackeval_results = run_trackeval(workspace)
    mot_summary = extract_mot_summary(trackeval_results) if trackeval_results else {}
    print()

    # ── Stages 2–4: Role / Team / Per-sequence ───────────────────────────────
    print("Stages 2-3: Role and team evaluation...")

    # Check if predictions have team info (ablation mode detection)
    has_team_info = False
    for seq in valid_seqs:
        pred_file = Path(pred_dir) / f"{seq}.txt"
        if pred_file.exists():
            rows = read_mot_file(pred_file)
            for r in rows:
                if len(r) > 8 and int(r[8]) >= 0:
                    has_team_info = True
                    break
        if has_team_info:
            break

    all_role_pairs = []
    per_seq_results = []
    per_class = {}
    role_acc = team_acc = acc_left = acc_right = combined_acc = 0.0
    perm = 'per-sequence'

    if not has_team_info:
        print("  No team/role labels in predictions (ablation mode) — skipping stages 2-3")
    else:
        for seq in valid_seqs:
            pred_file = Path(pred_dir) / f"{seq}.txt"
            gt_file = Path(gt_dir) / seq / 'gt' / 'gt.txt'

            if not pred_file.exists() or not gt_file.exists():
                continue

            role_pairs, team_pairs_with_roles = evaluate_sequence(
                pred_file, gt_file, gameinfo_maps[seq], min_frame
            )
            team_pairs = [(gt_t, pred_t) for gt_t, pred_t, _, _ in team_pairs_with_roles]

            all_role_pairs.extend(role_pairs)

            # Per-sequence metrics
            seq_per_class, seq_role_acc = compute_role_metrics(role_pairs)
            seq_team_acc, seq_acc_left, seq_acc_right, _ = compute_team_metrics(team_pairs)
            seq_combined = compute_combined_accuracy_exact(role_pairs, team_pairs_with_roles)

            per_seq_results.append({
                'sequence': seq,
                'role_accuracy': round(seq_role_acc, 4),
                'gk_f1': round(seq_per_class.get(1, {}).get('F1', 0), 4),
                'player_f1': round(seq_per_class.get(2, {}).get('F1', 0), 4),
                'referee_f1': round(seq_per_class.get(3, {}).get('F1', 0), 4),
                'team_accuracy': round(seq_team_acc, 4),
                'combined_accuracy': round(seq_combined, 4),
                'n_role_pairs': len(role_pairs),
                'n_team_pairs': len(team_pairs),
                'player_p': seq_per_class.get(2, {}).get('P', 0),
                'player_r': seq_per_class.get(2, {}).get('R', 0),
                'gk_p': seq_per_class.get(1, {}).get('P', 0),
                'gk_r': seq_per_class.get(1, {}).get('R', 0),
                'referee_p': seq_per_class.get(3, {}).get('P', 0),
                'referee_r': seq_per_class.get(3, {}).get('R', 0),
                'acc_side_a': seq_acc_left,
                'acc_side_b': seq_acc_right,
            })

        # ── Aggregate: weighted per-sequence means ────────────────────────────
        if per_seq_results:
            role_acc = weighted_mean(per_seq_results, 'role_accuracy', 'n_role_pairs')
            team_acc = weighted_mean(per_seq_results, 'team_accuracy', 'n_team_pairs')

            for role_id, prefix in [(2, 'player'), (1, 'gk'), (3, 'referee')]:
                p = weighted_mean(per_seq_results, f'{prefix}_p', 'n_role_pairs')
                r = weighted_mean(per_seq_results, f'{prefix}_r', 'n_role_pairs')
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                per_class[role_id] = {'P': p, 'R': r, 'F1': f1}

            acc_left = weighted_mean(per_seq_results, 'acc_side_a', 'n_team_pairs')
            acc_right = weighted_mean(per_seq_results, 'acc_side_b', 'n_team_pairs')
            combined_acc = weighted_mean(per_seq_results, 'combined_accuracy', 'n_role_pairs')

    # ── Stage 4: Save per-sequence CSV ──────────────────────────────────────
    save_per_sequence_csv(per_seq_results, out_dir / 'evaluation_per_sequence.csv')

    # ── Print and save summary ───────────────────────────────────────────────
    summary = build_summary(mot_summary, per_class, role_acc, team_acc, acc_left, acc_right,
                            combined_acc, perm)
    print('\n' + summary)

    summary_path = out_dir / 'evaluation_results.txt'
    with open(summary_path, 'w') as f:
        f.write(summary + '\n')
    print(f"\nSummary saved: {summary_path}")

    # ── Save confusion matrix ────────────────────────────────────────────────
    cm_text = build_confusion_matrix_text(all_role_pairs)
    cm_path = out_dir / 'confusion_matrix.txt'
    with open(cm_path, 'w') as f:
        f.write(cm_text + '\n')
    print(f"Confusion matrix saved: {cm_path}")


if __name__ == '__main__':
    main()
