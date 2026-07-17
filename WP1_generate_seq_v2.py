"""
Generate WP1 experiment sequences — VERSION 2 (unified array format).

Difference from v1:
    v1 wrote a separate 1-D sequence (-1/0/+1) per experiment type (f / p / a).
    v2 encodes everything in ONE (n_trials, 3) integer array:

        column 0 : frequency step   (-1 down / 0 stay / +1 up)   -- roving
        column 1 : azimuth  step    (-1 down / 0 stay / +1 up)   -- roving
        column 2 : shock            (0 none / 1 tap CS- / 2 cascade CS+)

    A trial only ever carries a deviant in ONE stimulus column at a time. The
    run-script iterates rows and maintains a per-modality roving index,
    resolving the step to actual Hz / degrees live.

Session types (--type)
----------------------
    f   frequency only            -> deviants in col 0; freq is CS+ (shocked)
    p   azimuth only              -> deviants in col 1; azimuth is CS+ (shocked)
    bf  both freq + azimuth       -> deviants in BOTH cols, but only FREQ is CS+
                                     (only freq deviants are ever shocked;
                                      azimuth deviants are an unreinforced control)
    bp  both freq + azimuth       -> both cols, only AZIMUTH is CS+ (shocked)

    Every type is a 2-block session: BASELINE (no shock) + CONDITIONING (shock).
    Think of each type as its own session / day. The abstract-pattern condition
    ('a') is unchanged — keep using v1's script for that.

    In the "both" modes the two deviant streams are interleaved while preserving
    each stream's internal order, so the +/-4 cumsum balance (and therefore the
    roving) is preserved for each dimension independently.

Usage
-----
Batch (N participants):
    python WP1_generate_seq_v2.py 20 --type bf
    python WP1_generate_seq_v2.py 20 --type f --seed 42

Single:
    python WP1_generate_seq_v2.py --participant 1 --type bp

Output per participant:
    WP1_sub001_bf_v2_seq.npy    -> the (n_trials, 3) int array
    WP1_sub001_bf_v2_seq.json   -> metadata (block table, roving lists, timing, CS+)
    WP1_sub001_bf_v2_seq.csv    -> human-readable long form (inspection only)
"""
import argparse
import json
import os
import random
import csv
from datetime import datetime

import numpy as np

# Reuse the validated v1 building blocks so the roving / balancing / shock
# logic stays identical and we don't fork two copies of it.
from WP1_generate_seq import (
    macke_feqlist,
    macke_azilist,
    create_roving_sequence,
    generate_reinforcement_schedule,
    get_cs_plus_assignment,
    sanitize_for_json,
    ITI,
    TONE_DURATION,
    ITI_WITHIN_PATTERN,
    MAX_CUMSUM,
    N_DEVIANTS_BASELINE,
    N_DEVIANTS_CONDITIONING,
    N_SHOCK,
    REINF_PROB_START,
    REINF_PROB_END,
    SHOCK_ONSET_IN_ITI,
    SOA,
)

# ============================================================================
# COLUMN LAYOUT  (single source of truth for the array shape)
# ============================================================================
COL_FREQ = 0      # frequency step  (-1/0/+1)
COL_AZI = 1       # azimuth  step  (-1/0/+1)
COL_SHOCK = 2     # shock amount   (0/1/2)
N_COLS = 3

# Which stimulus column each modality writes its roving steps into.
MODALITY_COL = {'f': COL_FREQ, 'p': COL_AZI}

# Base stimulus lists (same as v1). The centre index is the "resting" value.
FREQUENCY_LIST = macke_feqlist(700, MAX_CUMSUM, 0.1)
POSITION_LIST = macke_azilist(MAX_CUMSUM, 10)

# Session-type table: which modalities are present, and which one is CS+.
TYPE_CONFIG = {
    'f':  {'modalities': ['f'],       'cs_plus_modality': 'f'},
    'p':  {'modalities': ['p'],       'cs_plus_modality': 'p'},
    'bf': {'modalities': ['f', 'p'],  'cs_plus_modality': 'f'},
    'bp': {'modalities': ['f', 'p'],  'cs_plus_modality': 'p'},
}
TYPES = list(TYPE_CONFIG.keys())

# Every session is 2 blocks: baseline (no shock) + conditioning (shock).
BLOCK_PLAN_V2 = [
    {'label': 'BASELINE',     'n_deviants': N_DEVIANTS_BASELINE,     'use_reinforcement': False},
    {'label': 'CONDITIONING', 'n_deviants': N_DEVIANTS_CONDITIONING, 'use_reinforcement': True},
]


def split_into_trains(sequence):
    """Split a roving sequence into trains: each train = standards + 1 deviant."""
    trains, cur = [], []
    for s in sequence:
        cur.append(int(s))
        if s != 0:            # deviant terminates the train
            trains.append(cur)
            cur = []
    if cur:                   # trailing standards (no deviant) — keep them
        trains.append(cur)
    return trains


def merge_trains_preserving_order(streams):
    """
    Merge several ordered lists of trains into one interleaved order.

    Two guarantees:
      1. Each stream's internal order is preserved (we always take the next
         train off the front), so per-dimension cumsum balance survives.
      2. The features are spread EVENLY across the whole block (no clustering):
         at each step we emit whichever modality is furthest behind its target
         share (emitted / total). For equal deviant counts this yields near-
         strict F/A alternation; ties are broken randomly to avoid rhythmicity.

    `streams` is a dict: modality -> list of trains.
    Returns a list of (modality, train) tuples.
    """
    queues = {m: list(trains) for m, trains in streams.items()}
    totals = {m: len(q) for m, q in queues.items()}
    emitted = {m: 0 for m in queues}
    grand = sum(totals.values())

    out = []
    for _ in range(grand):
        cand = [m for m in queues if queues[m]]
        random.shuffle(cand)                       # random tie-break
        pick = min(cand, key=lambda m: emitted[m] / totals[m])
        out.append((pick, queues[pick].pop(0)))
        emitted[pick] += 1
    return out


def build_block(n_deviants, modalities, cs_plus_modality, cs_plus_value,
                use_reinforcement):
    """
    Build one block's (n, 3) array.

    Single modality  -> one roving stream in that column.
    Two modalities   -> two roving streams (n_deviants split evenly),
                        interleaved with order preserved; each deviant lands in
                        its own column. Only the CS+ modality is ever shocked.
    """
    if len(modalities) == 1:
        m = modalities[0]
        seq, _ = create_roving_sequence(n_deviants=n_deviants,
                                        max_cumsum=MAX_CUMSUM, soa=SOA)
        seq = np.asarray(seq, dtype=int)
        block = np.zeros((len(seq), N_COLS), dtype=int)
        block[:, MODALITY_COL[m]] = seq
    else:
        # Split the deviant budget evenly across the two modalities.
        per_mod = n_deviants // len(modalities)
        streams = {}
        for m in modalities:
            seq, _ = create_roving_sequence(n_deviants=per_mod,
                                            max_cumsum=MAX_CUMSUM, soa=SOA)
            streams[m] = split_into_trains(seq)
        merged = merge_trains_preserving_order(streams)

        rows = []
        for m, train in merged:
            col = MODALITY_COL[m]
            for s in train:
                row = [0, 0, 0]
                if s != 0:
                    row[col] = s
                rows.append(row)
        block = np.asarray(rows, dtype=int)

    # Reinforcement: only the CS+ modality's deviants can be shocked.
    if use_reinforcement:
        cs_col = MODALITY_COL[cs_plus_modality]
        num_shock, _ = generate_reinforcement_schedule(
            block[:, cs_col].tolist(), cs_plus_value,
            n_shock=N_SHOCK, prob_start=REINF_PROB_START, prob_end=REINF_PROB_END,
        )
        block[:, COL_SHOCK] = np.asarray(num_shock, dtype=int)

    return block


# ============================================================================
# BUILD ONE PARTICIPANT
# ============================================================================
def generate_one_v2(participant_id, exp_type, seed, out_dir='.'):
    random.seed(seed)
    np.random.seed(seed)

    tcfg = TYPE_CONFIG[exp_type]
    modalities = tcfg['modalities']
    cs_plus_modality = tcfg['cs_plus_modality']

    # CS+ direction per modality (same counterbalancing table as v1).
    assignment = get_cs_plus_assignment(participant_id)  # {'f':±1,'p':±1,'a':±1}
    cs_plus_value = assignment[cs_plus_modality]

    block_rows = []          # list of (n_i, 3) arrays, concatenated at the end
    block_table = []         # metadata describing each block's row span
    cursor = 0               # running row index into the final array

    for cfg in BLOCK_PLAN_V2:
        label = f"{exp_type.upper()}_{cfg['label']}"
        print(f"\n--- {label}  (modalities={modalities}, "
              f"CS+={cs_plus_modality}{cs_plus_value:+d}) ---")

        block = build_block(
            n_deviants=cfg['n_deviants'],
            modalities=modalities,
            cs_plus_modality=cs_plus_modality,
            cs_plus_value=cs_plus_value,
            use_reinforcement=cfg['use_reinforcement'],
        )
        n = block.shape[0]
        block_rows.append(block)

        block_table.append({
            'label': label,
            'modalities': modalities,
            'cs_plus_modality': cs_plus_modality,
            'cs_plus_column': MODALITY_COL[cs_plus_modality],
            'cs_plus_value': int(cs_plus_value),
            'cs_minus_value': int(-cs_plus_value),
            'use_reinforcement': cfg['use_reinforcement'],
            'row_start': cursor,              # inclusive
            'row_end': cursor + n,            # exclusive
            'n_trials': int(n),
            'n_freq_deviants': int(np.count_nonzero(block[:, COL_FREQ])),
            'n_azi_deviants': int(np.count_nonzero(block[:, COL_AZI])),
            'n_shock_trials': int(np.count_nonzero(block[:, COL_SHOCK])),
        })
        cursor += n

    seq_array = np.vstack(block_rows).astype(int)  # (total_trials, 3)

    tag = f"sub{participant_id:03d}_{exp_type}_v2"

    # --- Save the numpy array (the primary v2 artefact) ---------------------
    npy_path = os.path.join(out_dir, f"WP1_{tag}_seq.npy")
    np.save(npy_path, seq_array)

    # --- Save metadata ------------------------------------------------------
    meta = sanitize_for_json({
        'version': 2,
        'participant_id': participant_id,
        'experiment_type': exp_type,
        'modalities': modalities,
        'cs_plus_modality': cs_plus_modality,
        'cs_plus_value': int(cs_plus_value),
        'random_seed': seed,
        'generated_at': datetime.now().isoformat(),
        # array description
        'array_shape': list(seq_array.shape),
        'columns': {'freq_step': COL_FREQ, 'azi_step': COL_AZI, 'shock': COL_SHOCK},
        'shock_codes': {'none': 0, 'cs_minus_tap': 1, 'cs_plus_cascade': 2},
        # stimulus resolution tables (roving walks these with index clamping)
        'frequency_list': FREQUENCY_LIST,
        'position_list': POSITION_LIST,
        'freq_center_index': len(FREQUENCY_LIST) // 2,
        'azi_center_index': len(POSITION_LIST) // 2,
        # timing
        'ITI': ITI,
        'tone_duration': TONE_DURATION,
        'iti_within_pattern': ITI_WITHIN_PATTERN,
        'SOA': SOA,
        'max_cumsum': MAX_CUMSUM,
        'shock_onset_in_iti': SHOCK_ONSET_IN_ITI,
        # per-modality CS+ assignment
        'cs_plus_assignment': {k: int(v) for k, v in assignment.items()},
        # block layout
        'blocks': block_table,
    })
    json_path = os.path.join(out_dir, f"WP1_{tag}_seq.json")
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # --- Save a readable long-form CSV (inspection only) --------------------
    csv_path = os.path.join(out_dir, f"WP1_{tag}_seq.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row', 'block_label', 'freq_step', 'azi_step', 'shock'])
        for blk in block_table:
            for r in range(blk['row_start'], blk['row_end']):
                w.writerow([r, blk['label'],
                            seq_array[r, COL_FREQ], seq_array[r, COL_AZI], seq_array[r, COL_SHOCK]])

    total_trials = seq_array.shape[0]
    total_shocks = int(seq_array[:, COL_SHOCK].sum())
    print(f"\n  -> {npy_path}")
    print(f"  -> {json_path}")
    print(f"  -> {csv_path}")
    print(f"     {total_trials} trials, {len(block_table)} blocks, "
          f"{total_shocks} total shock pulses")

    return {
        'participant_id': participant_id,
        'experiment_type': exp_type,
        'total_trials': total_trials,
        'n_blocks': len(block_table),
        'total_shocks': total_shocks,
        'npy': npy_path,
        'json': json_path,
        'csv': csv_path,
        'seed': seed,
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate WP1 v2 unified-array sequences.')
    parser.add_argument('n_participants', type=int, nargs='?', default=None,
                        help='Number of participants (batch mode)')
    parser.add_argument('--participant', type=int, default=None,
                        help='Single participant ID (single mode)')
    parser.add_argument('--type', choices=TYPES, default=None, dest='experiment_type',
                        help='Session type: f, p, bf (both, freq CS+), bp (both, azimuth CS+)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random)')
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(0, 999999)

    # ---- Single mode ----
    if args.participant is not None:
        exp_type = args.experiment_type
        if exp_type is None:
            print("Session types:  f = freq,  p = azimuth,  bf = both (freq CS+),  bp = both (azimuth CS+)")
            exp_type = input("Enter session type (f / p / bf / bp): ").strip().lower()
            while exp_type not in TYPES:
                exp_type = input("  Invalid — enter f, p, bf, or bp: ").strip().lower()
        out_dir = 'sequences'
        os.makedirs(out_dir, exist_ok=True)
        info = generate_one_v2(args.participant, exp_type, base_seed, out_dir=out_dir)
        print(f"\nDone: {info['npy']}")

    # ---- Batch mode ----
    elif args.n_participants is not None:
        n = args.n_participants
        exp_types = [args.experiment_type] if args.experiment_type else TYPES
        out_dir = f"WP1_sequences_v2_n{n}"
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'#' * 70}")
        print(f"  BATCH GENERATION (v2): {n} participants x types {exp_types}")
        print(f"  Base seed: {base_seed}")
        print(f"  Output dir: {out_dir}/")
        print(f"{'#' * 70}")

        summaries = []
        for pid in range(1, n + 1):
            for t in exp_types:
                derived_seed = base_seed + pid * 10 + TYPES.index(t)
                summaries.append(generate_one_v2(pid, t, derived_seed, out_dir))

        print(f"\n{'=' * 70}")
        print(f"WP1 v2 SUMMARY  ({n} participants x {len(exp_types)} types)")
        print(f"{'=' * 70}")
        print(f"{'Sub':>4s} | {'type':>4s} | {'trials':>7s} | {'blocks':>6s} | {'shocks':>6s}")
        print(f"{'':->4s}-+-{'':->4s}-+-{'':->7s}-+-{'':->6s}-+-{'':->6s}")
        for s in summaries:
            print(f"{s['participant_id']:04d} | {s['experiment_type']:>4s} | {s['total_trials']:>7d} | "
                  f"{s['n_blocks']:>6d} | {s['total_shocks']:>6d}")
        print(f"{'=' * 70}")
        print(f"Files saved to: {out_dir}/\n")

    else:
        print("=== WP1 Sequence Generator (v2) ===")
        participant_id = int(input("Enter participant number: "))
        print("Session types:  f = freq,  p = azimuth,  bf = both (freq CS+),  bp = both (azimuth CS+)")
        exp_type = input("Enter session type (f / p / bf / bp): ").strip().lower()
        while exp_type not in TYPES:
            exp_type = input("  Invalid — enter f, p, bf, or bp: ").strip().lower()
        out_dir = 'sequences'
        os.makedirs(out_dir, exist_ok=True)
        info = generate_one_v2(participant_id, exp_type, base_seed, out_dir=out_dir)
        print(f"\nDone! Sequence saved to: {info['npy']}")
