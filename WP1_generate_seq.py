"""
Generate WP1 experiment sequences and save to JSON.

Batch mode (all 3 types for N participants):
    python generate_seq_WP1.py 20
    python generate_seq_WP1.py 20 --seed 42

Single mode:
    python generate_seq_WP1.py --participant 1 --type f

Output:
    WP1_sequences_n20/WP1_sub001_f_seq.json  (batch)
    WP1_sub001_f_seq.json                     (single)
"""
import argparse
import json
import os
import random
import numpy as np
from datetime import datetime


# ============================================================================
# SEQUENCE GENERATION
# ============================================================================

def macke_feqlist(base=700, max_cumsum=4, step=0.1):
    step = 1 + step
    feqlist = [base / (step ** i) for i in range(1, max_cumsum + 1)]
    feqlist.append(base)
    feqlist += [base * (step ** i) for i in range(1, max_cumsum + 1)]
    feqlist.sort()
    return feqlist


def macke_azilist(max_cumsum=4, step=10):
    azilist = [-step * i for i in range(1, max_cumsum + 1)]
    azilist.append(0)
    azilist += [step * i for i in range(1, max_cumsum + 1)]
    azilist.sort()
    return azilist


def create_balanced_deviant_vector(n_trains, max_cumsum=4, max_attempts=1000):
    if n_trains % 2 != 0:
        raise ValueError("n_trains must be even to have equal 1s and -1s")

    deviants = [1] * (n_trains // 2) + [-1] * (n_trains // 2)

    for attempt in range(max_attempts):
        random.shuffle(deviants)
        cumsum = np.cumsum(deviants)
        if np.all(cumsum >= -max_cumsum) and np.all(cumsum <= max_cumsum):
            print(f"  Valid sequence found after {attempt + 1} attempt(s)")
            print(f"  Cumsum range: [{np.min(cumsum)}, {np.max(cumsum)}]")
            return deviants

    raise RuntimeError(
        f"Could not find valid sequence after {max_attempts} attempts. "
        f"Try increasing max_attempts or max_cumsum."
    )


def create_roving_sequence(train_lengths=[5, 6, 7, 8, 9, 10],
                           n_deviants=200, max_cumsum=4, soa=0.3,
                           block_size=50):
    original_n_deviants = n_deviants
    if n_deviants % 2 != 0:
        n_deviants += 1
        print(f"  Adjusted to {n_deviants} deviants (must be even for balance)")

    n_blocks = int(np.ceil(n_deviants / block_size))
    base_deviants_per_block = n_deviants // n_blocks
    remainder = n_deviants % n_blocks

    deviants_per_block_list = [base_deviants_per_block] * n_blocks
    for i in range(remainder):
        deviants_per_block_list[i] += 1

    for i in range(len(deviants_per_block_list)):
        if deviants_per_block_list[i] % 2 != 0:
            deviants_per_block_list[i] += 1

    actual_n_deviants = sum(deviants_per_block_list)

    print(f"\n{'=' * 70}")
    print(f"CREATING BLOCK-BASED SEQUENCE")
    print(f"{'=' * 70}")
    print(f"Requested deviants:     {original_n_deviants}")
    print(f"Actual deviants:        {actual_n_deviants}")
    print(f"Number of sub-blocks:   {n_blocks}")
    print(f"Deviants per sub-block: {deviants_per_block_list}")
    print(f"Train lengths:          {train_lengths} (mean: {np.mean(train_lengths):.1f})")
    print(f"Max cumsum:             +/-{max_cumsum}")
    print(f"{'=' * 70}\n")

    all_sequences = []
    all_deviants = []

    for block_num in range(n_blocks):
        deviants_this_block = deviants_per_block_list[block_num]
        print(f"  Sub-block {block_num + 1}/{n_blocks} ({deviants_this_block} deviants)...")

        n_different_lengths = len(train_lengths)
        base_trains_per_length = deviants_this_block // n_different_lengths
        remainder_trains = deviants_this_block % n_different_lengths

        train_distribution = [base_trains_per_length] * n_different_lengths
        if remainder_trains > 0:
            remainder_indices = random.sample(range(n_different_lengths), remainder_trains)
            for idx in remainder_indices:
                train_distribution[idx] += 1

        train_templates = []
        for length_idx, length in enumerate(train_lengths):
            train_templates.extend([[0] * length] * train_distribution[length_idx])

        random.shuffle(train_templates)
        block_deviants = create_balanced_deviant_vector(len(train_templates), max_cumsum=max_cumsum)

        block_sequence = []
        for deviant, train_template in zip(block_deviants, train_templates):
            block_sequence.extend(train_template)
            block_sequence.append(deviant)

        all_sequences.append(block_sequence)
        all_deviants.extend(block_deviants)

        block_deviant_values = [x for x in block_sequence if x != 0]
        block_cumsum = np.cumsum(block_deviant_values)
        print(f"    {len(block_deviant_values)} deviants, "
              f"{len(block_sequence)} total tones, "
              f"cumsum range [{np.min(block_cumsum)}, {np.max(block_cumsum)}]")

    sequence = [tone for block in all_sequences for tone in block]

    deviant_positions = [i for i, val in enumerate(sequence) if val != 0]
    actual_train_lengths = []
    for i in range(len(deviant_positions)):
        if i == 0:
            train_len = deviant_positions[0]
        else:
            train_len = deviant_positions[i] - deviant_positions[i - 1] - 1
        actual_train_lengths.append(train_len)

    mean_train_length = np.mean(actual_train_lengths)
    total_seconds = len(sequence) * soa
    total_minutes = total_seconds / 60

    print(f"\n{'=' * 70}")
    print(f"FINAL SEQUENCE")
    print(f"{'=' * 70}")
    print(f"Total deviants:      {len(all_deviants)}")
    print(f"  - Up (+1):         {sum(1 for x in all_deviants if x == 1)}")
    print(f"  - Down (-1):       {sum(1 for x in all_deviants if x == -1)}")
    print(f"Total standards:     {sum(1 for x in sequence if x == 0)}")
    print(f"Total tones:         {len(sequence)}")
    print(f"Mean train length:   {mean_train_length:.2f} standards")
    print(f"Total run time:      {total_minutes:.2f} min ({total_seconds:.0f} sec)")
    print(f"{'=' * 70}\n")

    return sequence, all_deviants


# ============================================================================
# REINFORCEMENT SCHEDULE
# ============================================================================

def generate_reinforcement_schedule(sequence, cs_plus_value, n_shock=100,
                                    prob_start=0.9, prob_end=0.1):
    """
    Deterministic reinforcement schedule. Selects exactly n_shock CS+ trials
    for shock delivery, weighted toward early trials (linear decay).
    """
    cs_plus_indices = [i for i, s in enumerate(sequence) if s == cs_plus_value]
    n_cs_plus = len(cs_plus_indices)

    reinforcement = [False] * len(sequence)

    if n_cs_plus == 0:
        return reinforcement, {'n_cs_plus': 0, 'n_reinforced': 0, 'n_clean_cs_plus': 0}

    n_shock = min(n_shock, n_cs_plus)

    # Linear decay weights: early CS+ trials get higher weight
    weights = np.array([
        prob_start - (prob_start - prob_end) * (r / max(n_cs_plus - 1, 1))
        for r in range(n_cs_plus)
    ])
    weights /= weights.sum()

    # Select exactly n_shock CS+ trials (weighted toward early ones)
    shocked_ranks = set(np.random.choice(n_cs_plus, size=n_shock, replace=False, p=weights))

    for rank, trial_idx in enumerate(cs_plus_indices):
        if rank in shocked_ranks:
            reinforcement[trial_idx] = True

    n_clean = n_cs_plus - n_shock

    print(f"\n{'=' * 70}")
    print(f"REINFORCEMENT SCHEDULE (deterministic)")
    print(f"{'=' * 70}")
    print(f"CS+ value:           {cs_plus_value:+d}")
    print(f"Total CS+ trials:    {n_cs_plus}")
    print(f"Shocked CS+ trials:  {n_shock} (weighted {prob_start:.0%} -> {prob_end:.0%})")
    print(f"Clean CS+ trials:    {n_clean} (for MMN)")
    print(f"{'=' * 70}\n")

    schedule_info = {
        'n_cs_plus': n_cs_plus,
        'n_reinforced': n_shock,
        'n_clean_cs_plus': n_clean
    }
    return reinforcement, schedule_info


# ============================================================================
# VALUE RESOLUTION (pre-compute per-trial stimulus values)
# ============================================================================

def resolve_values_list_based(value_list, seq):
    """Walk seq through value_list with index clamping. Return per-trial dicts."""
    current_index = len(value_list) // 2
    trials = []
    for s in seq:
        current_index = max(0, min(current_index + s, len(value_list) - 1))
        trials.append({"value": value_list[current_index], "index": current_index})
    return trials


def resolve_values_pattern(base_freq_list, pattern_structures, seq, step=0.1):
    """Pick random base_freq per trial, compute frequencies."""
    step_mult = 1 + step
    pattern_names = {0: "standard", 1: "up", -1: "down"}
    trials = []
    for s in seq:
        base_freq = random.choice(base_freq_list)
        structure = pattern_structures[s]
        freq_map = {0: base_freq, 1: base_freq * step_mult, -1: base_freq / step_mult}
        trials.append({
            "pattern_name": pattern_names[s],
            "base_freq": base_freq,
            "frequencies": [freq_map[t] for t in structure],
            "structure": structure
        })
    return trials


# ============================================================================
# JSON HELPERS
# ============================================================================

def sanitize_for_json(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


# ============================================================================
# EXPERIMENT CONFIG (shared across all participants)
# ============================================================================

ITI = 0.2
TONE_DURATION = 0.1
ITI_WITHIN_PATTERN = 0.05
MAX_CUMSUM = 4
N_DEVIANTS_BASELINE = 400
N_DEVIANTS_CONDITIONING = 600
N_SHOCK = 100
REINF_PROB_START = 0.9
REINF_PROB_END = 0.1
SHOCK_ONSET_IN_ITI = 0.15

FREQUENCY_LIST = macke_feqlist(700, MAX_CUMSUM, 0.1)
POSITION_LIST = macke_azilist(MAX_CUMSUM, 10)
PATTERN_STRUCTURES = {
    0: [0, 0],      # standard: stay-stay
    1: [0, 1],      # deviant up: stay-up
    -1: [0, -1]     # deviant down: stay-down
}

SOA = ITI + TONE_DURATION

BLOCK_CONFIGS = [
    {'block_num': 1, 'label': 'BASELINE',
     'n_deviants': N_DEVIANTS_BASELINE, 'use_reinforcement': False},
    {'block_num': 2, 'label': 'CONDITIONING',
     'n_deviants': N_DEVIANTS_CONDITIONING, 'use_reinforcement': True},
]


# ============================================================================
# GENERATE ONE PARTICIPANT x TYPE
# ============================================================================

def generate_one(participant_id, experiment_type, seed, out_dir='.'):
    """
    Generate sequences for one participant + experiment type, save to JSON.
    Returns a summary dict.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Counterbalancing: even ID => +1 is CS+, odd ID => -1 is CS+
    cs_plus_value = 1 if participant_id % 2 == 0 else -1
    cs_minus_value = -cs_plus_value

    print(f"\n{'=' * 70}")
    print(f"Sub {participant_id:03d} | type={experiment_type} | CS+={cs_plus_value:+d} | seed={seed}")
    print(f"{'=' * 70}")

    # --- Generate blocks ---
    blocks = []
    for block_cfg in BLOCK_CONFIGS:
        print(f"  Block {block_cfg['block_num']}: {block_cfg['label']}")

        seq, dev = create_roving_sequence(
            train_lengths=[5, 6, 7, 8, 9, 10],
            n_deviants=block_cfg['n_deviants'],
            max_cumsum=MAX_CUMSUM,
            soa=SOA,
            block_size=50
        )

        # Resolve per-trial stimulus values
        if experiment_type in ['f', 'p']:
            value_list = FREQUENCY_LIST if experiment_type == 'f' else POSITION_LIST
            trials = resolve_values_list_based(value_list, seq)
        elif experiment_type == 'a':
            trials = resolve_values_pattern(
                FREQUENCY_LIST, PATTERN_STRUCTURES, seq, step=0.1
            )

        # Reinforcement schedule (conditioning blocks only)
        reinforcement = None
        if block_cfg['use_reinforcement']:
            reinforcement, reinf_info = generate_reinforcement_schedule(
                seq, cs_plus_value, n_shock=N_SHOCK,
                prob_start=REINF_PROB_START, prob_end=REINF_PROB_END
            )

        blocks.append({
            'block_num': block_cfg['block_num'],
            'label': block_cfg['label'],
            'sequence': seq,
            'reinforcement': reinforcement,
            'trials': trials
        })

    # --- Build output ---
    output = sanitize_for_json({
        'metadata': {
            'participant_id': participant_id,
            'experiment_type': experiment_type,
            'cs_plus_value': cs_plus_value,
            'cs_minus_value': cs_minus_value,
            'ITI': ITI,
            'tone_duration': TONE_DURATION,
            'iti_within_pattern': ITI_WITHIN_PATTERN,
            'max_cumsum': MAX_CUMSUM,
            'shock_onset_in_iti': SHOCK_ONSET_IN_ITI,
            'n_deviants_baseline': N_DEVIANTS_BASELINE,
            'n_deviants_conditioning': N_DEVIANTS_CONDITIONING,
            'n_shock': N_SHOCK,
            'reinf_prob_start': REINF_PROB_START,
            'reinf_prob_end': REINF_PROB_END,
            'frequency_list': FREQUENCY_LIST,
            'position_list': POSITION_LIST,
            'pattern_structures': PATTERN_STRUCTURES,
            'generated_at': datetime.now().isoformat(),
            'random_seed': seed
        },
        'blocks': blocks
    })

    # --- Save JSON ---
    filename = os.path.join(out_dir, f"WP1_sub{participant_id:03d}_{experiment_type}_seq.json")
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    # --- Summary ---
    total_trials = sum(len(b['sequence']) for b in blocks)
    total_deviants = sum(
        sum(1 for s in b['sequence'] if s != 0) for b in blocks
    )
    total_shocks = sum(
        sum(1 for r in (b['reinforcement'] or []) if r) for b in blocks
    )

    print(f"  -> {filename} ({total_trials} trials)")

    return {
        'participant_id': participant_id,
        'experiment_type': experiment_type,
        'cs_plus_value': cs_plus_value,
        'total_trials': total_trials,
        'total_deviants': total_deviants,
        'total_shocks': total_shocks,
        'filename': filename,
        'seed': seed
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate WP1 experiment sequences and save to JSON.'
    )
    parser.add_argument('n_participants', type=int, nargs='?', default=None,
                        help='Number of participants (batch mode: generates f, p, a for each)')
    parser.add_argument('--participant', type=int, default=None,
                        help='Single participant ID (single mode)')
    parser.add_argument('--type', choices=['f', 'p', 'a'], default=None,
                        dest='experiment_type',
                        help='Single experiment type (single mode)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: random)')
    args = parser.parse_args()

    base_seed = args.seed if args.seed is not None else random.randint(0, 999999)
    types = ['f', 'p', 'a']
    type_names = {'f': 'freq', 'p': 'pos', 'a': 'abstract'}

    # ---- Single mode: --participant P --type T ----
    if args.participant is not None and args.experiment_type is not None:
        info = generate_one(args.participant, args.experiment_type, base_seed,
                            out_dir='../PythonProject')
        print(f"\nDone: {info['filename']}")

    # ---- Batch mode: N ----
    elif args.n_participants is not None:
        n = args.n_participants
        out_dir = f"WP1_sequences_n{n}"
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'#' * 70}")
        print(f"  BATCH GENERATION: {n} participants x {len(types)} types = {n * len(types)} files")
        print(f"  Base seed: {base_seed}")
        print(f"  Output dir: {out_dir}/")
        print(f"{'#' * 70}")

        summaries = []
        for pid in range(1, n + 1):
            for exp_type in types:
                derived_seed = base_seed + pid * 10 + types.index(exp_type)
                info = generate_one(pid, exp_type, derived_seed, out_dir)
                summaries.append(info)

        # --- Print design matrix ---
        print(f"\n{'=' * 70}")
        print(f"WP1 DESIGN MATRIX  ({n} participants x {len(types)} types)")
        print(f"{'=' * 70}")
        print(f"{'Sub':>4s} | {'CS+':>3s} | ", end='')
        print(' | '.join(f"{type_names[t]:>15s}" for t in types))
        print(f"{'':->4s}-+-{'':->3s}-+-", end='')
        print('-+-'.join(f"{'':->15s}" for _ in types))

        for pid in range(1, n + 1):
            pid_rows = [s for s in summaries if s['participant_id'] == pid]
            cs_plus = pid_rows[0]['cs_plus_value']
            print(f"{pid:04d} | {cs_plus:+2d} | ", end='')
            cells = []
            for t in types:
                row = next(s for s in pid_rows if s['experiment_type'] == t)
                cells.append(f"{row['total_trials']:>15d}")
            print(' | '.join(cells))

        print(f"{'=' * 70}")
        print(f"Files saved to: {out_dir}/")
        print(f"Total files:    {len(summaries)}")
        print(f"Base seed:      {base_seed}")
        print(f"{'=' * 70}\n")

    else:
        # Interactive mode: ask the user
        print("=== WP1 Sequence Generator ===")
        participant_id = int(input("Enter participant number: "))
        print("Experiment types:  f = frequency,  p = position,  a = abstract")
        experiment_type = input("Enter experiment type (f / p / a): ").strip().lower()
        while experiment_type not in ['f', 'p', 'a']:
            experiment_type = input("  Invalid — please enter f, p, or a: ").strip().lower()
        seed = base_seed
        out_dir = 'sequences'
        os.makedirs(out_dir, exist_ok=True)
        info = generate_one(participant_id, experiment_type, seed, out_dir=out_dir)
        print(f"\nDone! Sequence saved to: {info['filename']}")
