import slab
import random
import numpy as np
import time
import freefield as ff
from pathlib import Path

def macke_feqlist(base=700, max_cumsum=4, step=0.1):
    step = 1 + step
    feqlist = [base / (step ** i) for i in range(1, max_cumsum + 1)]
    feqlist.append(base)
    feqlist += [base * (step ** i) for i in range(1, max_cumsum + 1)]
    feqlist.sort()
    return feqlist

def macke_azlist(max_cumsum=4, step=10):
    azlist = [-step * i for i in range(1, max_cumsum + 1)]
    azlist.append(0)
    azlist += [step * i for i in range(1, max_cumsum + 1)]
    azlist.sort()
    return azlist

def create_balanced_deviant_vector(n_trains, max_cumsum=4, max_attempts=1000):
    if n_trains % 2 != 0:
        raise ValueError("n_trains must be even to have equal 1s and -1s")

    deviants = [1] * (n_trains // 2) + [-1] * (n_trains // 2)

    for attempt in range(max_attempts):
        random.shuffle(deviants)
        cumsum = np.cumsum(deviants)
        if np.all(cumsum >= -max_cumsum) and np.all(cumsum <= max_cumsum):
            print(f"✓ Valid sequence found after {attempt + 1} attempt(s)")
            print(f"  Cumsum range: [{np.min(cumsum)}, {np.max(cumsum)}]")
            return deviants

    raise RuntimeError(f"Could not find valid sequence after {max_attempts} attempts. Try increasing max_attempts or max_cumsum.")

def create_roving_sequence(train_lengths=[5, 6, 7, 8, 9, 10],
                           n_deviants=200,
                           max_cumsum=4,
                           soa=0.3,
                           block_size=50):
    original_n_deviants = n_deviants
    if n_deviants % 2 != 0:
        n_deviants += 1
        print(f"⚠ Adjusted to {n_deviants} deviants (must be even for balance)")

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
    print(f"Number of blocks:       {n_blocks}")
    print(f"Deviants per block:     {deviants_per_block_list}")
    print(f"Train lengths:          {train_lengths} (mean: {np.mean(train_lengths):.1f})")
    print(f"Max cumsum:             ±{max_cumsum}")
    print(f"{'=' * 70}\n")

    all_sequences = []
    all_deviants = []

    for block_num in range(n_blocks):
        deviants_this_block = deviants_per_block_list[block_num]
        print(f"Creating Block {block_num + 1}/{n_blocks} ({deviants_this_block} deviants)...")

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
        print(f"  ✓ Block {block_num + 1}: {len(block_deviant_values)} deviants, "
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

def make_list_based_seq(value_list, seq, duration=0.1, stim_type='frequency'):
    start_index = len(value_list) // 2
    current_index = start_index

    tones = []
    values = []
    indices = []

    print(f"\n{'=' * 70}")
    print(f"CREATING LIST-BASED {stim_type.upper()} SEQUENCE")
    print(f"{'=' * 70}")
    print(f"Value list:          {value_list}")
    print(f"List length:         {len(value_list)}")
    print(f"Starting index:      {start_index} (value: {value_list[start_index]})")
    print(f"Max index range:     0 to {len(value_list) - 1}")
    print(f"{'=' * 70}\n")

    for s in seq:
        current_index = max(0, min(current_index + s, len(value_list) - 1))
        value = value_list[current_index]

        if stim_type == 'frequency':
            tone = slab.Sound.tone(frequency=value, duration=duration).ramp('offset', 0.02)
        elif stim_type == 'position':
            tone = slab.Sound.tone(frequency=700, duration=duration, n_channels=2)
            tone = slab.Binaural(tone).at_azimuth(value).ramp('both', 0.02)
        else:
            raise ValueError("stim_type must be 'frequency' or 'position'")

        tones.append(tone)
        values.append(value)
        indices.append(current_index)

    return tones, values, indices

def make_pattern_seq(base_freq_list, pattern_structures, seq,
                     duration_per_tone=0.1, iti_within_pattern=0.05):
    pattern_names = {0: 'standard', 1: 'CS+', -1: 'CS-'}

    print(f"\n{'=' * 70}")
    print(f"CREATING RANDOM-BASE PATTERN SEQUENCE")
    print(f"{'=' * 70}")
    print(f"Base frequency list: {[f'{f:.1f}' for f in base_freq_list]}")
    print(f"List length:         {len(base_freq_list)}")
    print(f"Base freq selection: RANDOM for each pattern")
    print(f"")
    print(f"Pattern structures:")
    for key, structure in pattern_structures.items():
        name = pattern_names[key]
        structure_labels = ['LOW' if x == 0 else 'HIGH' for x in structure]
        print(f"  {name:>8s} ({key:+2d}): {' - '.join(structure_labels)} {structure}")
    print(f"")
    print(f"Duration per tone:   {duration_per_tone * 1000:.0f} ms")
    print(f"ITI within pattern:  {iti_within_pattern * 1000:.0f} ms")
    print(f"{'=' * 70}\n")

    patterns_to_play = []
    pattern_info = []

    for s in seq:
        base_freq = random.choice(base_freq_list)
        stay_freq = base_freq
        up_freq = base_freq * 1.1
        down_freq = base_freq / 1.1

        structure = pattern_structures[s]
        pattern_name = pattern_names[s]

        freq_map = {0: stay_freq, 1: up_freq, -1: down_freq}
        pattern_tones = [slab.Sound.tone(frequency=freq_map[tone_type], duration=duration_per_tone).ramp('offset', 0.01) for tone_type in structure]
        pattern_freqs = [freq_map[tone_type] for tone_type in structure]

        patterns_to_play.append((pattern_tones, pattern_name, base_freq))
        pattern_info.append({
            'pattern_name': pattern_name,
            'base_freq': base_freq,
            'frequencies': pattern_freqs,
            'structure': structure
        })

    return patterns_to_play, pattern_info, iti_within_pattern

# ============================================================================
# EXPERIMENT
# ============================================================================

# Timing parameters
ITI = 0.2
tone_duration = 0.1
iti_within_pattern = 0.05
max_cumsum = 4
fs = 48828.125
slab.set_default_samplerate(fs)
data_dir = r'C:\Users\neurobio\Projects\roving_oddball\Skripts\.venv\Lib\site-packages\freefield\data\rcx'
rcx_file = 'bi_play_buf.rcx'
procsser = 'RP2'

frequency_list = macke_feqlist(700, max_cumsum, 0.1)
position_list = macke_azlist(4, 10)
pattern_structures = {
    0: [0, 0, 1],  # standard: low-low-high
    1: [0, 1, 0],  # CS+: low-high-low
    -1: [0, 1, 1]  # CS-: low-high-high
}
# Choose experiment type
experiment_type = ('p')  # 'f', 'p', or 'a'

ff.initialize(
    setup= 'headphones',
    device=[['RX81', 'RX8', ff.DIR / 'data' / 'rcx' / 'bits.rcx'],
        ['RX82', 'RX8', ff.DIR / 'data' / 'rcx' / 'bits.rcx'],
        ['RP2',procsser, ff.DIR / 'data' / 'rcx' / rcx_file]],
    zbus= True,
    connection='GB'
)

# Create roving sequence
sequence, deviants = create_roving_sequence(
    train_lengths=[5, 6, 7, 8, 9, 10],
    n_deviants=400,
    max_cumsum=max_cumsum,
    soa=ITI + tone_duration,
    block_size=50
)

# Generate stimuli
if experiment_type == 'f':
    tones, values, indices = make_list_based_seq(
        value_list=frequency_list,
        seq=sequence,
        duration=tone_duration,
        stim_type='frequency'
    )
    value_label = "freq"
    value_unit = "Hz"
    print("\n🎵 Creating FREQUENCY roving sequence...")

elif experiment_type == 'p':
    tones, values, indices = make_list_based_seq(
        value_list=position_list,
        seq=sequence,
        duration=tone_duration,
        stim_type='position'
    )
    value_label = "azimuth"
    value_unit = "°"
    print("\n🎧 Creating POSITION roving sequence...")

elif experiment_type == 'a':
    patterns_to_play, pattern_info, iti_within = make_pattern_seq(
        base_freq_list=frequency_list,
        pattern_structures=pattern_structures,
        seq=sequence,
        duration_per_tone=tone_duration,
        iti_within_pattern=iti_within_pattern
    )
    print("\n🎭 Creating ABSTRACT pattern sequence...")

# Play experiment
print("\nPress Enter to start playing...")
input()

print(f"\n{'=' * 70}")
print(f"PLAYING EXPERIMENT: {experiment_type.upper()}")
print(f"{'=' * 70}\n")

if experiment_type in ['f', 'p']:
    for i, tone in enumerate(tones):
        time1 = time.time()
        marker = "🔴 DEV" if sequence[i] != 0 else "⚪ STD"
        print(
            f"Tone {i + 1:3d}/{len(sequence)}: {marker} | {value_label}={values[i]:7.1f} {value_unit} | index={indices[i]-max_cumsum}")
        ff.write('playbuflen',len(tone),procsser)
        ff.write('data_l',tone.data ,procsser)
        ff.write('chan_l',1,procsser)
        ff.write('data_r', tone.data, procsser)
        ff.write('chan_r', 2, procsser)
        time2 = time.time()

        time.sleep(ITI+time1-time2)

        ff.play('zBusA')
        ff.wait_to_finish_playing()




elif experiment_type == 'a':
    for i, (pattern_tones, pattern_name, base_freq) in enumerate(patterns_to_play):
        marker = "🔴 DEV" if sequence[i] != 0 else "⚪ STD"
        info = pattern_info[i]
        freqs_str = '-'.join([f"{f:.0f}" for f in info['frequencies']])
        print(f"Trial {i + 1:3d}/{len(sequence)}: {marker} | {pattern_name:>8s} | "
              f"base={base_freq:.0f}Hz tones={freqs_str}")

        for tone_idx, tone in enumerate(pattern_tones):
            tone.play()
            if tone_idx < len(pattern_tones) - 1:
                time.sleep(iti_within)

        time.sleep(ITI)

