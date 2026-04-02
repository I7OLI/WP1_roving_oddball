"""
Run WP1 experiment from a pre-generated JSON sequence file.

Usage:
    python run_Exp_WP1.py WP1_sub001_f_seq.json
"""
import sys
import os
import json
import time
import csv
import numpy as np
import slab
import freefield as ff

# ============================================================================
# FREEFIELD CONFIGURATION
# ============================================================================
fs = 48828.125
slab.set_default_samplerate(fs)
rcx_file = 'shock.rcx'
procsser = 'RM1'

ff.initialize(
        setup='headphones',
        device=[['RM1', procsser, rcx_file]],
        zbus=False,
        connection='usb'
   )
# ============================================================================
# SOUND CREATION (from JSON trial data)
# ============================================================================

def create_sounds(trials, experiment_type, tone_duration, iti_within_pattern=0.05):
    """
    Create slab.Sound objects from pre-resolved JSON trial data.

    For f/p:  returns (tones, values, indices)
    For a:    returns (patterns, pattern_info)
              Each pattern is a single combined slab.Sound:
              [tone1 | silence | tone2 | ...] loaded as one buffer.
    """
    if experiment_type in ['f', 'p']:
        tones = []
        values = []
        indices = []
        for t in trials:
            if experiment_type == 'f':
                tone = slab.Sound.tone(frequency=t['value'], duration=tone_duration)
                tone = tone.ramp('offset', 0.02)
            else:  # 'p'
                tone = slab.Sound.tone(frequency=700, duration=tone_duration, n_channels=2)
                tone = slab.Binaural(tone).at_azimuth(t['value'])
                tone = tone.ramp('offset', 0.02)
            tones.append(tone)
            values.append(t['value'])
            indices.append(t['index'])
        return tones, values, indices

    elif experiment_type == 'a':
        patterns = []
        pattern_info = []
        silence_samples = int(iti_within_pattern * slab.get_default_samplerate())
        for t in trials:
            pattern_tones = [
                slab.Sound.tone(frequency=f, duration=tone_duration).ramp('offset', 0.01)
                for f in t['frequencies']
            ]
            # Concatenate tones with silence between them into one buffer.
            # Single write + single SoftTrg means within-pattern timing is
            # hardware-precise rather than controlled by Python sleeps.
            n_ch = pattern_tones[0].data.shape[1]
            parts = []
            for idx, tone in enumerate(pattern_tones):
                parts.append(tone.data)
                if idx < len(pattern_tones) - 1:
                    parts.append(np.zeros((silence_samples, n_ch)))
            combined = slab.Sound(np.vstack(parts),
                                  samplerate=slab.get_default_samplerate())
            patterns.append(combined)
            pattern_info.append(t)
        return patterns, pattern_info


# ============================================================================
# BLOCK RUNNER
# ============================================================================

def run_block(sequence, stimuli, experiment_type, block_num, block_label,
              participant_id, cs_plus_value, ITI, tone_duration, trial_log,
              reinforcement=None, shock_onset=0.15, max_cumsum=4,
              iti_within_pattern=0.05):
    """Play one block and log all trials."""
    print(f"\n{'=' * 70}")
    print(f"PLAYING BLOCK {block_num}: {block_label}")
    print(f"{'=' * 70}\n")

    cs_minus_value = -cs_plus_value

    # Unpack stimuli and pre-load first buffer
    if experiment_type in ['f', 'p']:
        tones, values, indices = stimuli
        value_label = "freq" if experiment_type == 'f' else "azimuth"
        value_unit = "Hz" if experiment_type == 'f' else "deg"
        ff.write('playbuflen', len(tones[0]), procsser)
        ff.write('data_l', tones[0].data, procsser)
        ff.write('chan_l', 1, procsser)
        ff.write('data_r', tones[0].data, procsser)
        ff.write('chan_r', 2, procsser)
    elif experiment_type == 'a':
        patterns, pattern_info = stimuli
        ff.write('playbuflen', len(patterns[0]), procsser)
        ff.write('data_l', patterns[0].data, procsser)
        ff.write('chan_l', 1, procsser)
        ff.write('data_r', patterns[0].data, procsser)
        ff.write('chan_r', 2, procsser)

    for i in range(len(sequence)):
        t_onset = time.time()

        # --- CS label ---
        if sequence[i] == cs_plus_value:
            cs_label = "CS+"
        elif sequence[i] == cs_minus_value:
            cs_label = "CS-"
        else:
            cs_label = "STD"
        marker = "DEV" if sequence[i] != 0 else "STD"
        shock_delivered = reinforcement is not None and reinforcement[i]

        # --- Print + play (type-specific) ---
        if experiment_type in ['f', 'p']:

            ff.play(1, [procsser])
            time.sleep(tone_duration)
            stimulus_value = values[i]

            print(f"Tone {i + 1:3d}/{len(sequence)}: {marker} {cs_label:>3s} | "
                  f"{value_label}={values[i]:7.1f} {value_unit} | "
                  f"index={indices[i] - max_cumsum}"
                  f"{' | SHOCK' if shock_delivered else ''}")

            # time1 = tone offset; time_elapsed updated if a write happens
            time1 = time.time()
            time_elapsed = 0
            if i + 1 < len(tones):
                ff.write('playbuflen', len(tones[i+1]), procsser)
                ff.write('data_l', tones[i+1].data, procsser)
                ff.write('chan_l', 1, procsser)
                ff.write('data_r', tones[i+1].data, procsser)
                ff.write('chan_r', 2, procsser)
                time_elapsed = time.time() - time1

            # Pre-tone silence = SOA - tone_duration (ITI in JSON is the full SOA)
            pre_tone_silence = ITI - tone_duration

            if shock_delivered:
                shock_wait = max(0, shock_onset - time_elapsed)
                time.sleep(shock_wait)
                ff.play(2,[procsser])
                elapsed = time.time() - time1
                remaining = max(0, pre_tone_silence - elapsed)
                time.sleep(remaining)
            else:
                time.sleep(max(0, pre_tone_silence - time_elapsed))

            # ff.play falls back to SoftTrg(1) when zbus=False.
            # time.sleep(tone_duration) replaces ff.wait_to_finish_playing(),
            # which polls a 'playback' tag that shock.rcx may not expose —
            # causing it to return immediately and shift sounds one tone forward.


        elif experiment_type == 'a':
            info = pattern_info[i]
            freqs_str = '-'.join([f"{f:.0f}" for f in info['frequencies']])

            # Play combined buffer (all tones + within-pattern gaps in one shot)
            ff.play(1, [procsser])
            time.sleep(patterns[i].duration)
            stimulus_value = info['base_freq']

            print(f"Trial {i + 1:3d}/{len(sequence)}: {marker} {cs_label:>3s} | "
                  f"{info['pattern_name']:>8s} | base={info['base_freq']:.0f}Hz "
                  f"tones={freqs_str}{' | SHOCK' if shock_delivered else ''}")

            # ITI: write next pattern buffer, then shock
            time1 = time.time()
            time_elapsed = 0
            if i + 1 < len(patterns):
                ff.write('playbuflen', len(patterns[i+1]), procsser)
                ff.write('data_l', patterns[i+1].data, procsser)
                ff.write('chan_l', 1, procsser)
                ff.write('data_r', patterns[i+1].data, procsser)
                ff.write('chan_r', 2, procsser)
                time_elapsed = time.time() - time1

            post_pattern_gap = ITI - tone_duration
            if shock_delivered:
                time.sleep(max(0, shock_onset - time_elapsed))
                ff.play(2, [procsser])
                elapsed = time.time() - time1
                time.sleep(max(0, post_pattern_gap - elapsed))
            else:
                time.sleep(max(0, post_pattern_gap - time_elapsed))

        # --- Log trial ---
        trial_log.append({
            'participant_id': participant_id,
            'block': block_num,
            'block_label': block_label,
            'trial_num': i + 1,
            'experiment_type': experiment_type,
            'sequence_value': sequence[i],
            'trial_type': cs_label,
            'stimulus_value': stimulus_value,
            'is_cs_plus': sequence[i] == cs_plus_value,
            'shock_delivered': shock_delivered,
            'timestamp': t_onset
        })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) == 2:
        seq_file = sys.argv[1]
    else:
        # Interactive mode: ask for participant and type, find the JSON
        import glob
        print("=== WP1 Experiment Runner ===")
        participant_id_input = int(input("Enter participant number: "))
        print("Experiment types:  f = frequency,  p = position,  a = abstract")
        exp_type_input = input("Enter experiment type (f / p / a): ").strip().lower()
        while exp_type_input not in ['f', 'p', 'a']:
            exp_type_input = input("  Invalid — please enter f, p, or a: ").strip().lower()

        seq_file = os.path.join('sequences', f"WP1_sub{participant_id_input:03d}_{exp_type_input}_seq.json")
        if not os.path.exists(seq_file):
            print(f"\nERROR: Could not find '{seq_file}'")
            print("Please run WP1_generate_seq.py first to create the sequence.")
            sys.exit(1)
        print(f"Found sequence file: {seq_file}\n")
    with open(seq_file) as f:
        data = json.load(f)


    meta = data['metadata']
    participant_id = meta['participant_id']
    experiment_type = meta['experiment_type']
    cs_plus_value = meta['cs_plus_value']
    ITI = meta['ITI']
    tone_duration = meta['tone_duration']
    iti_within_pattern = meta.get('iti_within_pattern', 0.05)
    shock_onset = meta.get('shock_onset_in_iti', 0.15)
    max_cumsum = meta.get('max_cumsum', 4)

    print(f"\n{'=' * 70}")
    print(f"WP1 EXPERIMENT")
    print(f"{'=' * 70}")
    print(f"Sequence file:       {seq_file}")
    print(f"Participant ID:      {participant_id}")
    print(f"Experiment type:     {experiment_type}")
    print(f"CS+ deviant:         {cs_plus_value:+d}")
    print(f"CS- deviant:         {meta['cs_minus_value']:+d}")
    print(f"Blocks:              {len(data['blocks'])}")
    print(f"Seed:                {meta.get('random_seed', 'unknown')}")
    print(f"Generated at:        {meta.get('generated_at', 'unknown')}")
    print(f"{'=' * 70}\n")

    trial_log = []

    for block_data in data['blocks']:
        block_num = block_data['block_num']
        label = block_data['label']
        seq = block_data['sequence']
        reinforcement = block_data['reinforcement']

        n_devs = sum(1 for s in seq if s != 0)
        print(f"\n>>> PREPARING BLOCK {block_num}: {label} <<<")
        print(f"    {len(seq)} trials, {n_devs} deviants")

        # Create slab.Sound objects from pre-resolved trial data
        stimuli = create_sounds(block_data['trials'], experiment_type,
                                tone_duration, iti_within_pattern)

        print(f"\nPress Enter to start BLOCK {block_num} ({label})...")
        input()

        run_block(
            sequence=seq,
            stimuli=stimuli,
            experiment_type=experiment_type,
            block_num=block_num,
            block_label=label,
            participant_id=participant_id,
            cs_plus_value=cs_plus_value,
            ITI=ITI,
            tone_duration=tone_duration,
            trial_log=trial_log,
            reinforcement=reinforcement,
            shock_onset=shock_onset,
            max_cumsum=max_cumsum,
            iti_within_pattern=iti_within_pattern
        )

        # Pause after all blocks except the last
        if block_num < len(data['blocks']):
            print(f"\n{'=' * 70}")
            print(f"BLOCK {block_num} COMPLETE. Press Enter to continue...")
            print(f"{'=' * 70}")
            input()

    # ========================================================================
    # SAVE DATA
    # ========================================================================

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'=' * 70}")

    filename = f"WP1_sub{participant_id:03d}_{experiment_type}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trial_log[0].keys())
        writer.writeheader()
        writer.writerows(trial_log)

    print(f"Data saved to {filename}")
    print(f"Total trials logged: {len(trial_log)}")
