"""
Run WP1 experiment from a pre-generated JSON sequence file,
with Pupil Labs eye-tracker recording and trial annotations.

Usage:
    python WP1_run_Exp_ff_pupil.py WP1_sub001_f_seq.json
"""
import sys
import os
import json
import time
import csv
import subprocess
import numpy as np
import slab
import freefield as ff
import zmq
import msgpack as serializer

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
# PUPIL LABS SETUP
# ============================================================================

def start_pupil_capture():
    """Launch Pupil Capture as a subprocess."""
    command = [
        "python",
        r"C:\Users\neurobio\Projects\WP1_roving_oddball\pupil\pupil_src\main.py",
        "capture"
    ]
    proc = subprocess.Popen(command)
    print("Waiting for Pupil Capture to start...")
    time.sleep(6)  # give it time to fully open
    return proc


def connect_to_pupil():
    """Connect to Pupil Remote and return (pupil_remote, pub_socket)."""
    ctx = zmq.Context()
    pupil_remote = zmq.Socket(ctx, zmq.REQ)
    pupil_remote.connect('tcp://127.0.0.1:50020')

    # Get the PUB port for sending annotations
    pupil_remote.send_string("PUB_PORT")
    pub_port = pupil_remote.recv_string()
    pub_socket = zmq.Socket(ctx, zmq.PUB)
    pub_socket.connect("tcp://127.0.0.1:{}".format(pub_port))
    time.sleep(0.5)  # allow PUB socket to connect

    print(f"Connected to Pupil Capture (PUB port: {pub_port})")
    return pupil_remote, pub_socket


def pupil_notify(pupil_remote, notification):
    """Send a notification to Pupil Remote."""
    topic = "notify." + notification["subject"]
    payload = serializer.dumps(notification, use_bin_type=True)
    pupil_remote.send_string(topic, flags=zmq.SNDMORE)
    pupil_remote.send(payload)
    return pupil_remote.recv_string()


def get_pupil_time(pupil_remote):
    """Get current Pupil Capture timestamp (float, in seconds)."""
    pupil_remote.send_string("t")
    return float(pupil_remote.recv_string())


def send_annotation(pub_socket, label, pupil_remote, duration=0.0, extra=None):
    """
    Send a timestamped annotation to Pupil Capture.
    extra: optional dict of additional key-value pairs added to the annotation.
    """
    timestamp = get_pupil_time(pupil_remote)
    annotation = {
        "topic": "annotation",
        "label": label,
        "timestamp": timestamp,
        "duration": duration,
    }
    if extra:
        annotation.update(extra)
    payload = serializer.dumps(annotation, use_bin_type=True)
    pub_socket.send_string(annotation["topic"], flags=zmq.SNDMORE)
    pub_socket.send(payload)
    return timestamp


def start_pupil_recording(pupil_remote, pub_socket):
    """Enable annotation plugin and start recording."""
    pupil_notify(pupil_remote, {
        "subject": "start_plugin",
        "name": "Annotation_Capture",
        "args": {}
    })
    pupil_remote.send_string('R')
    response = pupil_remote.recv_string()
    print(f"Pupil recording started: {response}")
    send_annotation(pub_socket, "experiment_start", pupil_remote)


def stop_pupil_recording(pupil_remote, pub_socket):
    """Send end annotation and stop recording."""
    send_annotation(pub_socket, "experiment_end", pupil_remote)
    time.sleep(0.5)  # make sure last annotation is written before stopping
    pupil_remote.send_string('r')
    response = pupil_remote.recv_string()
    print(f"Pupil recording stopped: {response}")


# ============================================================================
# PRECISE TIMING
# ============================================================================

def precise_sleep_until(target_time, busy_wait_threshold=0.002):
    remaining = target_time - time.time()
    if remaining > busy_wait_threshold:
        time.sleep(remaining - busy_wait_threshold)
    while time.time() < target_time:
        pass


# ============================================================================
# SOUND CREATION (from JSON trial data)
# ============================================================================

def create_sounds(trials, experiment_type, tone_duration, iti_within_pattern=0.05):
    """
    Create slab.Sound objects from pre-resolved JSON trial data.

    For f/p:  returns (tones, values, indices)
    For a:    returns (patterns, pattern_info)
    """
    if experiment_type in ['f', 'p']:
        tones, values, indices = [], [], []
        for t in trials:
            if experiment_type == 'f':
                tone = slab.Sound.tone(frequency=t['value'], duration=tone_duration)
                tone = slab.Binaural(tone).ramp('offset', 0.05)
            else:
                tone = slab.Sound.tone(frequency=700, duration=tone_duration, n_channels=2)
                tone = slab.Binaural(tone).at_azimuth(t['value']).ramp('offset', 0.05)
            tones.append(tone)
            values.append(t['value'])
            indices.append(t['index'])
        return tones, values, indices

    elif experiment_type == 'a':
        patterns, pattern_info = [], []
        silence_samples = int(iti_within_pattern * slab.get_default_samplerate())
        for t in trials:
            pattern_tones = [
                slab.Sound.tone(frequency=f, duration=tone_duration / 2).ramp('offset', 0.05)
                for f in t['frequencies']
            ]
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
              participant_id, cs_plus_value, ITI, SOA, tone_duration, trial_log,
              pupil_remote, pub_socket,
              reinforcement=None, shock_onset=0.25, max_cumsum=4,
              A_SOA=0.65, iti_within_pattern=0.1):
    """Play one block, send Pupil annotations, and log all trials."""
    print(f"\n{'=' * 70}")
    print(f"PLAYING BLOCK {block_num}: {block_label}")
    print(f"{'=' * 70}\n")

    send_annotation(pub_socket, f"block_{block_num}_start", pupil_remote,
                    extra={"block_label": block_label})

    cs_minus_value = -cs_plus_value

    if experiment_type in ['f', 'p']:
        tones, values, indices = stimuli
        value_label = "freq" if experiment_type == 'f' else "azimuth"
        value_unit  = "Hz"  if experiment_type == 'f' else "deg"
        ff.write('playbuflen', len(tones[0]), procsser)
        ff.write('data_l', tones[0].left.data, procsser)
        ff.write('chan_l', 1, procsser)
        ff.write('data_r', tones[0].right.data, procsser)
        ff.write('chan_r', 2, procsser)
    elif experiment_type == 'a':
        patterns, pattern_info = stimuli
        ff.write('playbuflen', len(patterns[0]), procsser)
        ff.write('data_l', patterns[0].data, procsser)
        ff.write('chan_l', 1, procsser)
        ff.write('data_r', patterns[0].data, procsser)
        ff.write('chan_r', 2, procsser)

    for i in range(len(sequence)):

        # --- CS label ---
        if sequence[i] == cs_plus_value:
            cs_label = "CS+"
        elif sequence[i] == cs_minus_value:
            cs_label = "CS-"
        else:
            cs_label = "STD"
        marker = "DEV" if sequence[i] != 0 else "STD"
        shock_delivered = reinforcement is not None and reinforcement[i]

        # --- Play + annotate ---
        if experiment_type in ['f', 'p']:
            t_onset = time.time()
            ff.play(1, [procsser])

            # Send trial annotation immediately after triggering sound
            send_annotation(pub_socket, f"trial_{marker}_{cs_label}", pupil_remote,
                            extra={
                                "block": block_num,
                                "trial": i + 1,
                                "cs_label": cs_label,
                                "marker": marker,
                                value_label: values[i],
                                "shock": int(shock_delivered)
                            })

            ff.wait_to_finish_playing()
            stimulus_value = values[i]

            print(f"Tone {i + 1:3d}/{len(sequence)}: {marker} {cs_label:>3s} | "
                  f"{value_label}={values[i]:7.1f} {value_unit} | "
                  f"index={indices[i] - max_cumsum}"
                  f"{' | SHOCK' if shock_delivered else ''}")

            if i + 1 < len(tones):
                ff.write('playbuflen', len(tones[i + 1]), procsser)
                ff.write('data_l', tones[i + 1].left.data, procsser)
                ff.write('chan_l', 1, procsser)
                ff.write('data_r', tones[i + 1].right.data, procsser)
                ff.write('chan_r', 2, procsser)

            if shock_delivered:
                precise_sleep_until(t_onset + shock_onset)
                ff.play(2, [procsser])
                send_annotation(pub_socket, "shock", pupil_remote,
                                extra={"block": block_num, "trial": i + 1})

            precise_sleep_until(t_onset + SOA)

        elif experiment_type == 'a':
            info = pattern_info[i]
            freqs_str = '-'.join([f"{f:.0f}" for f in info['frequencies']])
            stimulus_value = info['base_freq']

            t_onset = time.time()
            ff.play(1, [procsser])

            send_annotation(pub_socket, f"trial_{marker}_{cs_label}", pupil_remote,
                            extra={
                                "block": block_num,
                                "trial": i + 1,
                                "cs_label": cs_label,
                                "marker": marker,
                                "base_freq": info['base_freq'],
                                "pattern": info['pattern_name'],
                                "shock": int(shock_delivered)
                            })

            print(f"Trial {i + 1:3d}/{len(sequence)}: {marker} {cs_label:>3s} | "
                  f"{info['pattern_name']:>8s} | base={info['base_freq']:.0f}Hz "
                  f"tones={freqs_str}{' | SHOCK' if shock_delivered else ''}")

            ff.wait_to_finish_playing()

            if shock_delivered:
                precise_sleep_until(t_onset + shock_onset + iti_within_pattern)
                ff.play(2, [procsser])
                send_annotation(pub_socket, "shock", pupil_remote,
                                extra={"block": block_num, "trial": i + 1})

            if i + 1 < len(patterns):
                ff.write('playbuflen', len(patterns[i + 1]), procsser)
                ff.write('data_l', patterns[i + 1].data, procsser)
                ff.write('chan_l', 1, procsser)
                ff.write('data_r', patterns[i + 1].data, procsser)
                ff.write('chan_r', 2, procsser)

            precise_sleep_until(t_onset + A_SOA)

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

    send_annotation(pub_socket, f"block_{block_num}_end", pupil_remote,
                    extra={"block_label": block_label})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) == 2:
        seq_file = sys.argv[1]
    else:
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
    participant_id   = meta['participant_id']
    experiment_type  = meta['experiment_type']
    cs_plus_value    = meta['cs_plus_value']
    ITI              = meta['ITI']
    tone_duration    = meta['tone_duration']
    iti_within_pattern = meta.get('iti_within_pattern', 0.05)
    shock_onset      = meta.get('shock_onset_in_iti', 0.25)
    max_cumsum       = meta.get('max_cumsum', 4)

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

    # ── Start Pupil Capture ──────────────────────────────────────────────────

    pupil_remote, pub_socket = connect_to_pupil()
    start_pupil_recording(pupil_remote, pub_socket)

    # ── Run experiment blocks ────────────────────────────────────────────────
    trial_log = []

    for block_data in data['blocks']:
        block_num     = block_data['block_num']
        label         = block_data['label']
        seq           = block_data['sequence']
        reinforcement = block_data['reinforcement']

        n_devs = sum(1 for s in seq if s != 0)
        print(f"\n>>> PREPARING BLOCK {block_num}: {label} <<<")
        print(f"    {len(seq)} trials, {n_devs} deviants")

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
            SOA=ITI + tone_duration,
            tone_duration=tone_duration,
            trial_log=trial_log,
            pupil_remote=pupil_remote,
            pub_socket=pub_socket,
            reinforcement=reinforcement,
            shock_onset=shock_onset,
            max_cumsum=max_cumsum,
            iti_within_pattern=iti_within_pattern
        )

        if block_num < len(data['blocks']):
            print(f"\n{'=' * 70}")
            print(f"BLOCK {block_num} COMPLETE. Press Enter to continue...")
            print(f"{'=' * 70}")
            input()

    # ── Stop Pupil recording ─────────────────────────────────────────────────
    stop_pupil_recording(pupil_remote, pub_socket)

    # ── Save behavioural data ────────────────────────────────────────────────
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