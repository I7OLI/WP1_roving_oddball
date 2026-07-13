import msgpack
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import os

matplotlib.use('TkAgg')

#file_path = "D:/AudAtt/Data/EyeTracking/002_session1/"
file_path = "/Users/oliver/PycharmProjects/WP1_roving_oddball/Data/sub101_a/2026_05_13/000/"
filename = file_path + "pupil.pldata"


# Load raw binary data
with open(file_path + "pupil.pldata", "rb") as f:
    raw_data = f.read()

# could not load full data; idea: load a part, store in data frame, delete raw data, do same with next part
file_size = os.path.getsize(filename)
part = file_size // 10



with open(filename, "rb") as f:
    raw_data = f.read(part)

# Unpack binary messages
unpacker = msgpack.Unpacker(strict_map_key=False)
unpacker.feed(raw_data)

# Convert to list of dictionaries
pupil_data = list(unpacker)

decoded_data = [
    msgpack.unpackb(payload, raw=False)
    for topic, payload in pupil_data
]

# Create DataFrame
df = pd.DataFrame(decoded_data)



print(df.head())
print(df.columns)


# Filter high-confidence data
df_clean = df[df['confidence'] > 0.9]

df_2d = df_clean[df_clean['topic'] == 'pupil.0.2d']
df_3d = df_clean[df_clean['topic'] == 'pupil.0.3d']


print("Mean diameter:", df_3d['diameter'].mean())
print("Max diameter:", df_3d['diameter'].max())
print("Min diameter:", df_3d['diameter'].min())
print("Std deviation:", df_3d['diameter'].std())


plt.figure(figsize=(10, 4))
plt.plot(df_3d['timestamp'], df_3d['diameter_3d'])
plt.xlabel("Time (s)")
plt.ylabel("Pupil Diameter (px)")
plt.title("Pupil Size Over Time")
plt.grid(True)
plt.show(block=True)

plt.figure(figsize=(10, 4))
plt.hist(df_3d['model_confidence'])
plt.grid(True)
plt.show(block=True)


directory = (file_path + "offline_data/")
if not os.path.exists(directory):
    os.makedirs(directory)

df_3d.to_csv(directory + "pupil_data_clean.csv", index=False)

df_3d_minimal = df_3d[["timestamp","diameter_3d"]]
df_3d_minimal.to_csv(directory + "pupil_data_minimal.csv", index=False)

### add annotations

# Load annotation data
with open(file_path + "annotation.pldata", "rb") as f:
    raw_annotations = f.read()

# Unpack annotations
annotation_unpacker = msgpack.Unpacker(strict_map_key=False)
annotation_unpacker.feed(raw_annotations)
annotations = list(annotation_unpacker)

# Decode annotation payloads
decoded_annotations = [
    msgpack.unpackb(payload, raw=False)
    for topic, payload in annotations
]

# Create DataFrame for annotations
df_annotations = pd.DataFrame(decoded_annotations)

print(df_annotations.head())


# --- Plot pupil data with annotations ---
plt.figure(figsize=(10, 4))
plt.plot(df_3d['timestamp'], df_3d['diameter'], label="Pupil Diameter")

# Overlay annotations as vertical lines
for _, row in df_annotations.iterrows():
    timestamp = row.get('timestamp')
    label = row.get('label', 'Annotation')
    if pd.notnull(timestamp):
        plt.axvline(x=timestamp, color='red', linestyle='--', alpha=0.6)
        plt.text(timestamp, df_3d['diameter'].max(), label, rotation=90, color='red', va='bottom', fontsize=8)

plt.xlabel("Time (s)")
plt.ylabel("Pupil Diameter (px)")
plt.title("Pupil Size Over Time with Annotations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=True)


####### blinks

# Load raw binary data
with open(file_path + "blinks.pldata", "rb") as f:
    raw_blinks = f.read()

# Unpack binary messages
unpacker = msgpack.Unpacker(strict_map_key=False)
unpacker.feed(raw_blinks)

# Convert to list of dictionaries
blinks_data = list(unpacker)

decoded_blinks_data = [
    msgpack.unpackb(payload, raw=False)
    for topic, payload in blinks_data
]

# Create DataFrame
df_blinks = pd.DataFrame(decoded_blinks_data)

print(df_blinks.head())
print(df_blinks.columns)

df_blinks_high_conf = df_blinks[df_blinks['confidence'] > 0.6].copy()

onsets = df_blinks_high_conf[df_blinks_high_conf['type'] == 'onset']
offsets = df_blinks_high_conf[df_blinks_high_conf['type'] == 'offset']

plt.figure(figsize=(10, 4))
plt.plot(df_3d['timestamp'], df_3d['diameter'], label="Pupil Diameter")

# Plot high-confidence blink onsets
for t in onsets['timestamp']:
    plt.axvline(x=t, color='red', linestyle='--', alpha=0.6, label='Blink Onset' if 'Blink Onset' not in plt.gca().get_legend_handles_labels()[1] else "")

# Plot high-confidence blink offsets
for t in offsets['timestamp']:
    plt.axvline(x=t, color='green', linestyle='--', alpha=0.6, label='Blink Offset' if 'Blink Offset' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel("Time (s)")
plt.ylabel("Pupil Diameter (px)")
plt.title("Pupil Diameter with High-Confidence Blink Onsets and Offsets")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=True)


############################### saccades
# Extract 2D pupil positions
df_2d[['x', 'y']] = pd.DataFrame(df_2d['norm_pos'].tolist(), index=df_2d.index)

# Calculate velocity (difference between consecutive pupil positions)
df_2d['dx'] = df_2d['x'].diff()
df_2d['dy'] = df_2d['y'].diff()
df_2d['dt'] = df_2d['timestamp'].diff()
df_2d['velocity'] = np.sqrt(df_2d['dx']**2 + df_2d['dy']**2) / df_2d['dt']
df_2d['horizontal_velocity'] = np.abs(df_2d['dx']) / df_2d['dt']

plt.figure(figsize=(10, 4))
plt.plot(df_2d['timestamp'], df_2d['velocity'])
plt.show(block=True)

# Define saccades as high-velocity events (threshold might need tuning)
VELOCITY_THRESHOLD = 1
df_2d['saccade'] = df_2d['horizontal_velocity'] > VELOCITY_THRESHOLD

# # in 3d:
# df_3d['ellipse_center_x'] = df_3d['ellipse'].apply(lambda e: e['center'][0] if isinstance(e, dict) and 'center' in e else np.nan)
# df_3d['ellipse_center_y'] = df_3d['ellipse'].apply(lambda e: e['center'][1] if isinstance(e, dict) and 'center' in e else np.nan)
#
# # Calculate velocity (difference between consecutive pupil positions)
# df_3d['dx'] = df_3d['ellipse_center_x'].diff()
# df_3d['dy'] = df_3d['ellipse_center_y'].diff()
# df_3d['dt'] = df_3d['timestamp'].diff()
# df_3d['velocity'] = np.sqrt(df_3d['dx']**2 + df_3d['dy']**2) / df_3d['dt']
# df_3d['horizontal_velocity'] = np.abs(df_3d['dx']) / df_3d['dt']
#
# # Define saccades as high-velocity events (threshold might need tuning)
# VELOCITY_THRESHOLD = 100
# df_3d['saccade'] = df_3d['horizontal_velocity'] > VELOCITY_THRESHOLD

# Define velocity threshold and minimum duration (e.g., 5 consecutive frames)
velocity_thresh = 1
min_consecutive = 4

# Create a boolean column
df_2d['is_fast'] = df_2d['horizontal_velocity'] > velocity_thresh

# Group consecutive True values
from itertools import groupby
from operator import itemgetter

df_2d = df_2d.reset_index(drop=True)

fast_indices = df_2d.index[df_2d['is_fast']].tolist()

# Group consecutive indices
saccade_groups = []
for k, g in groupby(enumerate(fast_indices), lambda ix: ix[0] - ix[1]):
    group = list(map(itemgetter(1), g))
    if len(group) >= min_consecutive:
        saccade_groups.append(group)

# Get the first timestamp of each saccade group
saccade_times = [df_2d.loc[g[0], 'timestamp'] for g in saccade_groups]



plt.figure(figsize=(10, 4))
plt.plot(df_3d['timestamp'], df_3d['diameter'], label="Pupil Diameter")

# Overlay detected saccade times
#for t in df_2d[df_2d['saccade']]['timestamp']:
#    plt.axvline(x=t, color='purple', linestyle=':', alpha=0.4)

for t in saccade_times:
    plt.axvline(x=t, color='purple', linestyle='-.', alpha=0.6, label='Saccade' if 'Saccade' not in plt.gca().get_legend_handles_labels()[1] else "")


plt.xlabel("Time (s)")
plt.ylabel("Pupil Diameter (px)")
plt.title("Pupil Diameter with Inferred Saccades (from Pupil 2D Position)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=True)


plt.figure(figsize=(10, 4))
plt.plot(df_2d['timestamp'], df_2d['velocity'])
for t in saccade_times:
    plt.axvline(x=t, color='purple', linestyle='-.', alpha=0.6, label='Saccade' if 'Saccade' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.title("Velocity with Inferred Saccades (from Pupil 2D Position)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=True)

################

plt.figure(figsize=(14, 6))

# === 1. Pupil Diameter ===
plt.plot(df_3d['timestamp'], df_3d['diameter'], label='Pupil Diameter', color='blue')

# === 2. Blinks (high-confidence only) ===
df_blinks_high_conf = df_blinks[df_blinks['confidence'] > 0.8]
onsets = df_blinks_high_conf[df_blinks_high_conf['type'] == 'onset']
offsets = df_blinks_high_conf[df_blinks_high_conf['type'] == 'offset']

for t in onsets['timestamp']:
    plt.axvline(x=t, color='red', linestyle='--', alpha=0.6, label='Blink Onset' if 'Blink Onset' not in plt.gca().get_legend_handles_labels()[1] else "")
#for t in offsets['timestamp']:
#    plt.axvline(x=t, color='green', linestyle='--', alpha=0.6, label='Blink Offset' if 'Blink Offset' not in plt.gca().get_legend_handles_labels()[1] else "")

# === 3. Annotations ===
for _, row in df_annotations.iterrows():
    plt.axvline(x=row['timestamp'], color='orange', linestyle=':', alpha=0.8, label=row['label'] if row['label'] not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(row['timestamp'], plt.ylim()[1]*0.95, row['label'], rotation=90, verticalalignment='top', color='orange', fontsize=8)

# === 4. Saccades (horizontal movement spikes) ===
# Example: assume saccades is a list of timestamps
for t in saccade_times:
    plt.axvline(x=t, color='green', linestyle='-.', alpha=0.6, label='Saccade' if 'Saccade' not in plt.gca().get_legend_handles_labels()[1] else "")

# === Final touches ===
plt.xlabel("Time (s)")
plt.ylabel("Pupil Diameter (px)")
plt.title("Pupil Data with Blinks, Saccades, and Annotations")
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show(block=True)


###### with velocity

plt.figure(figsize=(14, 6))

# === 1. Pupil Diameter ===
plt.plot(df_2d['timestamp'], df_2d['velocity'], label='Velocity', color='blue')

# === 2. Blinks (high-confidence only) ===
df_blinks_high_conf = df_blinks[df_blinks['confidence'] > 0.8]
onsets = df_blinks_high_conf[df_blinks_high_conf['type'] == 'onset']
offsets = df_blinks_high_conf[df_blinks_high_conf['type'] == 'offset']

for t in onsets['timestamp']:
    plt.axvline(x=t, color='red', linestyle='--', alpha=0.6, label='Blink Onset' if 'Blink Onset' not in plt.gca().get_legend_handles_labels()[1] else "")
#for t in offsets['timestamp']:
#    plt.axvline(x=t, color='green', linestyle='--', alpha=0.6, label='Blink Offset' if 'Blink Offset' not in plt.gca().get_legend_handles_labels()[1] else "")

# === 3. Annotations ===
for _, row in df_annotations.iterrows():
    plt.axvline(x=row['timestamp'], color='orange', linestyle=':', alpha=0.8, label=row['label'] if row['label'] not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.text(row['timestamp'], plt.ylim()[1]*0.95, row['label'], rotation=90, verticalalignment='top', color='orange', fontsize=8)

# === 4. Saccades (horizontal movement spikes) ===
# Example: assume saccades is a list of timestamps
for t in saccade_times:
    plt.axvline(x=t, color='green', linestyle='-.', alpha=0.6, label='Saccade' if 'Saccade' not in plt.gca().get_legend_handles_labels()[1] else "")

# === Final touches ===
plt.xlabel("Time (s)")
plt.ylabel("Pupil Diameter (px)")
plt.title("Pupil Data with Blinks, Saccades, and Annotations")
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show(block=True)


plt.hist(df_2d['velocity'], bins=100, range=[0,4])
plt.show(block=True)


###### Scatter plot of eye movement

plt.scatter(df_2d['x'], df_2d['y'], c=df_2d['timestamp'])
plt.colorbar().ax.set_ylabel('Timestamps')
plt.xlabel('norm_pos_x')
plt.ylabel('norm_pos_y')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show(block=True)