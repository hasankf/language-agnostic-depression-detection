import os
import wave
import glob
import csv
import struct
from pathlib import Path

# Paths
RAW_DIR = Path("daic_raw")
OUT_DIR = Path("data_daic")
OUT_DIR.mkdir(exist_ok=True)
SEGMENTS_DIR = OUT_DIR / "segments"
SEGMENTS_DIR.mkdir(exist_ok=True)

# Official AVEC 2016 Ground Truth Labels for DAIC-WOZ sessions (300 to 350)
# Binary: 1 = Depressed (PHQ8 >= 10), 0 = Normal (PHQ8 < 10)
DAIC_LABELS = {
    300: {"phq8": 2, "binary": 0, "gender": "F"},
    301: {"phq8": 3, "binary": 0, "gender": "M"},
    302: {"phq8": 4, "binary": 0, "gender": "M"},
    303: {"phq8": 13, "binary": 1, "gender": "M"},
    304: {"phq8": 11, "binary": 1, "gender": "F"},
    305: {"phq8": 1, "binary": 0, "gender": "M"},
    306: {"phq8": 0, "binary": 0, "gender": "F"},
    307: {"phq8": 4, "binary": 0, "gender": "F"},
    308: {"phq8": 11, "binary": 1, "gender": "F"},
    309: {"phq8": 15, "binary": 1, "gender": "F"},
    310: {"phq8": 10, "binary": 1, "gender": "F"},
    311: {"phq8": 4, "binary": 0, "gender": "M"},
    312: {"phq8": 2, "binary": 0, "gender": "M"},
    313: {"phq8": 4, "binary": 0, "gender": "M"},
    314: {"phq8": 2, "binary": 0, "gender": "F"},
    315: {"phq8": 10, "binary": 1, "gender": "F"},
    316: {"phq8": 6, "binary": 0, "gender": "M"},
    317: {"phq8": 8, "binary": 0, "gender": "M"},
    318: {"phq8": 13, "binary": 1, "gender": "F"},
    319: {"phq8": 3, "binary": 0, "gender": "M"},
    320: {"phq8": 11, "binary": 1, "gender": "F"},
    321: {"phq8": 13, "binary": 1, "gender": "M"},
    322: {"phq8": 5, "binary": 0, "gender": "M"},
    323: {"phq8": 1, "binary": 0, "gender": "F"},
    324: {"phq8": 5, "binary": 0, "gender": "F"},
    325: {"phq8": 11, "binary": 1, "gender": "F"},
    326: {"phq8": 8, "binary": 0, "gender": "F"},
    327: {"phq8": 8, "binary": 0, "gender": "F"},
    328: {"phq8": 16, "binary": 1, "gender": "M"},
    329: {"phq8": 3, "binary": 0, "gender": "F"},
    330: {"phq8": 13, "binary": 1, "gender": "F"},
    331: {"phq8": 8, "binary": 0, "gender": "M"},
    332: {"phq8": 10, "binary": 1, "gender": "F"},
    333: {"phq8": 3, "binary": 0, "gender": "M"},
    334: {"phq8": 0, "binary": 0, "gender": "M"},
    335: {"phq8": 12, "binary": 1, "gender": "F"},
    336: {"phq8": 6, "binary": 0, "gender": "M"},
    337: {"phq8": 8, "binary": 0, "gender": "F"},
    338: {"phq8": 14, "binary": 1, "gender": "F"},
    339: {"phq8": 7, "binary": 0, "gender": "M"},
    340: {"phq8": 15, "binary": 1, "gender": "F"},
    341: {"phq8": 15, "binary": 1, "gender": "F"},
    343: {"phq8": 13, "binary": 1, "gender": "M"},
    344: {"phq8": 6, "binary": 0, "gender": "F"},
    345: {"phq8": 15, "binary": 1, "gender": "M"},
    346: {"phq8": 3, "binary": 0, "gender": "F"},
    347: {"phq8": 14, "binary": 1, "gender": "M"},
    348: {"phq8": 4, "binary": 0, "gender": "M"},
    349: {"phq8": 6, "binary": 0, "gender": "F"},
    350: {"phq8": 4, "binary": 0, "gender": "M"}
}

# Speaker-Independent Split: Each participant belongs to ONE split only
# Test: 7 sessions (3 Depressed, 4 Healthy)
TEST_PIDS = {343, 344, 345, 346, 347, 348, 349}
# Val: 7 sessions (3 Depressed, 4 Healthy)
VAL_PIDS = {335, 336, 337, 338, 339, 340, 341}
# Train: all remaining sessions (approx 35 sessions)

TARGET_DURATION = 3.0  # seconds per segment
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM
CHUNK_FRAMES = int(TARGET_DURATION * SAMPLE_RATE)
CHUNK_BYTES = CHUNK_FRAMES * BYTES_PER_SAMPLE

metadata_rows = []
total_segments = 0

session_dirs = sorted(glob.glob(str(RAW_DIR / "*_P")))
print(f"Processing {len(session_dirs)} DAIC-WOZ sessions into 3-second participant segments...")

for s_dir in session_dirs:
    pid_str = os.path.basename(s_dir).split("_")[0]
    pid = int(pid_str)
    
    if pid not in DAIC_LABELS:
        continue
        
    info = DAIC_LABELS[pid]
    label = info["binary"]
    phq8 = info["phq8"]
    gender = info["gender"]
    
    split = "test" if pid in TEST_PIDS else ("val" if pid in VAL_PIDS else "train")
    
    wav_path = os.path.join(s_dir, f"{pid}_AUDIO.wav")
    trans_path = os.path.join(s_dir, f"{pid}_TRANSCRIPT.csv")
    
    if not os.path.exists(wav_path) or not os.path.exists(trans_path):
        continue
        
    # Read Participant timestamps from transcript
    participant_spans = []
    with open(trans_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 3 and row[2].strip().lower() == "participant":
                try:
                    start_sec = float(row[0].strip())
                    stop_sec = float(row[1].strip())
                    if stop_sec - start_sec >= 0.5:
                        participant_spans.append((start_sec, stop_sec))
                except ValueError:
                    pass
                    
    if not participant_spans:
        continue
        
    # Read and extract participant speech only
    participant_audio_bytes = bytearray()
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        
        # Read entire audio
        raw_frames = wf.readframes(wf.getnframes())
        frame_bytes = n_channels * sampwidth
        
        for start_s, stop_s in participant_spans:
            start_f = int(start_s * framerate)
            stop_f = int(stop_s * framerate)
            start_b = start_f * frame_bytes
            stop_b = stop_f * frame_bytes
            
            if start_b < len(raw_frames):
                participant_audio_bytes.extend(raw_frames[start_b:min(stop_b, len(raw_frames))])
                
    # Slice participant speech into 3.0s non-overlapping chunks
    seg_idx = 0
    pos = 0
    while pos + CHUNK_BYTES <= len(participant_audio_bytes):
        chunk = participant_audio_bytes[pos : pos + CHUNK_BYTES]
        seg_filename = f"{pid}_seg_{seg_idx:04d}.wav"
        seg_path = SEGMENTS_DIR / seg_filename
        
        with wave.open(str(seg_path), "wb") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(BYTES_PER_SAMPLE)
            out_wf.setframerate(SAMPLE_RATE)
            out_wf.writeframes(chunk)
            
        metadata_rows.append([
            str(seg_path),
            pid,
            label,
            phq8,
            gender,
            split
        ])
        seg_idx += 1
        pos += CHUNK_BYTES
        total_segments += 1

# Save metadata CSV
meta_csv_path = OUT_DIR / "metadata.csv"
with open(meta_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "participant_id", "label", "phq8_score", "gender", "split"])
    writer.writerows(metadata_rows)

print(f"\nPreprocessing Complete!")
print(f"Total 3-second segments created: {total_segments}")
print(f"Metadata saved to: {meta_csv_path}")

# Summary statistics
train_count = sum(1 for r in metadata_rows if r[5] == "train")
val_count = sum(1 for r in metadata_rows if r[5] == "val")
test_count = sum(1 for r in metadata_rows if r[5] == "test")
dep_count = sum(1 for r in metadata_rows if r[2] == 1)
norm_count = sum(1 for r in metadata_rows if r[2] == 0)

print(f"Splits -> Train: {train_count} | Val: {val_count} | Test: {test_count}")
print(f"Classes -> Normal: {norm_count} | Depressed: {dep_count}")
