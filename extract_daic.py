import os
import shutil
import zipfile
from pathlib import Path

# Paths
DOWNLOADS_DIR = Path.home() / "Downloads"
PROJECT_DIR = Path(__file__).resolve().parent
DEST_DIR = PROJECT_DIR / "daic_raw"
DEST_DIR.mkdir(exist_ok=True)

print("=" * 60)
print(f"Scanning for DAIC-WOZ data in: {DOWNLOADS_DIR}")
print(f"Saving clean audio + transcripts to: {DEST_DIR}")
print("=" * 60)

found_sessions = set()

# --- 1. Scan for ALREADY EXTRACTED FOLDERS (e.g. 300_P, 301_P, etc.) ---
# Look in ~/Downloads and any immediate subfolder (like ~/Downloads/dataset-depression)
potential_folders = list(DOWNLOADS_DIR.glob("*_P*")) + list(DOWNLOADS_DIR.glob("*/*_P*"))

for folder in potential_folders:
    if folder.is_dir():
        session_id = folder.name
        audio_files = list(folder.glob("*_AUDIO.wav"))
        transcript_files = list(folder.glob("*_TRANSCRIPT.csv"))
        
        if audio_files and transcript_files:
            target_session_dir = DEST_DIR / session_id
            target_session_dir.mkdir(exist_ok=True)
            
            # Copy AUDIO and TRANSCRIPT only (skips all 500MB CLNF files!)
            shutil.copy2(audio_files[0], target_session_dir / audio_files[0].name)
            shutil.copy2(transcript_files[0], target_session_dir / transcript_files[0].name)
            
            found_sessions.add(session_id)
            print(f"✅ Found & Copied Folder: {session_id}")

# --- 2. Scan for REMAINING .ZIP FILES ---
potential_zips = list(DOWNLOADS_DIR.glob("*.zip")) + list(DOWNLOADS_DIR.glob("*/*.zip"))

for zip_path in potential_zips:
    session_id = zip_path.stem
    if session_id in found_sessions:
        continue
        
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            namelist = zf.namelist()
            targets = [f for f in namelist if f.endswith("_AUDIO.wav") or f.endswith("_TRANSCRIPT.csv")]
            
            if targets:
                target_session_dir = DEST_DIR / session_id
                target_session_dir.mkdir(exist_ok=True)
                
                for t in targets:
                    filename = os.path.basename(t)
                    with open(target_session_dir / filename, "wb") as f_out:
                        f_out.write(zf.read(t))
                        
                found_sessions.add(session_id)
                print(f"✅ Found & Unzipped: {session_id}")
    except Exception:
        # Skip broken or incomplete downloads
        pass

print("=" * 60)
print(f"🎉 Complete! Successfully gathered {len(found_sessions)} participant sessions into:")
print(f"{DEST_DIR}")
print("=" * 60)
