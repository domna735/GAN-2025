#!/usr/bin/env python3
"""
Normalize intensity (loudness) of generated audio files to match target dB level.
This fixes the issue where Griffin-Lim generated audio is much quieter than real recordings.
"""
import argparse
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


def compute_rms_db(y, frame_length=160, hop_length=80):
    """Compute RMS intensity in dB."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    return np.mean(rms_db)


def normalize_to_target_db(y, current_db, target_db):
    """
    Normalize audio to target dB level.
    
    Args:
        y: Audio signal (numpy array)
        current_db: Current RMS dB level
        target_db: Target RMS dB level
        
    Returns:
        Normalized audio signal
    """
    db_diff = target_db - current_db
    # Convert dB difference to linear gain
    gain = 10 ** (db_diff / 20.0)
    y_normalized = y * gain
    
    # Prevent clipping
    max_val = np.abs(y_normalized).max()
    if max_val > 0.99:
        y_normalized = y_normalized * (0.99 / max_val)
    
    return y_normalized


def main():
    parser = argparse.ArgumentParser(description='Normalize audio intensity to target dB level')
    parser.add_argument('--input-dir', required=True, help='Input directory with WAV files')
    parser.add_argument('--target-db', type=float, required=True, 
                        help='Target RMS dB level (e.g., -37.70)')
    parser.add_argument('--out', required=True, help='Output directory for normalized WAV files')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate (default: 16000)')
    parser.add_argument('--ext', default='.wav', help='Audio file extension (default: .wav)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = list(input_dir.rglob(f'*{args.ext}'))
    
    if not audio_files:
        print(f"No {args.ext} files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} files to normalize")
    print(f"Target dB: {args.target_db}")
    
    normalized_count = 0
    total_gain_db = 0.0
    
    for audio_file in audio_files:
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=args.sr, mono=True)
            
            # Compute current RMS dB
            current_db = compute_rms_db(y)
            
            # Normalize
            y_normalized = normalize_to_target_db(y, current_db, args.target_db)
            
            # Verify final dB
            final_db = compute_rms_db(y_normalized)
            gain_db = final_db - current_db
            total_gain_db += gain_db
            
            # Save normalized audio
            relative_path = audio_file.relative_to(input_dir)
            out_path = out_dir / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(out_path, y_normalized, sr)
            normalized_count += 1
            
            if normalized_count % 10 == 0:
                print(f"  Processed {normalized_count}/{len(audio_files)}: "
                      f"{audio_file.name} ({current_db:.2f}dB → {final_db:.2f}dB, gain={gain_db:.2f}dB)")
        
        except Exception as e:
            print(f"  Error processing {audio_file.name}: {e}")
            continue
    
    avg_gain_db = total_gain_db / normalized_count if normalized_count > 0 else 0
    print(f"\nNormalized {normalized_count} files")
    print(f"Average gain applied: {avg_gain_db:.2f} dB")
    print(f"Output directory: {out_dir}")


if __name__ == '__main__':
    main()
