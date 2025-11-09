import os
import librosa
import numpy as np

# Define paths to the directories containing audio files
input_dirs = [
    r"C:\Users\domna735\OneDrive\Desktop\Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration\Vietnamese\Cantonese",
    r"C:\Users\domna735\OneDrive\Desktop\Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration\Vietnamese\Thai",
    r"C:\Users\domna735\OneDrive\Desktop\Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration\Vietnamese\Vietnamese"
]

# Define the output directory for saving .npy files
output_dir = r"C:\Users\domna735\OneDrive\Desktop\Vowel_length_contrasts_in_deep_learning_Generative_Adversarial_Phonology_and_duration\processed_data"
os.makedirs(output_dir, exist_ok=True)

# Function to process audio files
def process_audio_files():
    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".mp3"):
                    file_path = os.path.join(root, file)
                    try:
                        # Load the audio file
                        y, sr = librosa.load(file_path, sr=None)

                        # Convert to Mel-spectrogram
                        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

                        # Convert to log scale (dB)
                        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

                        # Save the spectrogram as a .npy file
                        output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.npy")
                        np.save(output_file, log_mel_spectrogram)

                        print(f"Processed and saved: {output_file}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    process_audio_files()
