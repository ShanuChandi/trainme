import stempeg
import os
import soundfile as sf


def extract_stems_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".stem.mp4"):
            track_path = os.path.join(input_folder, file_name)
            stems, rate = stempeg.read_stems(track_path)

            track_name = file_name.replace(".stem.mp4", "")
            track_output = os.path.join(output_folder, track_name)
            os.makedirs(track_output, exist_ok=True)

            sources = ["mixture", "drums", "bass", "other", "vocals"]
            for i, source in enumerate(sources):
                sf.write(os.path.join(track_output, f"{source}.wav"), stems[i], rate)
            print(f"Extracted: {file_name}")


# Example usage:
extract_stems_from_folder("data/train", "musdb18/train")
extract_stems_from_folder("data/test", "musdb18/test")
