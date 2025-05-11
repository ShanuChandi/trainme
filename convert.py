import os
import librosa
import tensorflow as tf
import numpy as np

SAMPLE_RATE = 44100
CHUNK_DURATION = 6  # seconds
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION


def write_tfrecord(data_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)

    for track_name in os.listdir(data_dir):
        print(f"Processing {track_name}...")
        track_path = os.path.join(data_dir, track_name)
        mix_path = os.path.join(track_path, "mixture.wav")
        vocals_path = os.path.join(track_path, "vocals.wav")

        if not os.path.exists(mix_path) or not os.path.exists(vocals_path):
            continue

        mix, _ = librosa.load(mix_path, sr=SAMPLE_RATE, mono=True)
        vocals, _ = librosa.load(vocals_path, sr=SAMPLE_RATE, mono=True)

        length = min(len(mix), len(vocals))
        mix = mix[:length]
        vocals = vocals[:length]

        for i in range(0, length - CHUNK_SAMPLES, CHUNK_SAMPLES):

            mix_chunk = mix[i : i + CHUNK_SAMPLES]
            vocals_chunk = vocals[i : i + CHUNK_SAMPLES]

            feature = {
                "mix": tf.train.Feature(float_list=tf.train.FloatList(value=mix_chunk)),
                "vocals": tf.train.Feature(
                    float_list=tf.train.FloatList(value=vocals_chunk)
                ),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    writer.close()


write_tfrecord("musdb18/train", "train.tfrecord")
