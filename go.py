import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.keras import layers, Model

# --- TPU SETUP ---
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect TPU
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("Running on TPU.")
except:
    strategy = tf.distribute.get_strategy()
    print("Running on CPU/GPU.")

# --- CONFIG ---
SAMPLE_RATE = 44100
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 100
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 10


# --- WAVE-UNET MODEL ---
def build_waveunet(input_shape=(147443, 1)):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    skips = []

    # Encoder
    for filters in [24, 48, 96, 192, 384]:
        x = layers.Conv1D(filters, 15, strides=2, padding="same")(x)
        x = layers.LeakyReLU()(x)
        skips.append(x)

    # Bottleneck
    x = layers.Conv1D(768, 15, padding="same")(x)
    x = layers.LeakyReLU()(x)

    # Decoder
    for filters, skip in zip([384, 192, 96, 48, 24][::-1], reversed(skips)):
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(filters, 5, padding="same")(x)
        x = layers.LeakyReLU()(x)
        x = layers.Concatenate()([x, skip])

    outputs = layers.Conv1D(1, 1, activation="tanh")(x)
    return Model(inputs, outputs)


# --- LOSS FUNCTION ---
def sdr_loss(y_true, y_pred):
    noise = y_true - y_pred
    signal_power = tf.reduce_sum(tf.square(y_true), axis=[1, 2])
    noise_power = tf.reduce_sum(tf.square(noise), axis=[1, 2])
    sdr = (
        10
        * tf.math.log((signal_power + 1e-7) / (noise_power + 1e-7))
        / tf.math.log(10.0)
    )
    return -tf.reduce_mean(sdr)


# --- DATASET LOADER ---
def load_tfrecord_dataset(file_pattern, batch_size, shuffle=True):
    def parse_example(example_proto):
        features = {
            "mixture": tf.io.FixedLenFeature([], tf.string),
            "vocals": tf.io.FixedLenFeature([], tf.string),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        mixture = tf.io.parse_tensor(parsed_features["mixture"], out_type=tf.float32)
        vocals = tf.io.parse_tensor(parsed_features["vocals"], out_type=tf.float32)
        return mixture, vocals

    def preprocess(mixture, vocals):
        mixture = tf.reshape(mixture, [147443, 1])
        vocals = tf.reshape(vocals, [147443, 1])
        return mixture, vocals

    dataset = tf.data.TFRecordDataset(
        tf.io.gfile.glob(file_pattern), num_parallel_reads=tf.data.AUTOTUNE
    )
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset


train_dataset = load_tfrecord_dataset(
    "gs://your-bucket/train*.tfrecord", batch_size=BATCH_SIZE
)
val_dataset = load_tfrecord_dataset(
    "gs://your-bucket/val*.tfrecord", batch_size=BATCH_SIZE, shuffle=False
)

# --- TRAINING LOOP ---
with strategy.scope():
    model = build_waveunet()
    optimizer = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(mixture, vocals):
        with tf.GradientTape() as tape:
            predictions = model(mixture, training=True)
            loss = sdr_loss(vocals, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_step(mixture, vocals):
        predictions = model(mixture, training=False)
        loss = sdr_loss(vocals, predictions)
        return loss

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        start_time = time.time()

        total_train_loss = 0.0
        for step, (mixture, vocals) in enumerate(train_dataset.take(STEPS_PER_EPOCH)):
            loss = train_step(mixture, vocals)
            total_train_loss += loss

            if step % 10 == 0:
                print(f"Step {step}/{STEPS_PER_EPOCH} - Loss: {loss:.4f}")

        avg_train_loss = total_train_loss / STEPS_PER_EPOCH

        total_val_loss = 0.0
        for step, (mixture, vocals) in enumerate(val_dataset.take(VALIDATION_STEPS)):
            loss = val_step(mixture, vocals)
            total_val_loss += loss

        avg_val_loss = total_val_loss / VALIDATION_STEPS

        print(
            f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} - Time: {time.time() - start_time:.2f}s"
        )

        model.save_weights(f"gs://your-bucket/checkpoints/epoch_{epoch + 1}.ckpt")
