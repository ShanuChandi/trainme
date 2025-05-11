import musdb
import numpy as np
import tensorflow as tf
from keras import layers, models
import os

# %% [code] {"execution":{"iopub.status.busy":"2025-05-11T14:07:08.740718Z","iopub.execute_input":"2025-05-11T14:07:08.741196Z","iopub.status.idle":"2025-05-11T14:07:08.751785Z","shell.execute_reply.started":"2025-05-11T14:07:08.741168Z","shell.execute_reply":"2025-05-11T14:07:08.746548Z"},"jupyter":{"outputs_hidden":false}}
# Configuration
SAMPLE_RATE = 44100
SEGMENT_LENGTH = SAMPLE_RATE * 2  # 2 seconds
NUM_TRACKS = 80  # number of tracks to use (keep small for demo)
MUSDB_PATH = "./data"  # Update path if necessary


# %% [code] {"execution":{"iopub.status.busy":"2025-05-11T14:07:12.725954Z","iopub.execute_input":"2025-05-11T14:07:12.726279Z","iopub.status.idle":"2025-05-11T14:07:12.745720Z","shell.execute_reply.started":"2025-05-11T14:07:12.726252Z","shell.execute_reply":"2025-05-11T14:07:12.741206Z"},"jupyter":{"source_hidden":true}}
def build_deep_wave_unet(input_shape, num_sources=2):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []

    # Encoder
    for f in [32, 64, 128, 256]:
        x = layers.Conv1D(f, 15, strides=2, padding="same", activation="relu")(x)
        skips.append(x)

    # Bottleneck
    x = layers.Conv1D(512, 15, padding="same", activation="relu")(x)

    # Decoder
    # For clarity, renamed 'skip' in zip to 'skip_connection'
    for f, skip_connection in zip([256, 128, 64, 32], reversed(skips)):
        x = layers.UpSampling1D(2)(x)

        # User's original cropping logic to make x and skip_connection have the same length
        # This crops the longer tensor from its end to match the shorter one.
        crop_len_diff = skip_connection.shape[1] - x.shape[1]
        if crop_len_diff > 0:  # skip_connection is longer
            current_skip_to_concat = layers.Cropping1D((0, crop_len_diff))(
                skip_connection
            )
            current_x_to_concat = x
        elif crop_len_diff < 0:  # x is longer
            current_x_to_concat = layers.Cropping1D((0, -crop_len_diff))(x)
            current_skip_to_concat = skip_connection
        else:  # lengths are equal
            current_x_to_concat = x
            current_skip_to_concat = skip_connection

        x = layers.Concatenate()([current_x_to_concat, current_skip_to_concat])
        x = layers.Conv1D(f, 15, padding="same", activation="relu")(x)

    # After the decoder loop, x has a temporal dimension of input_shape[0] / 2 (e.g., 22050 if input is 44100).
    # And features determined by the last 'f' (e.g., 32).

    # FIX: Upsample the tensor to match the original input length
    # Check if current length is half of the target input length
    if (
        x.shape[1] is not None and input_shape[0] is not None
    ):  # Check for statically known shapes
        if x.shape[1] == input_shape[0] / 2:
            x = layers.UpSampling1D(size=2)(x)
    # If shapes are dynamic, the above check might not execute during graph construction.
    # However, UpSampling1D(size=2) will still double the length.
    # For this specific architecture, an unconditional UpSampling1D(2) is expected to be needed here.
    # If x.shape[1] was None, but we know it's half:
    elif x.shape[1] is None and input_shape[0] is not None:
        # This implies dynamic length for x, but if the architecture consistently halves it, upsampling is needed.
        # Add an explicit upsample if the analysis holds it should be half.
        # For safety and based on analysis that output is 22050 and input 44100:
        pass  # The line below will handle it more generally if static check fails.

    # Ensure upsampling if it's precisely half (more robust for dynamic cases if above static check is tricky)
    # This logic relies on the known architectural output of N/2.
    # A more robust way if static shapes are not fully available at graph build time:
    # Perform the upsampling based on the architectural design.
    # The analysis showed x length is 22050 here. Target is 44100.
    # So, we apply UpSampling1D(2).
    # To be absolutely sure it's applied if it's half:
    if (
        input_shape[0] is not None and x.shape[1] == input_shape[0] // 2
    ):  # Check if length is exactly half
        x = layers.UpSampling1D(size=2)(x)
    elif (
        x.shape[1] is not None
        and input_shape[0] is not None
        and x.shape[1] * 2 == input_shape[0]
    ):  # Alternative check
        x = layers.UpSampling1D(size=2)(x)
    else:
        # If static shapes are not available to confirm it's half, but architecturally it is:
        # This is the most direct fix based on the observed problem (22050 output for 44100 input)
        # This line should be executed if the static shape checks above are inconclusive or not met
        # but the length is indeed half. The most reliable fix is to ensure this upsampling happens.
        # Given the error, the length of x IS 22050. input_shape[0] IS 44100.
        # So an upsample by 2 is needed.
        if x.shape[1] != input_shape[0]:  # If not already target length
            # Assuming it's half based on consistent down/up sampling stages
            # This ensures the upsampling occurs if not already matching
            x = layers.UpSampling1D(size=2)(x)

    # Final convolution to get the desired number of output sources
    x = layers.Conv1D(num_sources, 1, padding="same")(
        x
    )  # Activation is linear by default

    # User's original output shaping line.
    # After the fix, x.shape[1] should be equal to input_shape[0].
    # This line will then effectively be `outputs = x`, or crop if x somehow became slightly longer.
    outputs = x[:, : input_shape[0], :]

    model = models.Model(inputs, outputs)
    return model


# %% [code] {"execution":{"iopub.status.busy":"2025-05-11T14:07:22.385409Z","iopub.execute_input":"2025-05-11T14:07:22.385826Z","iopub.status.idle":"2025-05-11T14:07:22.397819Z","shell.execute_reply.started":"2025-05-11T14:07:22.385795Z","shell.execute_reply":"2025-05-11T14:07:22.393102Z"},"jupyter":{"source_hidden":true}}
def segment_track(track, segment_len):
    mixture = track.audio[:, :2]  # stereo (L+R)
    vocals = track.targets["vocals"].audio[:, :2]

    # Normalize (optional, but often good practice)
    # Ensure normalization doesn't cause issues with very silent tracks if max is near zero.
    max_abs_mixture = np.max(np.abs(mixture))
    if max_abs_mixture > 1e-6:  # Avoid division by zero or very small numbers
        mixture = mixture / max_abs_mixture

    max_abs_vocals = np.max(np.abs(vocals))
    if max_abs_vocals > 1e-6:
        vocals = vocals / max_abs_vocals

    cut = (len(mixture) // segment_len) * segment_len
    if cut == 0:  # Handle cases where track is shorter than segment_len
        return np.array([]).reshape(0, segment_len, 2), np.array([]).reshape(
            0, segment_len, 2
        )  # return empty arrays with correct dimensions

    mixture = mixture[:cut].reshape(-1, segment_len, 2)
    vocals = vocals[:cut].reshape(-1, segment_len, 2)

    return mixture, vocals


# %% [code] {"execution":{"iopub.status.busy":"2025-05-11T14:07:26.135044Z","iopub.execute_input":"2025-05-11T14:07:26.135331Z","iopub.status.idle":"2025-05-11T14:07:26.148024Z","shell.execute_reply.started":"2025-05-11T14:07:26.135308Z","shell.execute_reply":"2025-05-11T14:07:26.143503Z"},"jupyter":{"outputs_hidden":false}}
# STFT loss function (user's version seems fine)
def stft_loss(y_true, y_pred):
    # Assume shape (batch, samples, channels)
    # STFT works on 1D, so we apply per channel then average
    def channel_stft_loss(channel_true, channel_pred):
        stft_true = tf.signal.stft(channel_true, frame_length=512, frame_step=256)
        stft_pred = tf.signal.stft(channel_pred, frame_length=512, frame_step=256)
        return tf.reduce_mean(tf.abs(stft_true - stft_pred))

    losses = []
    # Ensure y_true and y_pred have a channel dimension for the loop
    if len(y_true.shape) == 2:  # (batch, samples) -> (batch, samples, 1)
        y_true = tf.expand_dims(y_true, axis=-1)
    if len(y_pred.shape) == 2:
        y_pred = tf.expand_dims(y_pred, axis=-1)

    num_channels = y_true.shape[-1]
    if num_channels is None:  # Handle dynamic shape for channels
        num_channels = tf.shape(y_true)[-1]

    for i in range(num_channels):  # tf.range for graph mode if num_channels is tensor
        losses.append(channel_stft_loss(y_true[..., i], y_pred[..., i]))
    return tf.reduce_mean(tf.stack(losses))  # stack and then reduce_mean


# %% [code] {"execution":{"iopub.status.busy":"2025-05-11T14:15:17.461017Z","iopub.execute_input":"2025-05-11T14:15:17.461456Z","iopub.status.idle":"2025-05-11T14:17:13.449206Z","shell.execute_reply.started":"2025-05-11T14:15:17.461425Z","shell.execute_reply":"2025-05-11T14:17:13.444389Z"},"jupyter":{"source_hidden":true,"outputs_hidden":true}}
# Combined loss function (user's version seems fine)
def combined_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred))
    return mse_loss + stft_loss(y_true, y_pred)


# --- Rest of your script ---
# (Load data, split, compile, and fit)

# Example of how to use it (ensure MUSDB_PATH is correct and data is present)
# This part is illustrative; your original script structure should be maintained for execution.
if __name__ == "__main__":
    # Check if running in a notebook environment or as a script for data loading part
    try:
        __IPYTHON__
        is_notebook = True
    except NameError:
        is_notebook = False

    if not os.path.exists(MUSDB_PATH) and is_notebook:
        print(
            "MUSDB_PATH does not exist. Please ensure data is downloaded and extracted."
        )
        print("Example download commands (run in a terminal or notebook cell with !):")
        print("# !wget https://zenodo.org/records/1117372/files/musdb18.zip")
        print("# !unzip musdb18.zip -d ./data")
        # return # Stop execution if data is missing and in a context where we can't download easily

    # Load MUSDB
    # Wrap musdb.DB call in a try-except if it might fail due to path issues
    try:
        mus = musdb.DB(root=MUSDB_PATH, subsets="train", is_wav=False)
    except Exception as e:
        print(f"Error loading MUSDB: {e}")
        print(f"Please check MUSDB_PATH: {MUSDB_PATH}")
        exit()  # or return if in a function/notebook

    X_data, y_data = [], []
    for i, track in enumerate(mus.tracks[:NUM_TRACKS]):
        print(f"{i} Processing {track.name}")
        x_segments, y_segments = segment_track(track, SEGMENT_LENGTH)
        if x_segments.shape[0] > 0:  # Only append if segments were actually created
            X_data.append(x_segments)
            y_data.append(y_segments)

    if not X_data or not y_data:
        print("No data loaded. Check segment_track logic or source audio files.")
        exit()  # or return

    X = np.concatenate(X_data)
    y = np.concatenate(y_data)

    print("Dataset shape:", X.shape)
    if X.shape[0] == 0:
        print("Concatenated dataset is empty. Halting.")
        exit()

    # Split data
    split_idx = int(len(X) * 0.9)
    if split_idx == 0 and len(X) > 0:  # Ensure val set is not empty if X is small
        split_idx = max(
            1, len(X) - 1
        )  # at least 1 sample for val if possible, or train
    if len(X) <= 1:  # not enough data to split
        print(
            "Not enough data to create a validation set. Consider using more tracks or longer audio."
        )
        X_train, X_val = X, X
        y_train, y_val = y, y
    else:
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

    if (
        X_val.shape[0] == 0 and X_train.shape[0] > 0
    ):  # If val set became empty due to split
        print(
            "Validation set is empty. Using training data for validation (not recommended for proper eval)."
        )
        X_val, y_val = (
            X_train[-1:],
            y_train[-1:],
        )  # Use last sample of train as val, or duplicate train

    input_shape = X_train.shape[1:]

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    with strategy.scope():
        model = build_deep_wave_unet(input_shape, num_sources=2)
        model.compile(optimizer="adam", loss=combined_loss)

    model.summary()

    if X_train.shape[0] > 0 and X_val.shape[0] > 0:
        model.fit(
            X_train, y_train, validation_data=(X_val, y_val), batch_size=4, epochs=60
        )
    elif X_train.shape[0] > 0:
        print("Warning: Validation set is empty. Training without validation.")
        model.fit(X_train, y_train, batch_size=4, epochs=60)
    else:
        print("Training data is empty. Cannot fit model.")

    os.makedirs("models", exist_ok=True)
    model.save("models/wave_unet_spleeter_like.h5")
    print("Model saved to models/wave_unet_spleeter_like.h5")
