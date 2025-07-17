import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import os

# --------------------- EEG Preprocessing ---------------------
def preprocess_eeg_data(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)

    data = np.transpose(data, (0, 2, 1))
    data = data[:, :21, :]
    class_0_data = data[labels == 0]
    class_1_data = data[labels == 1]
    class_0_labels = np.zeros(len(class_0_data))
    class_1_labels = np.ones(len(class_1_data))

    data = np.concatenate([class_0_data, class_1_data], axis=0)
    labels = np.concatenate([class_0_labels, class_1_labels], axis=0)

    indices = np.arange(len(data))
    np.random.shuffle(indices)
    return data[indices], labels[indices]

# --------------------- STFT ---------------------
def compute_spectrograms(data, window_size, overlap, sampling_rate, n_channels):
    spectrograms = []
    for trial in data:
        ch_spects = []
        for ch in range(n_channels):
            _, _, Sxx = signal.spectrogram(trial[ch], fs=sampling_rate,
                                           window='hann', nperseg=window_size,
                                           noverlap=overlap)
            ch_spects.append(Sxx)
        spectrograms.append(np.stack(ch_spects, axis=-1))
    spectrograms = np.array(spectrograms)
    return (spectrograms - np.mean(spectrograms)) / np.std(spectrograms)

# --------------------- Data Split ---------------------
def split_and_prepare_data(spectrograms, labels, num_classes, test_size):
    X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels,
                                                        test_size=test_size, random_state=42)
    X_train = np.expand_dims(np.transpose(X_train, (0, 3, 1, 2)), axis=-1)
    X_test = np.expand_dims(np.transpose(X_test, (0, 3, 1, 2)), axis=-1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    return X_train, X_test, y_train, y_test

# --------------------- Self-Attention ---------------------
class SelfAttention(Model):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.W_Q = layers.Dense(dim)
        self.W_K = layers.Dense(dim)
        self.W_V = layers.Dense(dim)
        self.W_O = layers.Dense(dim)

    def call(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        Q = tf.reshape(self.W_Q(x), (B, T, self.num_heads, self.head_dim))
        K = tf.reshape(self.W_K(x), (B, T, self.num_heads, self.head_dim))
        V = tf.reshape(self.W_V(x), (B, T, self.num_heads, self.head_dim))
        Q, K, V = map(lambda t: tf.transpose(t, [0, 2, 1, 3]), [Q, K, V])
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(weights, V)
        attended = tf.transpose(attended, [0, 2, 1, 3])
        attended = tf.reshape(attended, (B, T, self.dim))
        return self.W_O(attended)

# --------------------- Custom 3D CNN ---------------------
class Custom3DCNNF4head(Model):
    def __init__(self, input_shape, num_classes=2, num_heads=4):
        super().__init__()
        C = 256
        self.stage1 = tf.keras.Sequential([
            layers.Conv3D(64, 3, padding='same', activation='relu'),
            layers.Conv3D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling3D(pool_size=(2, 2, 1))
        ])
        self.stage2 = tf.keras.Sequential([
            layers.Conv3D(128, 3, padding='same', activation='relu'),
            layers.Conv3D(128, 3, padding='same', activation='relu'),
            layers.MaxPooling3D(pool_size=(2, 2, 1))
        ])
        self.stage3 = tf.keras.Sequential([
            layers.Conv3D(256, 3, padding='same', activation='relu'),
            layers.Conv3D(256, 3, padding='same', activation='relu'),
            layers.MaxPooling3D(pool_size=(2, 2, 1))
        ])
        self.stage4 = tf.keras.Sequential([
            layers.Conv3D(512, 3, padding='same', activation='relu'),
            layers.Conv3D(512, 3, padding='same', activation='relu'),
            layers.MaxPooling3D(pool_size=(2, 2, 1))
        ])
        self.conv1x1 = [layers.Conv3D(C, 1, activation='relu') for _ in range(4)]
        self.gap = layers.GlobalAveragePooling3D()
        self.att = SelfAttention(dim=C, num_heads=num_heads)
        self.fc = layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        out1 = self.stage1(x)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)
        v = [self.gap(self.conv1x1[i](out)) for i, out in enumerate([out1, out2, out3, out4])]
        v = tf.stack(v, axis=1)
        v = self.att(v)
        return self.fc(tf.reduce_mean(v, axis=1))

# --------------------- Plotting ---------------------
def plot_metrics(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for metric in ['loss', 'accuracy']:
        plt.figure()
        plt.plot(history.history[metric], label=f"Train {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"Val {metric}")
        plt.title(f"{metric.capitalize()} Curve")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{metric}_curve.png"))
        plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(save_path)
    plt.close()

# --------------------- Main ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--model_path", type=str, default="./outputs/model.h5")
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--sampling_rate", type=int, default=256)
    parser.add_argument("--n_channels", type=int, default=21)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Pipeline
    data, labels = preprocess_eeg_data(args.data_path, args.labels_path)
    spectrograms = compute_spectrograms(data, args.window_size, args.overlap,
                                        args.sampling_rate, args.n_channels)
    X_train, X_test, y_train, y_test = split_and_prepare_data(spectrograms, labels,
                                                              args.num_classes, args.test_size)

    model = Custom3DCNNF4head(input_shape=X_train.shape[1:], num_classes=args.num_classes)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, batch_size=args.batch_size,
                        epochs=args.epochs, validation_data=(X_test, y_test))

    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save(args.model_path)

    # Metrics & confusion matrix
    plot_metrics(history, args.save_dir)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

    plot_confusion_matrix(y_true, y_pred, os.path.join(args.save_dir, "confusion_matrix.png"))
