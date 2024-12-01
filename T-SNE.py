import os
import numpy as np
import librosa
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Parameters
data_dir = "Instance"
classes = ["Beetle", "Cicada", "Termite", "Cricket"]
num_mfcc = 40
perplexity = 30  # Adjust based on dataset size
n_iter = 1000

# Initialize data and labels
X = []
y = []

# Load data from all classes and folders
for label, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name)
    for folder in ["1", "2", "3", "4", "5"]:
        folder_path = os.path.join(class_path, folder)
        audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]

        for file_path in audio_files:
            try:
                y_signal, sr = librosa.load(file_path, sr=44100)
                mfcc = librosa.feature.mfcc(y=y_signal, sr=sr, n_mfcc=num_mfcc)
                mfcc_mean = np.mean(mfcc, axis=1)  # Average across time
                X.append(mfcc_mean)
                y.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Check the dataset size
print(f"Total samples: {len(X)}")
if perplexity >= len(X):
    perplexity = len(X) // 2  # Adjust perplexity if too large

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
X_tsne = tsne.fit_transform(X)

# Plot t-SNE results
plt.figure(figsize=(10, 8))
for label, class_name in enumerate(classes):
    plt.scatter(
        X_tsne[y == label, 0],
        X_tsne[y == label, 1],
        label=class_name,
        alpha=0.7
    )

plt.legend()
plt.title("t-SNE Visualization of MFCC Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid()
plt.show()
