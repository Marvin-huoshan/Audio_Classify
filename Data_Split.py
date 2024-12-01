import os
import numpy as np
import librosa
import random
from sklearn.model_selection import train_test_split

# paths and parameters
data_dir = "Instance"  # Main directory
classes = ["Beetle", "Cicada", "Termite", "Cricket"]
instances_per_class = [30, 300, 1000]  # number of sample
output_features = {}  # Store feature
output_dir = "Processed_Features" # output directory path

# Set file paths for each category and extract features
for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    audio_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.wav')]

    # different number of sample
    for instance_count in instances_per_class:
        sampled_files = random.sample(audio_files, min(instance_count, len(audio_files)))  # sampling

        features = []

        # extract MFCC
        for file_path in sampled_files:
            y, sr = librosa.load(file_path, sr=None)  # audio file
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # 40 dim feature
            mfccs_mean = np.mean(mfccs, axis=1)  # average each dim
            features.append(mfccs_mean)

        features = np.array(features)

        # split training and testing 2 / 8
        X_train, X_test = train_test_split(features, test_size=0.2, random_state=42)

        # store the results as dic in output_path
        output_path = os.path.join(output_dir, f"{class_name}_{instance_count}_instances.npz")
        np.savez(output_path, X_train=X_train, X_test=X_test)
        print(f"Saved {output_path}")

# test output
for key, value in output_features.items():
    print(f"{key} - Training samples: {len(value['X_train'])}, Test samples: {len(value['X_test'])}")
