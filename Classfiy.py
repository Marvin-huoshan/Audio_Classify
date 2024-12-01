import os
import numpy as np
import random
import librosa
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Parameter settings
data_dir = "Instance"  # Main directory
classes = ["Beetle", "Cicada", "Termite", "Cricket"]
instance_counts = [30, 300, 1000]  # number of sample
num_mfcc = 40  # MFCC feature dim
m = 10  # number of rounds


# CNN module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 10, 64)
        self.fc2 = nn.Linear(64, len(classes))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 10)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Extract MFCC
def extract_mfcc(file_path, num_mfcc):
    y, sr = librosa.load(file_path, sr=44100)
    n_fft = min(2048, len(y) // 2)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, n_mels=40)
    return np.mean(mfccs, axis=1)


# Evaluate ML models
def evaluate_ML_models(X_train, X_test, y_train, y_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))

    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

    return dt_acc, rf_acc, knn_acc


# Evaluate CNN
def evaluate_cnn_model(X_train, X_test, y_train, y_test, num_epochs=10, batch_size=32):
    X_train_tensor = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test[:, np.newaxis, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    cnn_acc = correct / total
    return cnn_acc


# Main rounds
for instance_count in instance_counts:
    dt_accs, rf_accs, knn_accs, cnn_accs = [], [], [], []

    for i in tqdm(range(m), desc='Evaluating {} instances'.format(instance_count)):
        X, y = [], []
        for label, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)
            audio_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.wav')]
            sampled_files = random.sample(audio_files, min(instance_count, len(audio_files)))

            for file_path in sampled_files:
                mfcc = extract_mfcc(file_path, num_mfcc)
                X.append(mfcc)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        # split training set and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # evaluate and log the results
        dt_acc, rf_acc, knn_acc = evaluate_ML_models(X_train, X_test, y_train, y_test)
        cnn_acc = evaluate_cnn_model(X_train, X_test, y_train, y_test)

        dt_accs.append(dt_acc)
        rf_accs.append(rf_acc)
        knn_accs.append(knn_acc)
        cnn_accs.append(cnn_acc)

    print(f"Instance Count: {instance_count}")
    print(f" - Decision Tree Average Accuracy: {np.mean(dt_accs):.2f}")
    print(f" - Random Forest Average Accuracy: {np.mean(rf_accs):.2f}")
    print(f" - K-NN Average Accuracy: {np.mean(knn_accs):.2f}")
    print(f" - CNN Average Accuracy: {np.mean(cnn_accs):.2f}")
