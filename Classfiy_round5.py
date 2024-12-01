import os
import numpy as np
import random
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Parameter settings
data_dir = "Instance"
classes = ["Beetle", "Cicada", "Termite", "Cricket"]
num_mfcc = 40
num_rounds = 5
num_folds = 10
num_repeats = 10  # 运行的总次数
aggregate_results = []


# CNN module
class CNN(nn.Module):
    def __init__(self, num_selected_features):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # 根据特征数量动态调整 Linear 层
        reduced_length = num_selected_features // 4  # 假设两次池化操作
        self.fc1 = nn.Linear(64 * reduced_length, 64)
        self.fc2 = nn.Linear(64, len(classes))

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
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
def evaluate_ML_models(X_train, y_train, X_test, y_test):
    results = {}
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "K-NN": KNeighborsClassifier(n_neighbors=5),
        "SVM (Linear Kernel)": SVC(kernel='linear', probability=True),
        "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = accuracy_score(y_test, y_pred)
    return results


# Evaluate CNN
def evaluate_cnn_model(X_train, y_train, X_test, y_test, model, num_epochs=10, batch_size=32):
    X_train_tensor = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test[:, np.newaxis, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    return correct / total


# Main rounds
all_results = []
top_k = 10
def select_top_k_features(X_train, y_train, k):
    '''
    select top-k important features
    :param X_train: training feature
    :param y_train: training label
    :param k: number of features
    :return:
    '''
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    selected_features = sorted_indices[:k]
    return selected_features, feature_importances


for repeat in range(num_repeats):
    print(f"Running Repeat {repeat + 1}/{num_repeats}")
    all_results = []

    for round_num in tqdm(range(num_rounds), desc=f"Rounds in Repeat {repeat + 1}"):
        test_folder = str(round_num + 1)
        train_folders = [str(i) for i in range(1, 6) if i != round_num + 1]

        X_train, y_train, X_test, y_test = [], [], [], []
        min_train_samples = float('inf')  # 初始化为无穷大
        min_test_samples = float('inf')

        # 确定每轮的最小训练和测试样本数
        for label, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)

            # min sample size of training
            train_count = sum(
                len([f for f in os.listdir(os.path.join(class_path, folder)) if f.endswith('.wav')])
                for folder in train_folders
            )
            if train_count < min_train_samples:
                min_train_samples = train_count

            # min sample size of testing
            test_folder_path = os.path.join(class_path, test_folder)
            test_count = len([f for f in os.listdir(test_folder_path) if f.endswith('.wav')])
            if test_count < min_test_samples:
                min_test_samples = test_count

        # 平衡训练和测试样本数
        for label, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)

            train_samples = []
            for folder in train_folders:
                folder_path = os.path.join(class_path, folder)
                audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
                train_samples.extend(audio_files)

            sampled_files = random.sample(train_samples, min_train_samples)
            for file in sampled_files:
                X_train.append(extract_mfcc(file, num_mfcc))
                y_train.append(label)

            test_folder_path = os.path.join(class_path, test_folder)
            audio_files = [os.path.join(test_folder_path, f) for f in os.listdir(test_folder_path) if f.endswith('.wav')]
            sampled_files = random.sample(audio_files, min_test_samples)
            for file in sampled_files:
                X_test.append(extract_mfcc(file, num_mfcc))
                y_test.append(label)

        X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

        # 特征选择: 保留 top-k 个重要特征
        selected_features, feature_importances = select_top_k_features(X_train, y_train, top_k)
        sum_feature_importance = sum([feature_importances[i] for i in selected_features])
        print('sum_feature_importance: ', sum_feature_importance)
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]

        # 测试传统 ML 模型
        ml_test_results = evaluate_ML_models(X_train_selected, y_train, X_test_selected, y_test)

        # 保存每轮结果
        all_results.append({
            "Round": round_num + 1,
            "Decision Tree Accuracy": ml_test_results["Decision Tree"],
            "Random Forest Accuracy": ml_test_results["Random Forest"],
            "K-NN Accuracy": ml_test_results["K-NN"],
            "SVM (Linear Kernel) Accuracy": ml_test_results["SVM (Linear Kernel)"],
            "SVM (RBF Kernel) Accuracy": ml_test_results["SVM (RBF Kernel)"],
            "XGBoost Accuracy": ml_test_results["XGBoost"],
            "Selected Features": selected_features.tolist(),
            "sum_feature_importance": sum_feature_importance
        })

    aggregate_results.append(all_results)

final_results = []
for round_num in range(num_rounds):
    aggregated_round = {
        "Round": round_num + 1,
        "Decision Tree Accuracy": np.mean([repeat[round_num]["Decision Tree Accuracy"] for repeat in aggregate_results]),
        "Random Forest Accuracy": np.mean([repeat[round_num]["Random Forest Accuracy"] for repeat in aggregate_results]),
        "K-NN Accuracy": np.mean([repeat[round_num]["K-NN Accuracy"] for repeat in aggregate_results]),
        "SVM (Linear Kernel) Accuracy": np.mean([repeat[round_num]["SVM (Linear Kernel) Accuracy"] for repeat in aggregate_results]),
        "SVM (RBF Kernel) Accuracy": np.mean([repeat[round_num]["SVM (RBF Kernel) Accuracy"] for repeat in aggregate_results]),
        "XGBoost Accuracy": np.mean([repeat[round_num]["XGBoost Accuracy"] for repeat in aggregate_results]),
        "Selected Features": aggregate_results[0][round_num]["Selected Features"],  # 所有轮次特征相同，只保存一次
        "sum_feature_importance": np.mean([repeat[round_num]["sum_feature_importance"] for repeat in aggregate_results])
    }
    final_results.append(aggregated_round)

# 保存和打印最终结果
np.save(f"{top_k}_averaged_results.npy", final_results)
print("Final Averaged Results:")
for result in final_results:
    print(result)
