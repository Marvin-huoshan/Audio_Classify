import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"
import numpy as np
import random
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
import copy




# Parameter settings
data_dir = "Instance"
classes = ["Beetle", "Cicada", "Termite", "Cricket"]
num_mfcc = 40
num_rounds = 5
num_folds = 10
num_repeats = 10  # 运行的总次数
aggregate_results = []
# Augmentation parameters
pitch_shift_steps = [-2, -1, 1, 2]
speed_factors = [0.5, 2.0]

# Augmentation function
def augment_audio(y, sr, pitch_shift_steps, speed_factors):
    """
    Apply pitch shift and speed change to the audio.
    :param y: Audio time series.
    :param sr: Sampling rate.
    :param pitch_shift_steps: List of pitch shift steps.
    :param speed_factors: List of speed change factors.
    :return: List of augmented audio arrays.
    """
    augmented_audios = [y]  # Start with the original audio
    for steps in pitch_shift_steps:
        augmented_audios.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=steps))
    for factor in speed_factors:
        if factor != 1.0:  # Avoid redundant original speed
            augmented_audios.append(librosa.effects.time_stretch(y, rate=factor))
    return augmented_audios

# Updated extract_mfcc function with augmentation
def extract_mfcc_with_augmentation(file_path, num_mfcc, pitch_shift_steps, speed_factors):
    """
    Extract MFCCs for the original and augmented audio.
    :param file_path: Path to the audio file.
    :param num_mfcc: Number of MFCC coefficients.
    :param pitch_shift_steps: List of pitch shift steps.
    :param speed_factors: List of speed change factors.
    :return: List of MFCCs for the original and augmented audio.
    """
    y, sr = librosa.load(file_path, sr=44100)
    augmented_audios = augment_audio(y, sr, pitch_shift_steps, speed_factors)
    mfccs_list = []
    for audio in augmented_audios:
        n_fft = min(2048, len(audio) // 2)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, n_mels=40)
        mfccs_list.append(np.mean(mfccs, axis=1))
    return mfccs_list

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
    for model_name, model in tqdm(models.items(), desc='evaluate the model'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = accuracy_score(y_test, y_pred)
    return results


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
                augmented_mfccs = extract_mfcc_with_augmentation(file, num_mfcc, pitch_shift_steps, speed_factors)
                X_train.extend(augmented_mfccs)  # Append all augmented versions
                y_train.extend([label] * len(augmented_mfccs))  # Duplicate labels for augmented data

            test_folder_path = os.path.join(class_path, test_folder)
            audio_files = [os.path.join(test_folder_path, f) for f in os.listdir(test_folder_path) if
                           f.endswith('.wav')]
            sampled_files = random.sample(audio_files, min_test_samples)
            for file in sampled_files:
                X_test.append(extract_mfcc(file, num_mfcc))  # No augmentation for test
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

    aggregate_results.append(copy.deepcopy(all_results))

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
np.save(f"{top_k}_augment_results.npy", final_results)
print("Final Averaged Results:")
for result in final_results:
    print(result)
