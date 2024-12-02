##	Audio Classification with Augmentation
This repository contains the implementation of an audio classification pipeline that utilizes MFCC features, data augmentation, and various machine learning models (e.g., Decision Tree, Random Forest, K-NN, SVM, XGBoost) to classify audio signals into predefined classes.

##   Features
1. Audio Preprocessing: Extract MFCC features from audio files.
2. Data Augmentation: Apply pitch shift and speed changes to enhance training data diversity.
3. Feature Selection: Use tree-based methods to select the top-$k$ features.
4. Model Evaluation: Evaluate multiple classification models on balanced datasets.
5. Repeatable Experiments: Support for multiple rounds and repetitions for robust evaluation.


## Requirements
Below are the major Python libraries and their versions used in this project:

Python >= 3.11
numpy >= 1.26.4
scikit-learn >= 1.3.2
librosa >= 0.10.2
xgboost >= 2.1.3

##	Dataset Sturcutre

Instance/
├── Beetle/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
├── Cicada/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
├── Termite/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
├── Cricket/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/


Use the main script to start training and evaluation:
python Classfiy_round5_augementation.py


Results: The script will output evaluation metrics (e.g., accuracy, standard deviation) for each model and each value of $k$ (number of selected features). The results will also be saved in .npy files for further analysis.
