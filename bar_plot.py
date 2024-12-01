import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['axes.facecolor'] = '#e6f0ff'  # 浅蓝色背景
plt.rcParams['grid.color'] = 'lightgray'         # 蓝色网格线
plt.rcParams['grid.linestyle'] = '-'
# Load the results
results_10 = np.load('10_averaged_results.npy', allow_pickle=True)
results_20 = np.load('20_averaged_results.npy', allow_pickle=True)
results_30 = np.load('30_averaged_results.npy', allow_pickle=True)
results_40 = np.load('40_averaged_results.npy', allow_pickle=True)

# Combine all results into a dictionary for easier processing
k_values = [10, 20, 30, 40]
all_results = {10: results_10, 20: results_20, 30: results_30, 40: results_40}

# Models to evaluate
models = ["Decision Tree Accuracy", "Random Forest Accuracy", "K-NN Accuracy", "SVM (Linear Kernel) Accuracy", "SVM (RBF Kernel) Accuracy", "XGBoost Accuracy"]
model_name = ["Decision Tree", "Random Forest", "K-NN", "SVM", "RBF", "XGBoost"]

# Prepare data for plotting
mean_std_data = {model: {"mean": [], "std": [], "sum_feature_importance": []} for model in models}

for k, results in all_results.items():
    for model in models:
        accuracies = [round_data[model] for round_data in results]
        sum_feature_importance = [round_data['sum_feature_importance'] for round_data in results]
        mean_std_data[model]["mean"].append(np.mean(accuracies))
        mean_std_data[model]["std"].append(np.std(accuracies))
        mean_std_data[model]["sum_feature_importance"].append(np.mean(sum_feature_importance))

print(all_results)


for model in models:
    print('Model: {}'.format(model))
    print('mean:{}, std:{}, sum_feature_importance:{}'.format(mean_std_data[model]["mean"], mean_std_data[model]["std"], mean_std_data[model]["sum_feature_importance"]))

# Plot for each model
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust space between subplots

bar_width = 0.4
x_positions = np.arange(len(k_values))

for idx, model in enumerate(models):
    row, col = divmod(idx, 3)
    means = mean_std_data[model]["mean"]
    stds = mean_std_data[model]["std"]

    axes[row, col].bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,
        width=bar_width,
        color='skyblue',
        edgecolor='black',
    )
    axes[row, col].set_xticks(x_positions)
    axes[row, col].set_xticklabels(k_values, fontsize=16)
    axes[row, col].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axes[row, col].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=16)
    axes[row, col].set_title(model_name[idx], fontsize=16)
    axes[row, col].set_xlabel(r'$\mathcal{k}$', fontsize=20)

    # Set "Accuracy" label only for the leftmost subplots
    if col == 0:
        axes[row, col].set_ylabel("Accuracy", fontsize=16)

    axes[row, col].grid(axis='y')

plt.tight_layout()
plt.savefig('results.pdf')
plt.show()
