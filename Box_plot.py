import matplotlib.pyplot as plt

# Cicada and Termite data
cicada_data = [4.371, 5.195, 0.581, 0.761, 0.608, 0.511, 0.705, 2.479, 1.96, 2.42, 2.509, 1.473, 1.623, 3.173, 2.6,
               0.192, 0.57, 0.832, 0.304, 0.272, 0.313, 0.28, 0.286, 0.301, 0.289, 0.307, 0.251, 0.266, 0.248, 0.277,
               0.272, 0.274, 0.286, 0.738, 0.266, 0.251, 0.218, 0.239, 0.269, 0.242, 0.254, 0.28, 0.289]

termite_data = [0.321, 0.172, 0.233, 0.195, 0.481, 0.323, 0.177, 0.192, 0.107, 0.215, 0.268, 0.195, 0.123, 0.086, 0.435,
                0.909, 0.693, 0.882, 0.882, 0.407, 1.037, 0.762, 0.635, 1.328, 0.473, 0.895, 0.849, 0.869, 0.742, 0.511,
                1.237, 1.053, 0.462, 0.618, 0.473, 0.484, 0.636, 0.636, 0.774, 1.887, 0.616, 0.586, 0.31, 1.356, 0.352,
                0.458, 1.25, 1.027, 0.75, 0.693, 1.253, 1.38, 0.903, 2.38, 0.529, 0.636]

cricket_data = []
beetle_data = []

data = [cicada_data, termite_data, cricket_data, beetle_data]

plt.figure(figsize=(10, 6))
box = plt.boxplot(data, labels=["Cicada", "Termite", "Cricket", "Beetle"], patch_artist=True)
plt.title("Box Plot of Insect Segment Lengths")
plt.ylabel("Segment Length (seconds)")
plt.grid(True)


for i, category_data in enumerate([cicada_data, termite_data, cricket_data, beetle_data]):
    if category_data:
        min_value = min(category_data)
        plt.text(i+1, min_value, f'{min_value:.2f}', ha='center', va='bottom', fontsize=10, color='red')

# Show the plot
plt.savefig('Seg_length.pdf')
