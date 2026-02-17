import matplotlib.pyplot as plt

#  actual data extracted from the console logs
data = {
    "Adult": {
        "MLP": {"train": [83.28, 84.47, 84.80, 84.89, 85.02], "test": 84.92},
        "CNN": {"train": [81.73, 84.23, 84.58, 84.80, 84.73], "test": 84.57},
        "Transformer": {"train": [82.78, 84.43, 84.66, 84.88, 84.93], "test": 84.98}
    },
    "CIFAR100": {
        "MLP": {"train": [14.04, 20.31, 23.28, 25.82, 27.84], "test": 26.12},
        "CNN": {"train": [18.10, 33.46, 41.38, 47.11, 52.89], "test": 40.74},
        "Transformer": {"train": [10.59, 16.84, 20.21, 22.85, 24.95], "test": 24.01}
    },
    "PCam": {
        "MLP": {"train": [70.15, 74.91, 76.36, 77.25, 77.64], "test": 70.84},
        "CNN": {"train": [75.35, 81.42, 83.30, 84.25, 85.47], "test": 78.86},
        "Transformer": {"train": [66.34, 73.49, 74.56, 75.18, 75.73], "test": 69.22}
    }
}

def plot_training_results(dataset_name):
    plt.figure(figsize=(10, 6))
    epochs = range(1, 6)
    colors = {'MLP': 'red', 'CNN': 'green', 'Transformer': 'blue'}
    
    for arch in ['MLP', 'CNN', 'Transformer']:
        # Plot Training Line
        plt.plot(epochs, data[dataset_name][arch]['train'], 
                 label=f'{arch} Training', color=colors[arch], marker='o')
        
        # Plot Validation (Test) point at the final epoch
        plt.scatter(5, data[dataset_name][arch]['test'], 
                    color=colors[arch], s=100, edgecolors='black', 
                    label=f'{arch} Test Acc', zorder=5)

    plt.title(f'Training vs. Test Accuracy: {dataset_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_learning_curves.png')
    plt.show()

# Generate the three required plots
for ds in ["Adult", "CIFAR100", "PCam"]:
    plot_training_results(ds)
