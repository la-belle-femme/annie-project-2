import matplotlib.pyplot as plt

# Assuming `results` is your data
def plot_accuracy_vs_filters(results):
    filter_values = sorted(set(result[0] for result in results))
    batch_sizes = sorted(set(result[1] for result in results))

    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        accuracies = [result[2] for result in results if result[1] == batch_size]
        plt.plot(filter_values, accuracies, marker='o', label=f'Batch Size {batch_size}')

    plt.title('Model Accuracy vs Number of Filters')
    plt.xlabel('Number of Filters')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_filters.png')
    plt.close()

def plot_accuracy_vs_batch_size(results):
    batch_sizes = sorted(set(result[1] for result in results))
    filter_values = sorted(set(result[0] for result in results))

    plt.figure(figsize=(10, 6))
    for num_filters in filter_values:
        accuracies = [result[2] for result in results if result[0] == num_filters]
        plt.plot(batch_sizes, accuracies, marker='o', label=f'Filters {num_filters}')

    plt.title('Model Accuracy vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_batch_size.png')
    plt.close()

# Re-run the plotting functions
plot_accuracy_vs_filters(results)
plot_accuracy_vs_batch_size(results)
