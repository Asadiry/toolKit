import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def draw_tsne_distribution(repre_lists, label_lists, save_path):
    repre_lists = np.array(repre_lists)
    label_lists = np.array(label_lists)

    num_classes = len(np.unique(label_lists))
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(repre_lists)
    
    colors = plt.colormaps["tab10"]

    plt.figure(22, figsize=(16, 12))
    plt.xlabel('t-sne component 1')
    plt.ylabel('t-sne component 2')
    plt.title('T-SNE Visualization with Multiple Classes')
    for i in range(num_classes):
        class_data = data_tsne[label_lists == i]
        plt.scatter(class_data[:, 0], class_data[:, 1], color=colors(i), label=f'Class {i}')

    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":

    embedding = np.random.random((5000, 300))
    label = np.random.randint(0, 2, size=5000)
    draw_tsne_distribution(embedding, label)