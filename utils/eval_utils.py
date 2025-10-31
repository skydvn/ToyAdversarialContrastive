import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_tsne(model, test_loader, device, save_path='tsne_plot.png'):
    """
    Generate T-SNE visualization for h, z1 and z2 embeddings on test dataset.

    Args:
        model: Trained ResNetEncoder model
        test_loader: DataLoader for test dataset
        device: cuda or cpu
        save_path: Path to save the visualization
    """
    model.eval()

    h_list = []
    z1_list = []
    z2_list = []
    labels_list = []

    print("Extracting embeddings from test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # Forward pass
            h, z1, z2 = model(images)

            h_list.append(h.cpu().numpy())
            z1_list.append(z1.cpu().numpy())
            z2_list.append(z2.cpu().numpy())
            labels_list.append(labels.numpy())

    # Concatenate all batches
    h_embeddings = np.concatenate(h_list, axis=0)
    z1_embeddings = np.concatenate(z1_list, axis=0)
    z2_embeddings = np.concatenate(z2_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    print(f"Total samples: {len(labels)}")
    print(f"h shape: {h_embeddings.shape}, z1 shape: {z1_embeddings.shape}, z2 shape: {z2_embeddings.shape}")

    # Apply T-SNE
    print("Applying T-SNE to h...")
    tsne_h = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    h_2d = tsne_h.fit_transform(h_embeddings)

    print("Applying T-SNE to z1...")
    tsne_z1 = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    z1_2d = tsne_z1.fit_transform(z1_embeddings)

    print("Applying T-SNE to z2...")
    tsne_z2 = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    z2_2d = tsne_z2.fit_transform(z2_embeddings)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Plot h (Encoder Features)
    scatter0 = axes[0].scatter(h_2d[:, 0], h_2d[:, 1],
                               c=labels, cmap='tab10',
                               s=5, alpha=0.6)
    axes[0].set_title('T-SNE of h (Encoder Features)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('T-SNE Component 1')
    axes[0].set_ylabel('T-SNE Component 2')
    axes[0].grid(True, alpha=0.3)

    # Plot z1 (Projection Head)
    scatter1 = axes[1].scatter(z1_2d[:, 0], z1_2d[:, 1],
                               c=labels, cmap='tab10',
                               s=5, alpha=0.6)
    axes[1].set_title('T-SNE of z1 (Projection Head)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('T-SNE Component 1')
    axes[1].set_ylabel('T-SNE Component 2')
    axes[1].grid(True, alpha=0.3)

    # Plot z2 (Adversarial Projection Head)
    scatter2 = axes[2].scatter(z2_2d[:, 0], z2_2d[:, 1],
                               c=labels, cmap='tab10',
                               s=5, alpha=0.6)
    axes[2].set_title('T-SNE of z2 (Adversarial Projection Head)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('T-SNE Component 1')
    axes[2].set_ylabel('T-SNE Component 2')
    axes[2].grid(True, alpha=0.3)

    # Add colorbar
    cbar0 = plt.colorbar(scatter0, ax=axes[0], ticks=range(10))
    cbar0.set_label('Class')
    cbar0.ax.set_yticklabels(class_names)

    cbar1 = plt.colorbar(scatter1, ax=axes[1], ticks=range(10))
    cbar1.set_label('Class')
    cbar1.ax.set_yticklabels(class_names)

    cbar2 = plt.colorbar(scatter2, ax=axes[2], ticks=range(10))
    cbar2.set_label('Class')
    cbar2.ax.set_yticklabels(class_names)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"T-SNE visualization saved to {save_path}")
    plt.show()

    return h_2d, z1_2d, z2_2d, labels


def visualize_tsne_comparison(model, test_loader, device, save_path='tsne_comparison.png'):
    """
    Create a more detailed comparison visualization with statistics.
    """
    model.eval()

    h_list = []
    z1_list = []
    z2_list = []
    labels_list = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            h, z1, z2 = model(images)

            h_list.append(h.cpu().numpy())
            z1_list.append(z1.cpu().numpy())
            z2_list.append(z2.cpu().numpy())
            labels_list.append(labels.numpy())

    h_embeddings = np.concatenate(h_list, axis=0)
    z1_embeddings = np.concatenate(z1_list, axis=0)
    z2_embeddings = np.concatenate(z2_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Apply T-SNE
    print("Applying T-SNE...")
    tsne_h = TSNE(n_components=2, random_state=42, perplexity=30)
    h_2d = tsne_h.fit_transform(h_embeddings)

    tsne_z1 = TSNE(n_components=2, random_state=42, perplexity=30)
    z1_2d = tsne_z1.fit_transform(z1_embeddings)

    tsne_z2 = TSNE(n_components=2, random_state=42, perplexity=30)
    z2_2d = tsne_z2.fit_transform(z2_embeddings)

    # Create figure with subplots
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Main T-SNE plots
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])
    ax2 = fig.add_subplot(gs[:, 2])

    for i, class_name in enumerate(class_names):
        mask = labels == i
        ax0.scatter(h_2d[mask, 0], h_2d[mask, 1],
                    c=[colors[i]], label=class_name, s=10, alpha=0.6)
        ax1.scatter(z1_2d[mask, 0], z1_2d[mask, 1],
                    c=[colors[i]], label=class_name, s=10, alpha=0.6)
        ax2.scatter(z2_2d[mask, 0], z2_2d[mask, 1],
                    c=[colors[i]], label=class_name, s=10, alpha=0.6)

    ax0.set_title('h (Encoder Features)', fontsize=14, fontweight='bold')
    ax0.set_xlabel('T-SNE Component 1')
    ax0.set_ylabel('T-SNE Component 2')
    ax0.legend(loc='best', fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1.set_title('z1 (Projection Head)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('T-SNE Component 1')
    ax1.set_ylabel('T-SNE Component 2')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_title('z2 (Adversarial Projection)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('T-SNE Component 1')
    ax2.set_ylabel('T-SNE Component 2')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Statistics
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')

    # Calculate class separation metrics
    from sklearn.metrics import silhouette_score
    sil_h = silhouette_score(h_embeddings, labels)
    sil_z1 = silhouette_score(z1_embeddings, labels)
    sil_z2 = silhouette_score(z2_embeddings, labels)

    stats_text = f"""
    Embedding Statistics:

    h (Encoder):
    - Shape: {h_embeddings.shape}
    - Silhouette Score: {sil_h:.4f}
    - Mean norm: {np.linalg.norm(h_embeddings, axis=1).mean():.4f}
    - Std norm: {np.linalg.norm(h_embeddings, axis=1).std():.4f}

    z1 (Projection):
    - Shape: {z1_embeddings.shape}
    - Silhouette Score: {sil_z1:.4f}
    - Mean norm: {np.linalg.norm(z1_embeddings, axis=1).mean():.4f}
    - Std norm: {np.linalg.norm(z1_embeddings, axis=1).std():.4f}

    z2 (Adversarial):
    - Shape: {z2_embeddings.shape}
    - Silhouette Score: {sil_z2:.4f}
    - Mean norm: {np.linalg.norm(z2_embeddings, axis=1).mean():.4f}
    - Std norm: {np.linalg.norm(z2_embeddings, axis=1).std():.4f}

    Higher silhouette score indicates
    better class separation.
    """

    ax3.text(0.1, 0.5, stats_text, fontsize=9,
             verticalalignment='center', family='monospace')

    # Embedding norm distribution
    ax4 = fig.add_subplot(gs[1, 3])
    h_norms = np.linalg.norm(h_embeddings, axis=1)
    z1_norms = np.linalg.norm(z1_embeddings, axis=1)
    z2_norms = np.linalg.norm(z2_embeddings, axis=1)

    ax4.hist(h_norms, bins=50, alpha=0.5, label='h', color='green')
    ax4.hist(z1_norms, bins=50, alpha=0.5, label='z1', color='blue')
    ax4.hist(z2_norms, bins=50, alpha=0.5, label='z2', color='red')
    ax4.set_xlabel('Embedding Norm')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Embedding Norm Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed visualization saved to {save_path}")
    plt.show()

    return h_2d, z1_2d, z2_2d, labels