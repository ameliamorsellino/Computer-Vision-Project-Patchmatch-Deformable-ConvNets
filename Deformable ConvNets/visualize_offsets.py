import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import datasets, transforms

from models import DeformableCNN

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(42)


def get_sampling_locations(offsets, kernel_size=3):
    """
    Converts offsets into absolute sampling positions.
    For a 3×3 convolution, the base kernel positions are:
    (-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)
    The offsets shift these positions: p_n + delta(p_n)
    """
    B, C, H, W = offsets.shape
    n_points = kernel_size * kernel_size
    
    # base offsets of 3x3 kernel
    base_offsets = []
    for i in range(-(kernel_size // 2), kernel_size // 2 + 1):
        for j in range(-(kernel_size // 2), kernel_size // 2 + 1):
            base_offsets.append((i, j))
    
    cy, cx = H // 2, W // 2
    
    locations = []
    for k in range(n_points):
        # offset[:, 2k] = offset y, offset[:, 2k+1] = offset x
        dy = offsets[0, 2 * k, cy, cx].item()
        dx = offsets[0, 2 * k + 1, cy, cx].item()
        
        base_y, base_x = base_offsets[k]
        locations.append((cx + base_x + dx, cy + base_y + dy))
    
    return locations, (cx, cy)


def visualize_offsets_on_images(model, dataset_name='MNIST', n_samples=8):
    """Displays the sampling positions of the deformable convolutions overlaid on the input images"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset_class = datasets.MNIST if dataset_name == 'MNIST' else datasets.FashionMNIST
    test_set = dataset_class(root='./data', train=False, download=True, transform=transform)
    
    # select sample for class
    class_samples = {}
    for img, label in test_set:
        if label not in class_samples and len(class_samples) < n_samples:
            class_samples[label] = img
        if len(class_samples) >= n_samples:
            break
    
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 9))
    fig.suptitle(
        f'Deformable Conv Sampling Locations - {dataset_name}\n'
        '(Red: learned sampling points, Blue: standard grid)',
        fontsize=14, fontweight='bold'
    )
    
    layer_names = ['deform1', 'deform2', 'deform3']
    layer_titles = [
        'Deformable Conv Layer 1',
        'Deformable Conv Layer 2',
        'Deformable Conv Layer 3'
    ]
    
    model.eval()
    
    for col_idx, (label, img) in enumerate(sorted(class_samples.items())):
        if col_idx >= n_samples:
            break
        
        with torch.no_grad():
            _ = model(img.unsqueeze(0))
        
        for row_idx, (layer_name, layer_title) in enumerate(zip(layer_names, layer_titles)):
            ax = axes[row_idx, col_idx]
            offsets = model.saved_offsets[layer_name]
            
            locations, center = get_sampling_locations(offsets)
            
            # show image
            img_np = img.squeeze().numpy()
            ax.imshow(img_np, cmap='gray', alpha=0.7, extent=[0, offsets.shape[3], offsets.shape[2], 0])
            
            # standard offsets in blue
            kernel_size = 3
            for i in range(-(kernel_size // 2), kernel_size // 2 + 1):
                for j in range(-(kernel_size // 2), kernel_size // 2 + 1):
                    ax.plot(center[0] + j, center[1] + i, 's', color='dodgerblue', markersize=6, alpha=0.5)
            
            # deformable positions in red
            for (lx, ly) in locations:
                ax.plot(lx, ly, 'o', color='red', markersize=6, alpha=0.8)
                ax.plot([center[0], lx], [center[1], ly], '-', color='red', alpha=0.3, linewidth=1)
            
            # center
            ax.plot(center[0], center[1], '*', color='yellow', markersize=10, zorder=5)
            
            if col_idx == 0:
                ax.set_ylabel(layer_title, fontsize=10)
            if row_idx == 0:
                ax.set_title(f'Class {label}', fontsize=11)
            
            ax.set_xlim(center[0] - 5, center[0] + 5)
            ax.set_ylim(center[1] + 5, center[1] - 5)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'offsets_visualization_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_offset_magnitude_heatmap(model, dataset_name='MNIST'):
    """Heatmap of the offset magnitudes to show where the network deforms the kernel the most"""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    dataset_class = datasets.MNIST if dataset_name == 'MNIST' else datasets.FashionMNIST
    test_set = dataset_class(root='./data', train=False, download=True, transform=transform)
    
    # accumulate offset magnitudes over many samples
    model.eval()
    n_samples = 100
    accumulated = {name: None for name in ['deform1', 'deform2', 'deform3']}
    
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    
    for i, (img, _) in enumerate(loader):
        if i >= n_samples:
            break
        with torch.no_grad():
            _ = model(img)
        
        for name in accumulated:
            offsets = model.saved_offsets[name]
            # magnitude: sqrt(dx^2 + dy^2) averaged over the kernel points
            B, C, H, W = offsets.shape
            n_points = C // 2
            magnitude = torch.zeros(H, W)
            for k in range(n_points):
                dy = offsets[0, 2 * k]
                dx = offsets[0, 2 * k + 1]
                magnitude += torch.sqrt(dy ** 2 + dx ** 2)
            magnitude /= n_points
            
            if accumulated[name] is None:
                accumulated[name] = magnitude.numpy()
            else:
                accumulated[name] += magnitude.numpy()
    
    # mean
    for name in accumulated:
        accumulated[name] /= n_samples
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f'Average Offset Magnitude Heatmaps - {dataset_name}\n'
        '(Brighter = larger deformation from standard grid)',
        fontsize=14, fontweight='bold'
    )
    
    layer_titles = ['Deform Layer 1', 'Deform Layer 2', 'Deform Layer 3']
    
    for idx, (name, title) in enumerate(zip(['deform1', 'deform2', 'deform3'], layer_titles)):
        ax = axes[idx]
        im = ax.imshow(accumulated[name], cmap='hot', interpolation='bilinear')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'offset_heatmaps_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    for dataset_name in ['MNIST', 'FashionMNIST']:
        print(f"Visualizing offsets for {dataset_name}")

        model = DeformableCNN()
        model_path = os.path.join(RESULTS_DIR, f'DeformableCNN_{dataset_name}.pth')
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path,map_location='cpu'))
            print(f"Loaded model from {model_path}")
        else:
            print(f"WARNING: {model_path} not found. Run train.py first.")
            print("Using random weights for demonstration")
        
        visualize_offsets_on_images(model, dataset_name)
        visualize_offset_magnitude_heatmap(model, dataset_name)