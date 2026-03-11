import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import StandardCNN, DeformableCNN

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(42) 


def get_transformed_loader(dataset_name, rotation=0, scale=1.0,
                           translate=(0, 0), shear=0):
    """Creates a test loader with specific geometric transformations"""
    transform_list = [transforms.ToTensor(),]
    
    # apply geometric transformation before normalization
    if rotation != 0 or scale != 1.0 or translate != (0, 0) or shear != 0:
        geo_transform = transforms.RandomAffine(
            degrees=(rotation, rotation),
            translate=translate if translate != (0, 0) else None,
            scale=(scale, scale) if scale != 1.0 else None,
            shear=(shear, shear) if shear != 0 else None,
            fill=0
        )
        transform_list.insert(0, geo_transform)
    
    transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(transform_list)
    
    dataset_class = (datasets.MNIST if dataset_name == 'MNIST'
                     else datasets.FashionMNIST)
    test_set = dataset_class(root='./data', train=False,
                             download=True, transform=transform)
    
    return DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def experiment_rotation_robustness(dataset_name='MNIST'):
    """Tests rotation robustness with rotations from 0° to 180° by 30° steps"""
    print(f"\nRotation Robustness Test ({dataset_name})")
    
    std_model = StandardCNN()
    def_model = DeformableCNN()
    
    std_path = os.path.join(RESULTS_DIR, f'StandardCNN_{dataset_name}.pth')
    def_path = os.path.join(RESULTS_DIR, f'DeformableCNN_{dataset_name}.pth')
    
    if os.path.exists(std_path) and os.path.exists(def_path):
        std_model.load_state_dict(torch.load(std_path, map_location='cpu'))
        def_model.load_state_dict(torch.load(def_path, map_location='cpu'))
    else:
        print("Trained models not found, first run train.py (using random weights instead)")
    
    rotations = list(range(0, 195, 30))
    std_accs = []
    def_accs = []
    
    for rot in tqdm(rotations, desc="Testing rotations"):
        loader = get_transformed_loader(dataset_name, rotation=rot)
        
        std_acc = evaluate_accuracy(std_model, loader)
        def_acc = evaluate_accuracy(def_model, loader)
        
        std_accs.append(std_acc)
        def_accs.append(def_acc)
        
        print(f"Rotation {rot:3d}°: Standard={std_acc:.2f}%, "
              f"Deformable={def_acc:.2f}%")
    
    return rotations, std_accs, def_accs


def experiment_scale_robustness(dataset_name='MNIST'):
    print(f"\nScale Robustness Test ({dataset_name})")
    
    std_model = StandardCNN()
    def_model = DeformableCNN()
    
    std_path = os.path.join(RESULTS_DIR, f'StandardCNN_{dataset_name}.pth')
    def_path = os.path.join(RESULTS_DIR, f'DeformableCNN_{dataset_name}.pth')
    
    if os.path.exists(std_path) and os.path.exists(def_path):
        std_model.load_state_dict(torch.load(std_path, map_location='cpu'))
        def_model.load_state_dict(torch.load(def_path, map_location='cpu'))
    
    scales = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    std_accs = []
    def_accs = []
    
    for scale in tqdm(scales, desc="Testing scales"):
        loader = get_transformed_loader(dataset_name, scale=scale)
        
        std_acc = evaluate_accuracy(std_model, loader)
        def_acc = evaluate_accuracy(def_model, loader)
        
        std_accs.append(std_acc)
        def_accs.append(def_acc)
        
        print(f"  Scale {scale:.1f}: Standard={std_acc:.2f}%, "
              f"Deformable={def_acc:.2f}%")
    
    return scales, std_accs, def_accs


def experiment_shear_robustness(dataset_name='MNIST'):
    print(f"\nShear Robustness Test ({dataset_name})")
    
    std_model = StandardCNN()
    def_model = DeformableCNN()
    
    std_path = os.path.join(RESULTS_DIR, f'StandardCNN_{dataset_name}.pth')
    def_path = os.path.join(RESULTS_DIR, f'DeformableCNN_{dataset_name}.pth')
    
    if os.path.exists(std_path) and os.path.exists(def_path):
        std_model.load_state_dict(torch.load(std_path, map_location='cpu'))
        def_model.load_state_dict(torch.load(def_path, map_location='cpu'))
    
    shears = list(range(0, 65, 10))
    std_accs = []
    def_accs = []
    
    for shear in tqdm(shears, desc="Testing shears"):
        loader = get_transformed_loader(dataset_name, shear=shear)
        
        std_acc = evaluate_accuracy(std_model, loader)
        def_acc = evaluate_accuracy(def_model, loader)
        
        std_accs.append(std_acc)
        def_accs.append(def_acc)
    
    return shears, std_accs, def_accs

# plots
def plot_geometric_results(dataset_name, rot_results, scale_results,
                           shear_results):
    """Plots all geometric robustness results"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Geometric Transformation Robustness - {dataset_name}\n'
        f'(Models trained on standard data, tested on transformed data)',
        fontsize=14, fontweight='bold'
    )
    
    colors = {'Standard': '#2196F3', 'Deformable': '#FF5722'}
    
    # rotation
    ax = axes[0]
    rotations, std_accs, def_accs = rot_results
    ax.plot(rotations, std_accs, '-o', color=colors['Standard'],
            label='StandardCNN', linewidth=2, markersize=4)
    ax.plot(rotations, def_accs, '-s', color=colors['Deformable'],
            label='DeformableCNN', linewidth=2, markersize=4)
    ax.fill_between(rotations, std_accs, def_accs, alpha=0.1,
                    color='green' if def_accs[-1] > std_accs[-1] else 'red')
    ax.set_xlabel('Rotation (degrees)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Rotation Robustness', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    
    # scale
    ax = axes[1]
    scales, std_accs, def_accs = scale_results
    ax.plot(scales, std_accs, '-o', color=colors['Standard'],
            label='StandardCNN', linewidth=2, markersize=4)
    ax.plot(scales, def_accs, '-s', color=colors['Deformable'],
            label='DeformableCNN', linewidth=2, markersize=4)
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Scale Factor', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Scale Robustness', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    
    # shear
    ax = axes[2]
    shears, std_accs, def_accs = shear_results
    ax.plot(shears, std_accs, '-o', color=colors['Standard'],
            label='StandardCNN', linewidth=2, markersize=4)
    ax.plot(shears, def_accs, '-s', color=colors['Deformable'],
            label='DeformableCNN', linewidth=2, markersize=4)
    ax.set_xlabel('Shear (degrees)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Shear Robustness', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR,
                        f'geometric_robustness_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_transformed_samples(dataset_name='MNIST'):
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle(f'Transformed Samples: {dataset_name}', fontsize=14, fontweight='bold')
    
    configs = [
        ("Original", dict(rotation=0)),
        ("Rot 45°", dict(rotation=45)),
        ("Rot 90°", dict(rotation=90)),
        ("Scale 0.6", dict(scale=0.6)),
    ]
    
    for row, (title, kwargs) in enumerate(configs):
        loader = get_transformed_loader(dataset_name, **kwargs)
        images, labels = next(iter(loader))
        
        for col in range(8):
            ax = axes[row, col]
            img = images[col].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'transformed_samples_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    for dataset_name in ['MNIST', 'FashionMNIST']:
        print(f"Geometric Robustness (experiment 3) - {dataset_name}")
        
        visualize_transformed_samples(dataset_name)
        
        rot_results = experiment_rotation_robustness(dataset_name)
        scale_results = experiment_scale_robustness(dataset_name)
        shear_results = experiment_shear_robustness(dataset_name)
        
        plot_geometric_results(dataset_name, rot_results, scale_results, shear_results)