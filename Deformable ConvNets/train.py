import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from models import StandardCNN, DeformableCNN, count_parameters

torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 256
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
DEVICE = torch.device('cpu')
RESULTS_DIR = 'results'

# test mode
TRAIN_SUBSET = 5000   
TEST_SUBSET = 1000    
SUBSET_SEED = 5

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_dataloaders(dataset_name='MNIST'):
    """Creates train/test dataloaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset_class = datasets.MNIST if dataset_name == 'MNIST' else datasets.FashionMNIST

    train_set = dataset_class(root='./data', train=True, download=True, transform=transform)
    test_set  = dataset_class(root='./data', train=False, download=True, transform=transform)

    def make_subset(ds, n, seed):
        if n is None:
            return ds
        n = int(n)
        if n <= 0:
            raise ValueError("Subset size must be > 0")
        if n > len(ds):
            raise ValueError(f"Subset size {n} > dataset size {len(ds)}")

        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
        return Subset(ds, idx)

    train_set = make_subset(train_set, TRAIN_SUBSET, SUBSET_SEED)
    test_set  = make_subset(test_set,  TEST_SUBSET,  SUBSET_SEED + 1)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluation on test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, 100.0 * correct / total


def measure_inference_time(model, input_shape=(1, 1, 28, 28), n_runs=100):
    model.eval()
    x = torch.randn(*input_shape).to(DEVICE)
    
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return np.mean(times), np.std(times)


def train_model(model, dataset_name='MNIST'):
    print(f"Training {model.get_name()} on {dataset_name}")
    
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    train_loader, test_loader = get_dataloaders(dataset_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_time': []
    }
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        epoch_time = time.time() - epoch_start
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
    
    # inference time
    mean_time, std_time = measure_inference_time(model)
    print(f"\nInference time: {mean_time:.2f} ± {std_time:.2f} ms")
    
    history['inference_ms_mean'] = mean_time
    history['inference_ms_std'] = std_time
    history['total_params'] = total_params
    history['trainable_params'] = trainable_params
    history['final_test_acc'] = history['test_acc'][-1]
    history['best_test_acc'] = max(history['test_acc'])
    
    return history


def plot_comparison(results, dataset_name):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'Standard CNN vs Deformable CNN - {dataset_name}', fontsize=16, fontweight='bold')
    
    colors = {'StandardCNN': '#2196F3', 'DeformableCNN': '#FF5722'}
    epochs = range(1, NUM_EPOCHS + 1)
    
    # training loss
    ax = axes[0, 0]
    for name, hist in results.items():
        ax.plot(epochs, hist['train_loss'], '-o', color=colors[name],
                label=name, markersize=4, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    
    # test loss
    ax = axes[0, 1]
    for name, hist in results.items():
        ax.plot(epochs, hist['test_loss'], '-o', color=colors[name],
                label=name, markersize=4, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Test Loss')
    ax.legend()
    
    # test accuracy
    ax = axes[1, 0]
    for name, hist in results.items():
        ax.plot(epochs, hist['test_acc'], '-o', color=colors[name], label=f"{name} (best: {hist['best_test_acc']:.2f}%)", markersize=4, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy')
    ax.legend()
    
    # summary table
    ax = axes[1, 1]
    model_names = list(results.keys())
    x_pos = np.arange(len(model_names))
    
    metrics = {
        'Best Acc (%)': [results[n]['best_test_acc'] for n in model_names],
        'Params (K)': [results[n]['total_params']/1000 for n in model_names],
        'Inference (ms)': [results[n]['inference_ms_mean'] for n in model_names],
    }
    
    table_data = []
    for name in model_names:
        r = results[name]
        table_data.append([
            name,
            f"{r['best_test_acc']:.2f}%",
            f"{r['total_params']:,}",
            f"{r['inference_ms_mean']:.2f}ms"
        ])
    
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Best Acc', 'Params', 'Inference'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Summary', pad=20)
    
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'comparison_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def run_experiment(dataset_name='MNIST'):
    results = {}
    models_trained = {}

    # standard cnn
    std_model = StandardCNN().to(DEVICE)
    results['StandardCNN'] = train_model(std_model, dataset_name)
    models_trained['StandardCNN'] = std_model
    
    # deformable cn
    def_model = DeformableCNN().to(DEVICE)
    results['DeformableCNN'] = train_model(def_model, dataset_name)
    models_trained['DeformableCNN'] = def_model

    plot_comparison(results, dataset_name)
    
    # save numerical results and converts to serializable types
    serializable = {}
    for name, hist in results.items():
        serializable[name] = {
            k: (v if not isinstance(v, (np.floating, np.integer))
                 else float(v))
            for k, v in hist.items()
        }
    
    path = os.path.join(RESULTS_DIR, f'results_{dataset_name}.json')
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved results: {path}")
    
    # save models
    for name, model in models_trained.items():
        mpath = os.path.join(RESULTS_DIR, f'{name}_{dataset_name}.pth')
        torch.save(model.state_dict(), mpath)
    
    return results, models_trained


if __name__ == "__main__":
    print("EXPERIMENT 1: Standard CNN vs Deformable CNN")
    
    # MNIST
    results_mnist, models_mnist = run_experiment('MNIST')
    
    # FashionMNIST 
    results_fmnist, models_fmnist = run_experiment('FashionMNIST')
