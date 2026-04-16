
import os
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models import StandardCNN, DeformableCNN

RESULTS_DIR = 'results'
DEVICE = 'cpu'
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_LABELS = {
    'MNIST': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'FashionMNIST': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
}

TARGET_CLASSES = {
    'MNIST': [1, 0, 6, 9],
    'FashionMNIST': [1, 5, 7, 8],
}

STAGE_CONFIGS = {
    'stage1': {
        'std_module': lambda m: m.features[3].conv,
        'def_module': lambda m: m.deform_conv1.deform_conv,
        'offset_key': 'deform1',
    },
    'stage2': {
        'std_module': lambda m: m.features[4].conv,
        'def_module': lambda m: m.deform_conv2.deform_conv,
        'offset_key': 'deform2',
    },
    'stage3': {
        'std_module': lambda m: m.features[6].conv,
        'def_module': lambda m: m.deform_conv3.deform_conv,
        'offset_key': 'deform3',
    },
}


def _dataset_class(dataset_name):
    if dataset_name == 'MNIST':
        return datasets.MNIST
    if dataset_name == 'FashionMNIST':
        return datasets.FashionMNIST
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def _load_test_dataset(dataset_name):
    dataset_cls = _dataset_class(dataset_name)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return dataset_cls(root='./data', train=False, download=True, transform=transform)


def _select_indices(dataset, target_classes, per_class):
    counts = defaultdict(int)
    selected = []
    for idx, (_, label) in enumerate(dataset):
        if label in target_classes and counts[label] < per_class:
            selected.append(idx)
            counts[label] += 1
        if all(counts[c] >= per_class for c in target_classes):
            break
    return selected


def _build_batch(dataset, indices):
    images = []
    labels = []
    for idx in indices:
        x, y = dataset[idx]
        images.append(x)
        labels.append(y)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long)


def _load_models(dataset_name):
    std_path = os.path.join(RESULTS_DIR, f'StandardCNN_{dataset_name}.pth')
    def_path = os.path.join(RESULTS_DIR, f'DeformableCNN_{dataset_name}.pth')

    if not os.path.exists(std_path) or not os.path.exists(def_path):
        missing = [p for p in [std_path, def_path] if not os.path.exists(p)]
        raise FileNotFoundError(
            f'Missing checkpoints: {missing}. Run train.py or demo.py first.'
        )

    std_model = StandardCNN().to(DEVICE)
    def_model = DeformableCNN().to(DEVICE)
    std_model.load_state_dict(torch.load(std_path, map_location=DEVICE))
    def_model.load_state_dict(torch.load(def_path, map_location=DEVICE))
    std_model.eval()
    def_model.eval()
    return std_model, def_model


def _capture_stage_outputs(std_model, def_model, batch, stage_name):
    cfg = STAGE_CONFIGS[stage_name]
    activations = {}

    def hook_std(_, __, output):
        activations['std'] = output.detach().cpu()

    def hook_def(_, __, output):
        activations['def'] = output.detach().cpu()

    h_std = cfg['std_module'](std_model).register_forward_hook(hook_std)
    h_def = cfg['def_module'](def_model).register_forward_hook(hook_def)

    with torch.no_grad():
        std_model(batch)
        def_model(batch)

    h_std.remove()
    h_def.remove()

    offsets = def_model.saved_offsets[cfg['offset_key']].detach().cpu()
    return activations['std'], activations['def'], offsets


def _offset_magnitude(offsets):
    b, c, h, w = offsets.shape
    offsets = offsets.view(b, -1, 2, h, w)
    return torch.sqrt(torch.sum(offsets ** 2, dim=2)).mean(dim=1)


def _top_channels(acts, top_k):
    scores = acts.abs().mean(dim=(0, 2, 3))
    k = min(top_k, scores.numel())
    return torch.topk(scores, k=k).indices.tolist()


def _dataset_stub(dataset_name):
    return 'fashion' if dataset_name == 'FashionMNIST' else 'mnist'


def _plot_stage(dataset_name, stage_name, images, labels, std_acts, def_acts, offset_mag, std_top, def_top):
    label_names = DATASET_LABELS[dataset_name]
    n = images.shape[0]
    cols = 1 + len(std_top) + len(def_top) + 1

    fig, axes = plt.subplots(n, cols, figsize=(2.2 * cols, 2.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(n):
        ax = axes[row, 0]
        ax.imshow(images[row, 0].cpu(), cmap='gray')
        ax.set_title('input' if row == 0 else '')
        ax.set_ylabel(label_names[int(labels[row])])
        ax.axis('off')

        for col_idx, ch in enumerate(std_top, start=1):
            ax = axes[row, col_idx]
            fmap = std_acts[row, ch].numpy()
            ax.imshow(fmap, cmap='magma')
            ax.set_title(f'std c{ch}' if row == 0 else '')
            ax.axis('off')

        for offset_col, ch in enumerate(def_top, start=1 + len(std_top)):
            ax = axes[row, offset_col]
            fmap = def_acts[row, ch].numpy()
            ax.imshow(fmap, cmap='magma')
            ax.set_title(f'def c{ch}' if row == 0 else '')
            ax.axis('off')

        ax = axes[row, cols - 1]
        ax.imshow(offset_mag[row].numpy(), cmap='inferno')
        ax.set_title('offset |Δ|' if row == 0 else '')
        ax.axis('off')

    fig.suptitle(
        f'{dataset_name} - {stage_name}: standard vs deformable feature responses',
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(RESULTS_DIR, f'feature_maps_{_dataset_stub(dataset_name)}_{stage_name}.png'),
        dpi=180,
    )
    plt.close(fig)


def visualize_feature_responses(dataset_name, top_k=4, support_per_class=3, plot_per_class=1):
    dataset = _load_test_dataset(dataset_name)
    std_model, def_model = _load_models(dataset_name)

    target_classes = TARGET_CLASSES[dataset_name]
    support_indices = _select_indices(dataset, target_classes, per_class=support_per_class)
    plot_indices = _select_indices(dataset, target_classes, per_class=plot_per_class)

    support_batch, _ = _build_batch(dataset, support_indices)
    plot_batch, plot_labels = _build_batch(dataset, plot_indices)
    support_batch = support_batch.to(DEVICE)
    plot_batch = plot_batch.to(DEVICE)

    for stage_name in STAGE_CONFIGS:
        support_std, support_def, _ = _capture_stage_outputs(std_model, def_model, support_batch, stage_name)
        std_top = _top_channels(support_std, top_k=top_k)
        def_top = _top_channels(support_def, top_k=top_k)

        plot_std, plot_def, offsets = _capture_stage_outputs(std_model, def_model, plot_batch, stage_name)
        offset_mag = _offset_magnitude(offsets)

        _plot_stage(
            dataset_name=dataset_name,
            stage_name=stage_name,
            images=plot_batch.cpu(),
            labels=plot_labels,
            std_acts=plot_std,
            def_acts=plot_def,
            offset_mag=offset_mag,
            std_top=std_top,
            def_top=def_top,
        )


def run_feature_response_visualizations(dataset_names=('MNIST', 'FashionMNIST')):
    for dataset_name in dataset_names:
        print(f'Feature-map comparison: {dataset_name}')
        visualize_feature_responses(dataset_name)


if __name__ == '__main__':
    run_feature_response_visualizations()
