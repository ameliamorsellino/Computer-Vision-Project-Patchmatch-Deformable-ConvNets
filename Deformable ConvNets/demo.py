
"""
Experiments
1: Training and comparison StandardCNN vs DeformableCNN (MNIST + FashionMNIST)
2: Visualization of learned offsets
3: Robustness tests to geometric transformations
4: Toy visualizations for standard conv, deformable conv, pooling, bilinear interpolation
5: Feature-response comparison between standard and deformable layers
"""

import os
import time


def main():
    start = time.time()
    print('DEFORMABLE CONVOLUTIONAL NETWORKS')

    print('Experiment 1: training and comparison')
    from train import run_experiment
    for dataset_name in ['MNIST', 'FashionMNIST']:
        run_experiment(dataset_name)

    print('Experiment 2: offset visualization')
    from visualize_offsets import (
        visualize_offsets_on_images,
        visualize_offset_magnitude_heatmap,
    )
    from models import DeformableCNN
    import torch

    for dataset_name in ['MNIST', 'FashionMNIST']:
        model = DeformableCNN()
        model_path = os.path.join('results', f'DeformableCNN_{dataset_name}.pth')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        visualize_offsets_on_images(model, dataset_name)
        visualize_offset_magnitude_heatmap(model, dataset_name)

    print('Experiment 3: geometric transformation robustness')
    from experiment_geometric import (
        experiment_rotation_robustness,
        experiment_scale_robustness,
        experiment_shear_robustness,
        plot_geometric_results,
        visualize_transformed_samples,
    )

    for dataset_name in ['MNIST', 'FashionMNIST']:
        visualize_transformed_samples(dataset_name)
        rot = experiment_rotation_robustness(dataset_name)
        scale = experiment_scale_robustness(dataset_name)
        shear = experiment_shear_robustness(dataset_name)
        plot_geometric_results(dataset_name, rot, scale, shear)

    print('Experiment 4: toy layer visualizations')
    from toy_layers import run_toy_layer_visualizations
    run_toy_layer_visualizations()

    print('Experiment 5: feature-response comparison')
    from visualize_feature_maps import run_feature_response_visualizations
    run_feature_response_visualizations()

    elapsed = time.time() - start
    print(f'done in {elapsed:.2f} s')


if __name__ == '__main__':
    main()
