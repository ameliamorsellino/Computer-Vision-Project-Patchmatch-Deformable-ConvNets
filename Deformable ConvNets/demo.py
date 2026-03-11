"""
Experiments
1: Training and comparison StandardCNN vs DeformableCNN (MNIST + FashionMNIST)
2: Visualization of offsets learned by the Deformable Convolutions
3: Robustness tests to geometric transformations
"""

import os
import time

def main():
    start = time.time()
    print("DEFORMABLE CONVOLUTIONAL NETWORKS")
    
    # experiment 1
    print("Experiment 1: training and comparison")

    from train import run_experiment
    
    results_mnist, models_mnist = run_experiment('MNIST')
    results_fmnist, models_fmnist = run_experiment('FashionMNIST')
    
    # experiment 2
    print("Experiment 2: Offset Visualization")

    from visualize_offsets import (
        visualize_offsets_on_images,
        visualize_offset_magnitude_heatmap
    )
    
    for ds_name in ['MNIST', 'FashionMNIST']:
        from models import DeformableCNN
        import torch
        model = DeformableCNN()
        model_path = os.path.join('results', f'DeformableCNN_{ds_name}.pth')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        visualize_offsets_on_images(model, ds_name)
        visualize_offset_magnitude_heatmap(model, ds_name)
    
    # experiment 3
    print("Experiment 3: Geometric Transformation Robustness")
    
    from experiment_geometric import (
        experiment_rotation_robustness,
        experiment_scale_robustness,
        experiment_shear_robustness,
        plot_geometric_results,
        visualize_transformed_samples
    )
    
    for ds_name in ['MNIST', 'FashionMNIST']:
        visualize_transformed_samples(ds_name)
        rot = experiment_rotation_robustness(ds_name)
        scale = experiment_scale_robustness(ds_name)
        shear = experiment_shear_robustness(ds_name)
        plot_geometric_results(ds_name, rot, scale, shear)

if __name__ == "__main__":
    main()