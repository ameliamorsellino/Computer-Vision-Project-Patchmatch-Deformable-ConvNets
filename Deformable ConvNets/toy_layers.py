import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def _draw_matrix(ax, mat, title='', cmap='viridis', vmin=None, vmax=None):
    ax.imshow(mat, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.grid(color='white', alpha=0.35, linewidth=0.7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:.1f}' if isinstance(mat[i, j], float) else f'{mat[i, j]}',
                    ha='center', va='center', fontsize=8, color='white')
    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)


def _bilinear_sample(mat, y, x):
    h, w = mat.shape
    y = np.clip(y, 0, h - 1)
    x = np.clip(x, 0, w - 1)

    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, h - 1)
    x1 = min(x0 + 1, w - 1)

    wy = y - y0
    wx = x - x0

    w00 = (1 - wy) * (1 - wx)
    w01 = (1 - wy) * wx
    w10 = wy * (1 - wx)
    w11 = wy * wx

    value = (
        w00 * mat[y0, x0]
        + w01 * mat[y0, x1]
        + w10 * mat[y1, x0]
        + w11 * mat[y1, x1]
    )
    return value, {(y0, x0): w00, (y0, x1): w01, (y1, x0): w10, (y1, x1): w11}


def plot_standard_convolution_toy(results_dir=RESULTS_DIR):
    inp = np.zeros((7, 7), dtype=float)
    inp[1:6, 2] = [0.2, 0.6, 1.0, 0.6, 0.2]
    inp[3, 2:6] = [0.3, 0.8, 0.6, 0.2]
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)

    cy, cx = 3, 3
    patch = inp[cy - 1:cy + 2, cx - 1:cx + 2]
    response = float(np.sum(patch * kernel))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    _draw_matrix(axes[0], inp, 'Input feature map', cmap='magma', vmin=0, vmax=1.2)
    axes[0].add_patch(Rectangle((cx - 1 - 0.5, cy - 1 - 0.5), 3, 3, fill=False, ec='cyan', lw=2))
    yy, xx = np.mgrid[cy - 1:cy + 2, cx - 1:cx + 2]
    axes[0].scatter(xx, yy, c='cyan', s=35)
    axes[0].text(0.1, -0.15, 'Regular 3x3 sampling grid', transform=axes[0].transAxes, fontsize=10)

    _draw_matrix(axes[1], kernel, 'Kernel weights', cmap='coolwarm')

    axes[2].axis('off')
    axes[2].set_title('Output at p0')
    axes[2].text(
        0.0,
        0.95,
        '\n'.join([
            'Standard convolution:',
            'sample fixed 3x3 points',
            '',
            f'patch =\n{np.array2string(patch, precision=1)}',
            '',
            f'response = sum(patch * kernel) = {response:.3f}',
        ]),
        va='top',
        family='monospace',
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'toy_standard_conv.png'), dpi=180)
    plt.close(fig)


def plot_deformable_convolution_toy(results_dir=RESULTS_DIR):
    inp = np.zeros((7, 7), dtype=float)
    for k in range(1, 6):
        inp[k, k] = 1.0
    inp[3, 2] = 0.55
    inp[4, 3] = 0.55
    kernel = np.ones((3, 3), dtype=float) / 9.0

    cy, cx = 3, 3
    regular_points = [(cy + dy, cx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    offsets = [(-0.5, -0.2), (-0.3, 0.0), (0.2, 0.3),
               (-0.2, -0.2), (0.0, 0.0), (0.3, 0.3),
               (0.1, -0.1), (0.2, 0.2), (0.45, 0.45)]
    deformed_points = [(y + oy, x + ox) for (y, x), (oy, ox) in zip(regular_points, offsets)]

    regular_vals = [inp[int(y), int(x)] for y, x in regular_points]
    deform_vals = [_bilinear_sample(inp, y, x)[0] for y, x in deformed_points]
    regular_response = float(np.sum(np.array(regular_vals) * kernel.reshape(-1)))
    deform_response = float(np.sum(np.array(deform_vals) * kernel.reshape(-1)))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    _draw_matrix(axes[0], inp, 'Input with regular grid', cmap='magma', vmin=0, vmax=1.0)
    for y, x in regular_points:
        axes[0].scatter([x], [y], c='cyan', s=35)
    axes[0].add_patch(Rectangle((cx - 1 - 0.5, cy - 1 - 0.5), 3, 3, fill=False, ec='cyan', lw=2))

    _draw_matrix(axes[1], inp, 'Input with deformed grid', cmap='magma', vmin=0, vmax=1.0)
    for (ry, rx), (dy, dx) in zip(regular_points, deformed_points):
        axes[1].scatter([rx], [ry], c='white', s=16, alpha=0.65)
        axes[1].scatter([dx], [dy], c='lime', s=30)
        axes[1].plot([rx, dx], [ry, dy], color='lime', alpha=0.6, linewidth=1.2)

    axes[2].axis('off')
    axes[2].set_title('Effect on the response')
    axes[2].text(
        0.0,
        0.95,
        '\n'.join([
            'Deformable convolution:',
            'offsets move the 3x3 samples toward the diagonal',
            '',
            f'regular response   = {regular_response:.3f}',
            f'deformable response= {deform_response:.3f}',
            '',
            'When structure is curved or misaligned,',
            'moving the samples can capture it better.',
        ]),
        va='top',
        family='monospace',
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'toy_deform_conv.png'), dpi=180)
    plt.close(fig)


def plot_pooling_toy(results_dir=RESULTS_DIR):
    inp = np.array([
        [0.1, 0.2, 0.9, 0.1],
        [0.0, 0.8, 0.4, 0.2],
        [0.2, 0.3, 0.1, 0.7],
        [0.6, 0.1, 0.2, 0.5],
    ], dtype=float)

    max_pool = np.array([
        [np.max(inp[0:2, 0:2]), np.max(inp[0:2, 2:4])],
        [np.max(inp[2:4, 0:2]), np.max(inp[2:4, 2:4])],
    ])

    adap_inp = np.array([
        [0.2, 0.3, 0.1, 0.0],
        [0.4, 0.6, 0.3, 0.2],
        [0.8, 0.7, 0.5, 0.1],
        [0.9, 0.6, 0.2, 0.2],
    ], dtype=float)
    adaptive_avg = np.array([[np.mean(adap_inp)]])

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    _draw_matrix(axes[0, 0], inp, 'MaxPool 2x2 input', cmap='magma', vmin=0, vmax=1.0)
    for y in [1.5]:
        axes[0, 0].axhline(y, color='cyan', linewidth=1.5)
    for x in [1.5]:
        axes[0, 0].axvline(x, color='cyan', linewidth=1.5)

    _draw_matrix(axes[0, 1], max_pool, 'MaxPool 2x2 output', cmap='magma', vmin=0, vmax=1.0)

    _draw_matrix(axes[1, 0], adap_inp, 'AdaptiveAvgPool input', cmap='magma', vmin=0, vmax=1.0)
    _draw_matrix(axes[1, 1], adaptive_avg, 'AdaptiveAvgPool(1) output', cmap='magma', vmin=0, vmax=1.0)

    fig.suptitle('Pooling toy examples', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'toy_pooling.png'), dpi=180)
    plt.close(fig)


def plot_bilinear_interpolation_toy(results_dir=RESULTS_DIR):
    mat = np.array([
        [0.0, 0.2, 0.4, 0.1, 0.0],
        [0.1, 0.5, 0.8, 0.3, 0.1],
        [0.2, 0.7, 1.0, 0.4, 0.2],
        [0.0, 0.3, 0.6, 0.2, 0.0],
        [0.0, 0.1, 0.2, 0.1, 0.0],
    ], dtype=float)

    y, x = 2.2, 1.7
    value, weights = _bilinear_sample(mat, y, x)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    _draw_matrix(axes[0], mat, 'Fractional sampling point', cmap='magma', vmin=0, vmax=1.0)
    axes[0].scatter([x], [y], c='cyan', s=80)
    for (yy, xx), w in weights.items():
        axes[0].scatter([xx], [yy], c='white', s=40)
        axes[0].text(xx + 0.1, yy - 0.15, f'w={w:.2f}', color='white', fontsize=8)

    axes[1].axis('off')
    axes[1].set_title('Bilinear interpolation')
    axes[1].text(
        0.0,
        0.95,
        '\n'.join([
            f'point = ({y:.1f}, {x:.1f})',
            '',
            'interpolated value =',
            f'{value:.3f}',
            '',
            'It is a weighted sum of the 4 neighbors,',
            'which is why deformable conv can sample',
            'at non-integer coordinates.',
        ]),
        va='top',
        family='monospace',
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'toy_bilinear_interp.png'), dpi=180)
    plt.close(fig)


def run_toy_layer_visualizations(results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    plot_standard_convolution_toy(results_dir)
    plot_deformable_convolution_toy(results_dir)
    plot_pooling_toy(results_dir)
    plot_bilinear_interpolation_toy(results_dir)


if __name__ == '__main__':
    run_toy_layer_visualizations()
