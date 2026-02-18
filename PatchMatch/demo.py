"""
Experiments
1: NNF + NNF visualization + reconstruction
2: patchmatch convergence (RMS per iteration)
3: multiscale inpainting search + vote 
4: Root Mean Square (RMS) - patchmatch vs ground truth
5: effect of patch size on quality and time
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import patchmatch as pm

# utility functions
def load_rgb(path, max_side=240):
    """Loads RGB image and resizes if necessary"""
    img = Image.open(path).convert("RGB")
    ratio = max_side / max(img.size)
    if ratio < 1:
        img = img.resize((int(img.width * ratio), int(img.height * ratio)),
                         Image.LANCZOS)
    return np.array(img)


def create_test_images():
    """Synthetic images in case there are none in input"""
    size = 120
    A = np.zeros((size, size, 3), dtype=np.uint8)
    B = np.zeros((size, size, 3), dtype=np.uint8)
    A[10:60, 10:60] = [220, 60, 60]
    A[60:110, 60:110] = [60, 60, 220]
    B[30:80, 30:80] = [60, 200, 60]
    B[20:70, 70:120] = [220, 200, 60]
    return A, B


def rms_from_nnd(nnd):
    """RMS = sqrt(mean MSE)"""
    return float(np.sqrt(np.maximum(nnd.mean(), 0.0)))


# experiment 1
def exp1(A, B, ps=5, iters=6):
    print("Experiment 1: NNF + NNF visualization + reconstruction")

    t0 = time.time()
    nnf, nnd = pm.patchmatch(A, B, patch_size=ps, iterations=iters,
                              alpha=0.5, attempts=2, verbose=False)
    t = time.time() - t0
    print(f"  time={t:.2f}s | RMS={rms_from_nnd(nnd):.4f}")

    rec = pm.reconstruct(nnf, B)
    nnf_vis = pm.visualize_nnf(nnf)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes[0, 0].imshow(A);       axes[0, 0].set_title("A (source)")
    axes[0, 1].imshow(B);       axes[0, 1].set_title("B (target)")
    axes[1, 0].imshow(nnf_vis); axes[1, 0].set_title("NNF (HSV: hue=dir, val=mag)")
    axes[1, 1].imshow(rec);     axes[1, 1].set_title("pixelwise reconstruction")
    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle(f"Experiment 1: patch_size={ps}, iters={iters}, RMS={rms_from_nnd(nnd):.4f}",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("ex1.png", dpi=150)
    plt.show()

    return nnf, nnd

# experiment 2
def exp2(A, B, ps=5, max_iters=10):
    print("Experiment 2: patchmatch convergence (RMS per iteration)")

    B_f = B.astype(np.float64)
    nnf, nnd, A_pad = pm.initialize(A, B, ps, seed=0)
    H, W = A.shape[:2]

    rms_hist = [rms_from_nnd(nnd)]   # iteration 0 = random init
    print(f"iter 0 (init): RMS={rms_hist[0]:.4f}")

    for it in range(1, max_iters + 1):
        is_odd = (it % 2 == 1)
        rows = range(H) if is_odd else range(H - 1, -1, -1)
        cols = range(W) if is_odd else range(W - 1, -1, -1)

        for i in rows:
            for j in cols:
                pm.propagate(nnf, nnd, A_pad, B_f, ps, i, j, is_odd)
                pm.random_search(nnf, nnd, A_pad, B_f, ps, i, j,
                                 alpha=0.5, attempts=2)

        rms = rms_from_nnd(nnd)
        rms_hist.append(rms)
        print(f"  iter {it}/{max_iters}: RMS={rms:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(range(0, max_iters + 1), rms_hist, "o-", markersize=5)
    plt.title("Experiment 2: patchmatch convergence (RMS per iteration)")
    plt.xlabel("iteration (0 = random init)")
    plt.ylabel("RMS = sqrt(mean MSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ex2.png", dpi=150)
    plt.show()


# experiment 3
def exp3(A, ps=5):
    print("Experiment 3: multiscale inpainting search + vote")

    H, W = A.shape[:2]
    ch, cw = H // 2, W // 2
    mh, mw = H // 5, W // 5
    mask = pm.create_rect_mask(A.shape,
                               ch - mh // 2, cw - mw // 2,
                               ch + mh // 2, cw + mw // 2)

    damaged = A.copy()
    damaged[mask] = 255

    t0 = time.time()
    out = pm.multiscale_inpainting(A, mask, patch_size=ps, num_scales=3,
                                   em_iters=6, pm_iters=3, verbose=True)
    elapsed = time.time() - t0
    print(f"  time={elapsed:.2f}s")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(A);       axes[0].set_title("original")
    axes[1].imshow(damaged);  axes[1].set_title("damaged with hole")
    axes[2].imshow(out);      axes[2].set_title(f"inpainting ({elapsed:.1f}s)")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Experiment 3: multiscale inpainting search + vote, patch_size={ps}", fontsize=13)
    plt.tight_layout()
    plt.savefig("ex3.png", dpi=150)
    plt.show()


# experiment 4
def _sample_grid_coords(h, w, stride=10, max_points=300, seed=0):
    """Samples coordinates on a regular grid (+ random subsample if there are too many)"""
    coords = [(i, j) for i in range(0, h, stride) for j in range(0, w, stride)]
    if max_points is not None and len(coords) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords = [coords[k] for k in idx]
    return np.array(coords, dtype=np.int32)


def _brute_force_gt_exact(A, B, ps=5, query_stride=10, max_queries=300, seed=0):
    """
    Ground truth for query subset:
      - query: grid on A (stride query_stride, max max_queries points)
      - candidates: all valid centers in B (stride=1, exhaustive search)

    For each query it finds the globally optinal match in B
    Complexity: O(queries x (BH-ps) x (BW-ps)) - slow but correct
    Returns coords (N,2) e gt_mse (N,).
    """
    H, W = A.shape[:2]
    BH, BW = B.shape[:2]
    p = ps // 2

    coords = _sample_grid_coords(H, W, stride=query_stride,
                                 max_points=max_queries, seed=seed)

    A_pad = np.full((H + 2 * p, W + 2 * p, 3), np.nan, dtype=np.float64)
    A_pad[p:H + p, p:W + p, :] = A.astype(np.float64)
    B_f = B.astype(np.float64)

    n_cand_r = BH - 2 * p   # valid centers for each row
    n_cand_c = BW - 2 * p   # valid centers for each column
    total_cand = n_cand_r * n_cand_c

    gt_mse = np.full(len(coords), np.inf, dtype=np.float64)

    t0 = time.time()
    total = len(coords)
    print(f"brute force: {total} query x {total_cand} candidates "
          f"= {total * total_cand:,} comparisons")

    for idx, (i, j) in enumerate(coords):
        a = np.array([i, j], dtype=np.int32)
        best = np.inf

        for br in range(p, BH - p):
            for bc in range(p, BW - p):
                b = np.array([br, bc], dtype=np.int32)
                d = pm.patch_distance(a, b, A_pad, B_f, ps)
                if d < best:
                    best = d

        gt_mse[idx] = best

        if (idx + 1) % 20 == 0 or idx == total - 1:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (total - idx - 1) / speed if speed > 0 else 0
            print(f"groubnd truth {idx + 1}/{total}  "
                  f"elapsed={elapsed:.1f}s  ETA~{eta:.0f}s")

    elapsed = time.time() - t0
    print(f"ground truth completed in {elapsed:.1f}s")
    return coords, gt_mse


def exp4(A, B, ps=5, pm_iters=6, query_stride=10, max_queries=300):
    print("Experiment 4: Root Mean Square (RMS) - (patchmatch vs ground truth)")

    # patchmatch
    t0 = time.time()
    nnf, nnd = pm.patchmatch(A, B, patch_size=ps, iterations=pm_iters,
                              alpha=0.5, attempts=2, verbose=False)
    t_pm = time.time() - t0

    # ground truth
    coords, gt_mse = _brute_force_gt_exact(
        A, B, ps=ps, query_stride=query_stride,
        max_queries=max_queries, seed=0
    )

    pm_mse = np.array([nnd[i, j] for (i, j) in coords], dtype=np.float64)

    gt_rms = np.sqrt(np.maximum(gt_mse, 0.0))
    pm_rms = np.sqrt(np.maximum(pm_mse, 0.0))

    ratio = float(pm_rms.mean() / (gt_rms.mean() + 1e-12))
    delta = pm_rms - gt_rms

    # sanity check: patchmatch >= ground truth because ground truth is the global minimum
    violations = int(np.sum(pm_rms < gt_rms - 1e-6))

    print(f"\nresults")
    print(f"patchmatch time = {t_pm:.2f}s ({pm_iters} iterazioni)")
    print(f"queries = {len(coords)} (stride={query_stride})")
    print(f"RMS mean: ground truth={gt_rms.mean():.4f}  |  patchmatch={pm_rms.mean():.4f}  |  "
          f"patchmatch/ground truth={ratio:.3f}")
    print(f"RMS median: ground truth={np.median(gt_rms):.4f}  |  patchmatch={np.median(pm_rms):.4f}")
    print(f"mean delta RMS (patchmatch−ground truth) = {delta.mean():.4f}  (must be >= 0)")
    print(f"violations patchmatch < ground truth: {violations}/{len(coords)}")


    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    # histogram
    axes[0].hist(gt_rms, bins=30, alpha=0.7, label="ground truth RMS")
    axes[0].hist(pm_rms, bins=30, alpha=0.7, label="patchmatch RMS")
    axes[0].set_title("RMS distribution")
    axes[0].set_xlabel("RMS")
    axes[0].set_ylabel("count")
    axes[0].legend()

    # scatterplot
    axes[1].scatter(gt_rms, pm_rms, s=12, alpha=0.5)
    lo = float(min(gt_rms.min(), pm_rms.min()))
    hi = float(max(gt_rms.max(), pm_rms.max()))
    axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    axes[1].set_title(f"patchmatch vs ground truth (patchmatch/ground truth = {ratio:.3f})")
    axes[1].set_xlabel("ground truth RMS")
    axes[1].set_ylabel("patchmatch RMS")
    axes[1].legend()

    # delta histogram
    axes[2].hist(delta, bins=30, alpha=0.7, color="tab:green")
    axes[2].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[2].set_title(f"delta(RMS) (patchmatch − ground truth), mean={delta.mean():.3f}")
    axes[2].set_xlabel("delta(RMS)")
    axes[2].set_ylabel("count")

    plt.suptitle(f"Experiment 4: Root Mean Square (RMS) - (patchmatch vs ground truth)"
                 f"(ps={ps}, {len(coords)} query)", fontsize=13)
    plt.tight_layout()
    plt.savefig("ex4.png", dpi=150)
    plt.show()

# experiment 5
def exp5(A, B, ps_list=None, iters=6):
    print("Experiment 5: patch size effect on quality (RMS)")

    if ps_list is None:
        ps_list = [3, 5, 7, 9]

    results = []
    reconstructions = []

    for ps in ps_list:
        t0 = time.time()
        nnf, nnd = pm.patchmatch(A, B, patch_size=ps, iterations=iters,
                                  alpha=0.5, attempts=2, verbose=False)
        elapsed = time.time() - t0
        rms = rms_from_nnd(nnd)
        rec = pm.reconstruct(nnf, B)
        results.append((ps, rms, elapsed))
        reconstructions.append(rec)
        print(f"  patch_size={ps:2d}  RMS={rms:.4f}  time={elapsed:.2f}s")

    ps_vals = [r[0] for r in results]
    rms_vals = [r[1] for r in results]

    # table
    print(f"\n{'PS':>4} | {'RMS':>8}")
    print(f"  {'----':>4}-+-{'--------':>8}")
    for ps, rms, _ in results:
        print(f"  {ps:>4} | {rms:>8.4f}")

    # rms vs patch size graph
    fig1, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ps_vals, rms_vals, width=1.2, color="steelblue", edgecolor="black", alpha=0.85)
    for x, y in zip(ps_vals, rms_vals):
        ax.text(x, y + 0.15, f"{y:.2f}", ha="center", fontsize=10)
    ax.set_xlabel("patch size")
    ax.set_ylabel("RMS = sqrt(mean MSE)")
    ax.set_title("patch size effect on matching quality")
    ax.set_xticks(ps_vals)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ex5_patch_size_rms.png", dpi=150)
    plt.show()

    # visual reconstruction
    n = len(ps_list)
    fig2, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    axes[0].imshow(A)
    axes[0].set_title("A (original)")
    axes[0].axis("off")
    for k, (ps, rec) in enumerate(zip(ps_list, reconstructions)):
        axes[k + 1].imshow(rec)
        axes[k + 1].set_title(f"ps={ps}  RMS={results[k][1]:.2f}")
        axes[k + 1].axis("off")
    plt.suptitle("reconstruction as patch size varies", fontsize=13)
    plt.tight_layout()
    plt.savefig("ex5_patch_size_visual.png", dpi=150)
    plt.show()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pathA = os.path.join(script_dir, "cup_a.jpg")
    pathB = os.path.join(script_dir, "cup_b.jpg")

    if os.path.exists(pathA) and os.path.exists(pathB):
        A = load_rgb(pathA, max_side=240)
        B = load_rgb(pathB, max_side=240)
        print(f"images: A={A.shape}, B={B.shape}")
    else:
        print("no images found (using synthetic images)")
        A, B = create_test_images()

    PS = 5

    exp1(A, B, ps=PS, iters=6)
    exp2(A, B, ps=PS, max_iters=10)
    exp3(A, ps=PS)
    exp4(A, B, ps=PS, pm_iters=6, query_stride=10, max_queries=300)
    exp5(A, B, ps_list=[3, 5, 7, 9], iters=6)


if __name__ == "__main__":
    main()
    