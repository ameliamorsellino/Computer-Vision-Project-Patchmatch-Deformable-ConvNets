"""
Experiments
1: NNF + NNF visualization + reconstruction
2: patchmatch convergence (RMS per iteration)
3: multiscale inpainting search + vote
4: Root Mean Square (RMS) - patchmatch vs ground truth
5: effect of patch size on quality and time
6: Toy example step-by-step
7: Failure cases
8: PatchMatch vs Lama
9: applications
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

import patchmatch as pm


# utility functions
def load_rgb(path, max_side=240):
    """Loads RGB image and resizes if necessary"""
    img = Image.open(path).convert("RGB")
    ratio = max_side / max(img.size)
    if ratio < 1:
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.LANCZOS
        )
    return np.array(img)


def rms_from_nnd(nnd):
    """RMS = sqrt(mean MSE)"""
    return float(np.sqrt(np.maximum(nnd.mean(), 0.0)))


def create_test_images():
    """Synthetic images in case there are none in input"""
    size = 120
    A = np.zeros((size, size, 3), dtype=np.uint8)
    B = np.zeros((size, size, 3), dtype=np.uint8)

    A[:] = [20, 20, 20]
    B[:] = [20, 20, 20]

    A[10:60, 10:60] = [220, 60, 60]
    A[60:110, 60:110] = [60, 60, 220]

    B[30:80, 30:80] = [60, 200, 60]
    B[20:70, 70:120] = [220, 200, 60]
    return A, B


def create_toy_translation_example(size=80, shift=(10, 14)):
    """
    B is a translation of the pattern A
    """
    H = W = size
    A = np.zeros((H, W, 3), dtype=np.uint8)
    A[:] = [25, 25, 25]

    # pattern
    cv2.rectangle(A, (8, 8), (28, 28), (255, 80, 80), -1)
    cv2.circle(A, (50, 20), 10, (80, 255, 80), -1)
    cv2.rectangle(A, (20, 45), (60, 60), (80, 80, 255), -1)
    cv2.line(A, (5, 70), (70, 65), (220, 220, 60), 3)

    dy, dx = shift
    B = np.zeros_like(A)
    B[:] = [25, 25, 25]

    src_r0 = 0
    src_r1 = H - dy
    src_c0 = 0
    src_c1 = W - dx
    dst_r0 = dy
    dst_r1 = H
    dst_c0 = dx
    dst_c1 = W

    B[dst_r0:dst_r1, dst_c0:dst_c1] = A[src_r0:src_r1, src_c0:src_c1]

    return A, B


def create_failure_case_images(size=140):
    """
    Some cases where PatchMatch / patch-based inpainting struggles:
    - ambiguous repetitive patterns
    - long geometric structures crossing the hole
    """
    # Case 1: repetitive texture
    rep = np.zeros((size, size, 3), dtype=np.uint8)
    tile = 10
    for i in range(0, size, tile):
        for j in range(0, size, tile):
            color = 210 if ((i // tile + j // tile) % 2 == 0) else 60
            rep[i:i + tile, j:j + tile] = [color, color, color]

    # Case 2: diagonal line / long structure
    line = np.ones((size, size, 3), dtype=np.uint8) * 235
    cv2.line(line, (10, 20), (size - 20, size - 10), (20, 20, 20), 4)
    cv2.line(line, (20, size - 25), (size - 25, 25), (40, 90, 220), 4)

    return rep, line


def create_realistic_texture_example(size=160):
    """
    Texture example to show where patch based method works best
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = [160, 180, 205]

    for _ in range(1400):
        r = np.random.randint(0, size)
        c = np.random.randint(0, size)
        img[r, c] = [
            20 + np.random.randint(40),
            120 + np.random.randint(80),
            20 + np.random.randint(40)
        ]

    cv2.circle(img, (size // 2, size // 2), 28, (210, 80, 80), -1)
    cv2.rectangle(img, (15, 15), (50, 50), (240, 230, 120), -1)

    return img


def mse_on_mask(gt, pred, mask):
    if not mask.any():
        return 0.0
    diff = gt[mask].astype(np.float64) - pred[mask].astype(np.float64)
    return float(np.mean(diff * diff))

def make_text_placeholder(shape, lines, bg=235, fg=(30, 30, 30)):
    """
    Creates a placeholder image with text, used when LaMa output is missing.
    """
    H, W = shape[:2]
    img = np.ones((H, W, 3), dtype=np.uint8) * bg

    if isinstance(lines, str):
        lines = [lines]

    y = 25
    for line in lines:
        cv2.putText(
            img,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            fg,
            1,
            cv2.LINE_AA
        )
        y += 22

    return img


def load_lama_result_or_placeholder(case_name, shape, lama_dir="lama_outputs"):
    """
    Loads a precomputed LaMa result from disk.
    Expected filenames (tries in this order):
      - lama_outputs/<case_name>_lama.png
      - lama_outputs/<case_name>.png

    If missing, returns a placeholder image.
    """
    candidates = [
        os.path.join(lama_dir, f"{case_name}_lama.png"),
        os.path.join(lama_dir, f"{case_name}.png"),
    ]

    for path in candidates:
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            img = img.resize((shape[1], shape[0]), Image.LANCZOS)
            return np.array(img), path

    placeholder = make_text_placeholder(
        shape,
        [
            "LaMa output not found.",
            f"Expected file in: {lama_dir}/",
            f"Try: {case_name}_lama.png",
            "or",
            f"{case_name}.png"
        ]
    )
    return placeholder, None

# experiment 1
def exp1(A, B, ps=5, iters=6):
    print("Experiment 1: NNF + NNF visualization + reconstruction")

    t0 = time.time()
    nnf, nnd = pm.patchmatch(
        A, B, patch_size=ps, iterations=iters,
        alpha=0.5, attempts=2, verbose=False
    )
    elapsed = time.time() - t0
    print(f"  time={elapsed:.2f}s | RMS={rms_from_nnd(nnd):.4f}")

    rec_pix = pm.reconstruct(nnf, B)
    rec_patch = pm.reconstruct_from_patches(nnf, B, patch_size=ps)
    nnf_vis = pm.visualize_nnf(nnf)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(A)
    axes[0, 0].set_title("A (query/source)")
    axes[0, 1].imshow(B)
    axes[0, 1].set_title("B (target)")
    axes[0, 2].imshow(nnf_vis)
    axes[0, 2].set_title("NNF visualization")

    axes[1, 0].imshow(rec_pix)
    axes[1, 0].set_title("reconstruction pixel-wise")
    axes[1, 1].imshow(rec_patch)
    axes[1, 1].set_title("reconstruction patch average")
    axes[1, 2].imshow(np.sqrt(np.maximum(nnd, 0.0)), cmap="magma")
    axes[1, 2].set_title("RMS map")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(f"Experiment 1 - ps={ps}, iters={iters}, RMS={rms_from_nnd(nnd):.4f}")
    plt.tight_layout()
    plt.savefig("ex1_overview.png", dpi=150)
    plt.show()

    return nnf, nnd


# experiment 2
def exp2(A, B, ps=5, max_iters=10):
    print("Experiment 2: patchmatch convergence (RMS per iteration)")

    B_f = B.astype(np.float64)
    nnf, nnd, A_pad = pm.initialize(A, B, ps, seed=0)
    H, W = A.shape[:2]

    rms_hist = [rms_from_nnd(nnd)]
    print(f"iter 0 (init): RMS={rms_hist[0]:.4f}")

    for it in range(1, max_iters + 1):
        is_odd = (it % 2 == 1)
        rows = range(H) if is_odd else range(H - 1, -1, -1)
        cols = range(W) if is_odd else range(W - 1, -1, -1)

        for i in rows:
            for j in cols:
                pm.propagate(nnf, nnd, A_pad, B_f, ps, i, j, is_odd)
                pm.random_search(
                    nnf, nnd, A_pad, B_f, ps, i, j,
                    alpha=0.5, attempts=2
                )

        rms = rms_from_nnd(nnd)
        rms_hist.append(rms)
        print(f"  iter {it}/{max_iters}: RMS={rms:.4f}")

    plt.figure(figsize=(7, 4))
    plt.plot(range(0, max_iters + 1), rms_hist, "o-", markersize=5)
    plt.title("Experiment 2: patchmatch convergence")
    plt.xlabel("iteration (0 = random init)")
    plt.ylabel("RMS = sqrt(mean MSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ex2_convergence.png", dpi=150)
    plt.show()


# experiment 3
def exp3(A, ps=5):
    print("Experiment 3: multiscale inpainting search + vote")

    H, W = A.shape[:2]
    ch, cw = H // 2, W // 2
    mh, mw = H // 5, W // 5

    mask = pm.create_rect_mask(
        A.shape,
        ch - mh // 2, cw - mw // 2,
        ch + mh // 2, cw + mw // 2
    )

    damaged = A.copy()
    damaged[mask] = 255

    t0 = time.time()
    out = pm.multiscale_inpainting(
        A, mask, patch_size=ps, num_scales=3,
        em_iters=6, pm_iters=3, verbose=True
    )
    elapsed = time.time() - t0
    print(f"  time={elapsed:.2f}s")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(A)
    axes[0].set_title("original")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("mask")
    axes[2].imshow(damaged)
    axes[2].set_title("damaged")
    axes[3].imshow(out)
    axes[3].set_title(f"patchmatch inpainting ({elapsed:.1f}s)")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(f"Experiment 3: multiscale inpainting, ps={ps}", fontsize=13)
    plt.tight_layout()
    plt.savefig("ex3_inpainting.png", dpi=150)
    plt.show()

    return mask, damaged, out


# experiment 4
def _sample_grid_coords(h, w, stride=10, max_points=300, seed=0):
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

    For each query it finds the globally optimal match in B.
    Complexity: O(queries x (BH-ps) x (BW-ps)) - slow but correct
    Returns coords (N,2) and gt_mse (N,)
    """
    H, W = A.shape[:2]
    BH, BW = B.shape[:2]
    p = ps // 2

    coords = _sample_grid_coords(
        H, W, stride=query_stride,
        max_points=max_queries, seed=seed
    )

    A_pad = np.full((H + 2 * p, W + 2 * p, 3), np.nan, dtype=np.float64)
    A_pad[p:H + p, p:W + p, :] = A.astype(np.float64)
    B_f = B.astype(np.float64)

    n_cand_r = BH - 2 * p
    n_cand_c = BW - 2 * p
    total_cand = n_cand_r * n_cand_c

    gt_mse = np.full(len(coords), np.inf, dtype=np.float64)

    t0 = time.time()
    total = len(coords)
    print(f"brute force: {total} query x {total_cand} candidates = {total * total_cand:,} comparisons")

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
            print(f"ground truth {idx + 1}/{total} elapsed={elapsed:.1f}s ETA~{eta:.0f}s")

    elapsed = time.time() - t0
    print(f"ground truth completed in {elapsed:.1f}s")
    return coords, gt_mse


def exp4(A, B, ps=5, pm_iters=6, query_stride=10, max_queries=300):
    print("Experiment 4: Root Mean Square (RMS) - (patchmatch vs ground truth)")

    # patchmatch
    t0 = time.time()
    nnf, nnd = pm.patchmatch(
        A, B, patch_size=ps, iterations=pm_iters,
        alpha=0.5, attempts=2, verbose=False
    )
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

    print("\nresults")
    print(f"patchmatch time = {t_pm:.2f}s ({pm_iters} iterations)")
    print(f"queries = {len(coords)} (stride={query_stride})")
    print(f"RMS mean: gt={gt_rms.mean():.4f} | pm={pm_rms.mean():.4f} | ratio={ratio:.3f}")
    print(f"RMS median: gt={np.median(gt_rms):.4f} | pm={np.median(pm_rms):.4f}")
    print(f"mean delta RMS (pm-gt) = {delta.mean():.4f}")
    print(f"violations pm < gt: {violations}/{len(coords)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # histogram
    axes[0].hist(gt_rms, bins=30, alpha=0.7, label="ground truth RMS")
    axes[0].hist(pm_rms, bins=30, alpha=0.7, label="patchmatch RMS")
    axes[0].set_title("RMS distribution")
    axes[0].legend()

    # scatterplot
    axes[1].scatter(gt_rms, pm_rms, s=12, alpha=0.5)
    lo = float(min(gt_rms.min(), pm_rms.min()))
    hi = float(max(gt_rms.max(), pm_rms.max()))
    axes[1].plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    axes[1].set_title(f"PatchMatch vs GT (ratio={ratio:.3f})")
    axes[1].set_xlabel("GT RMS")
    axes[1].set_ylabel("PM RMS")

    # delta histogram
    axes[2].hist(delta, bins=30, alpha=0.7, color="tab:green")
    axes[2].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[2].set_title(f"Delta RMS (PM-GT), mean={delta.mean():.3f}")

    plt.suptitle(f"Experiment 4 - PatchMatch vs ground truth (ps={ps})", fontsize=13)
    plt.tight_layout()
    plt.savefig("ex4_patchmatch_vs_gt.png", dpi=150)
    plt.show()


# experiment 5
def exp5(A, B, ps_list=None, iters=6):
    print("Experiment 5: patch size effect on quality")

    if ps_list is None:
        ps_list = [3, 5, 7, 9]

    results = []
    reconstructions = []

    for ps in ps_list:
        t0 = time.time()
        nnf, nnd = pm.patchmatch(
            A, B, patch_size=ps, iterations=iters,
            alpha=0.5, attempts=2, verbose=False
        )
        elapsed = time.time() - t0
        rms = rms_from_nnd(nnd)
        rec = pm.reconstruct_from_patches(nnf, B, patch_size=ps)
        results.append((ps, rms, elapsed))
        reconstructions.append(rec)
        print(f"  patch_size={ps:2d}  RMS={rms:.4f}  time={elapsed:.2f}s")

    ps_vals = [r[0] for r in results]
    rms_vals = [r[1] for r in results]

    print(f"\n{'PS':>4} | {'RMS':>8} | {'time(s)':>8}")
    for ps, rms, elapsed in results:
        print(f"  {ps:>4} | {rms:>8.4f} | {elapsed:>8.2f}")

    fig1, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ps_vals, rms_vals, width=1.2, color="steelblue", edgecolor="black", alpha=0.85)
    for x, y in zip(ps_vals, rms_vals):
        ax.text(x, y + 0.15, f"{y:.2f}", ha="center", fontsize=10)
    ax.set_xlabel("patch size")
    ax.set_ylabel("RMS")
    ax.set_title("patch size effect on quality")
    ax.set_xticks(ps_vals)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ex5_patch_size_rms.png", dpi=150)
    plt.show()

    n = len(ps_list)
    fig2, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
    axes[0].imshow(A)
    axes[0].set_title("A")
    axes[0].axis("off")

    for k, (ps, rec) in enumerate(zip(ps_list, reconstructions)):
        axes[k + 1].imshow(rec)
        axes[k + 1].set_title(f"ps={ps}  RMS={results[k][1]:.2f}")
        axes[k + 1].axis("off")

    plt.suptitle("Reconstruction vs patch size", fontsize=13)
    plt.tight_layout()
    plt.savefig("ex5_patch_size_visual.png", dpi=150)
    plt.show()


# experiment 6
def exp6_toy_step_by_step(ps=5, iters=5):
    print("Experiment 6: toy example visual step by step")

    A, B = create_toy_translation_example(size=80, shift=(10, 14))
    nnf_list, nnd_list, rms_list = pm.patchmatch_history(
        A, B, patch_size=ps, iterations=iters,
        alpha=0.5, attempts=2, seed=0
    )

    ncols = len(nnf_list)
    fig, axes = plt.subplots(3, ncols, figsize=(3.4 * ncols, 9))

    for k in range(ncols):
        nnf_k = nnf_list[k]
        nnd_k = nnd_list[k]

        rec = pm.reconstruct_from_patches(nnf_k, B, patch_size=ps)
        nnf_vis = pm.visualize_nnf(nnf_k)
        rms_map = np.sqrt(np.maximum(nnd_k, 0.0))

        axes[0, k].imshow(rec)
        axes[0, k].set_title(f"rec iter {k}\nRMS={rms_list[k]:.2f}")

        axes[1, k].imshow(nnf_vis)
        axes[1, k].set_title(f"NNF iter {k}")

        axes[2, k].imshow(rms_map, cmap="magma")
        axes[2, k].set_title(f"RMS map iter {k}")

        for r in range(3):
            axes[r, k].axis("off")

    fig.suptitle("Toy example: step-by-step evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig("ex6_toy_step_by_step.png", dpi=150)
    plt.show()

    fig_curve, ax_curve = plt.subplots(figsize=(7, 4))
    ax_curve.plot(range(len(rms_list)), rms_list, "o-", lw=2)
    ax_curve.set_title("Toy example: RMS evolution")
    ax_curve.set_xlabel("iteration")
    ax_curve.set_ylabel("RMS")
    ax_curve.grid(True, alpha=0.3)
    fig_curve.tight_layout()
    fig_curve.savefig("ex6_toy_rms_curve.png", dpi=150)
    plt.show()

    fig2, ax2 = plt.subplots(1, 2, figsize=(8, 4))
    ax2[0].imshow(A)
    ax2[0].set_title("Toy A")
    ax2[1].imshow(B)
    ax2[1].set_title("Toy B")

    for ax in ax2:
        ax.axis("off")

    fig2.tight_layout()
    fig2.savefig("ex6_toy_inputs.png", dpi=150)
    plt.show()


# experiment 7
def exp7_failure_cases(ps=7):
    print("Experiment 7: failure cases")

    rep, line = create_failure_case_images()

    examples = [
        ("repetitive_texture", rep),
        ("long_structure", line),
    ]

    for name, img in examples:
        H, W = img.shape[:2]
        mask = pm.create_rect_mask(img.shape, H // 3, W // 3, 2 * H // 3, 2 * W // 3)

        damaged = img.copy()
        damaged[mask] = 255

        out_pm = pm.multiscale_inpainting(
            img, mask, patch_size=ps, num_scales=3,
            em_iters=6, pm_iters=3, verbose=False, seed=0
        )

        mse_pm = mse_on_mask(img, out_pm, mask)

        print(f"\n{name}")
        print(f"  hole MSE - PatchMatch: {mse_pm:.2f}")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        axes[0].imshow(img)
        axes[0].set_title("original")
        axes[1].imshow(damaged)
        axes[1].set_title("damaged")
        axes[2].imshow(out_pm)
        axes[2].set_title(f"PatchMatch\nMSE={mse_pm:.1f}")

        for ax in axes:
            ax.axis("off")

        plt.suptitle(f"Failure case: {name}", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"ex7_failure_{name}.png", dpi=150)
        plt.show()


# experiment 8
def exp10_compare_patchmatch_vs_lama(ps=7, lama_dir="lama_outputs"):
    print("Experiment 8: PatchMatch vs LaMa")

    img = create_realistic_texture_example(size=160)

    masks = [
        ("small_hole", pm.create_rect_mask(img.shape, 55, 55, 85, 85)),
        ("large_hole", pm.create_rect_mask(img.shape, 40, 40, 120, 120)),
    ]

    os.makedirs(lama_dir, exist_ok=True)

    for name, mask in masks:
        damaged = img.copy()
        damaged[mask] = 255

        # Save inputs so you can run LaMa externally (Colab / another env)
        input_path = os.path.join(lama_dir, f"{name}_input.png")
        mask_path = os.path.join(lama_dir, f"{name}_mask.png")
        Image.fromarray(damaged).save(input_path)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)

        # PatchMatch result
        t0 = time.time()
        out_pm = pm.multiscale_inpainting(
            img, mask, patch_size=ps, num_scales=3,
            em_iters=6, pm_iters=3, verbose=False, seed=0
        )
        t_pm = time.time() - t0
        mse_pm = mse_on_mask(img, out_pm, mask)

        # LaMa result loaded from disk
        out_lama, lama_path = load_lama_result_or_placeholder(name, img.shape, lama_dir=lama_dir)
        mse_lama = None if lama_path is None else mse_on_mask(img, out_lama, mask)

        print(f"\n{name}")
        print(f"  PatchMatch: MSE={mse_pm:.2f}, time={t_pm:.2f}s")
        if lama_path is not None:
            print(f"  LaMa:       MSE={mse_lama:.2f}  | loaded from: {lama_path}")
        else:
            print(f"  LaMa:       NOT FOUND")
            print(f"              saved input -> {input_path}")
            print(f"              saved mask  -> {mask_path}")
            print(f"              put LaMa output in {lama_dir}/{name}_lama.png")

        # Plot
        if lama_path is not None:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))

        axes[0].imshow(img)
        axes[0].set_title("ground truth")

        axes[1].imshow(damaged)
        axes[1].set_title("damaged")

        axes[2].imshow(out_pm)
        axes[2].set_title(f"PatchMatch\nMSE={mse_pm:.1f}")

        axes[3].imshow(out_lama)
        if mse_lama is None:
            axes[3].set_title("LaMa\nmissing output")
        else:
            axes[3].set_title(f"LaMa\nMSE={mse_lama:.1f}")

        for ax in axes:
            ax.axis("off")

        plt.suptitle(f"Experiment 8: PatchMatch vs LaMa ({name})", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"ex8_patchmatch_vs_lama_{name}.png", dpi=150)
        plt.show()

# experiment 9
def exp9_applications_panel(ps=5):
    print("Experiment 9: applications")

    img = create_realistic_texture_example(size=160)

    masks = [
        ("object_removal", pm.create_rect_mask(img.shape, 55, 55, 95, 105)),
        ("scratch_repair", pm.create_rect_mask(img.shape, 20, 78, 140, 84)),
        ("logo_removal", pm.create_rect_mask(img.shape, 10, 10, 45, 45)),
    ]

    fig, axes = plt.subplots(len(masks), 3, figsize=(12, 4 * len(masks)))

    if len(masks) == 1:
        axes = np.array([axes])

    for r, (title, mask) in enumerate(masks):
        damaged = img.copy()
        damaged[mask] = 255

        out = pm.multiscale_inpainting(
            img, mask, patch_size=ps, num_scales=3,
            em_iters=5, pm_iters=3, verbose=False, seed=0
        )

        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{title} - original")
        axes[r, 1].imshow(damaged)
        axes[r, 1].set_title("damaged")
        axes[r, 2].imshow(out)
        axes[r, 2].set_title("restored")

        for c in range(3):
            axes[r, c].axis("off")

    plt.suptitle("Possible applications of PatchMatch-based inpainting", fontsize=14)
    plt.tight_layout()
    plt.savefig("ex9_applications.png", dpi=150)
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
        print("No input images found, using synthetic images.")
        A, B = create_test_images()

    RUN_ALL = False

    if RUN_ALL:
        PS = 5
        exp1(A, B, ps=PS, iters=6)
        exp2(A, B, ps=PS, max_iters=10)
        exp3(A, ps=PS)
        exp4(A, B, ps=PS, pm_iters=6, query_stride=10, max_queries=120)
        exp5(A, B, ps_list=[3, 5, 7, 9], iters=6)
        exp6_toy_step_by_step(ps=5, iters=5)
        exp7_failure_cases(ps=7)
        exp10_compare_patchmatch_vs_lama(ps=7, lama_dir="lama_outputs")
        exp9_applications_panel(ps=5)
    else:
        exp10_compare_patchmatch_vs_lama(ps=7, lama_dir="lama_outputs")

if __name__ == "__main__":
    main()
