"""
patchmatch + inpainting multiscale search + vote

functions
  1: patch_distance
  2: initialize
  3: propagate
  4: random_search
  5: patchmatch
  6: reconstruct
  7: visualize_nnf

inpainting
  - multiscale_inpainting (main function)
  - internal helpers (prefix _)

notes on metrics
  - patch_distance returns normalized MSE (mean over pixels x channels)
  - RMS (in the paper) = sqrt(MSE)
"""

import numpy as np
import cv2
from scipy.ndimage import binary_erosion, distance_transform_edt


def patch_distance(a, b, A_pad, B, patch_size):
    """
    Normalized MSE between the patch centered at a (in A) and the patch centered at b (in B)
    Coordinates
        - a = (row, col) in the original A. With p-pixel padding A_pad[a[0] : a[0]+patch_size, ...] extracts the patch centered at a
        - b = (row, col) patch center in B (without padding)

    NaN pixels in A_pad, so the border, are ignored
    """
    p = patch_size // 2
    pa = A_pad[a[0]:a[0] + patch_size, a[1]:a[1] + patch_size, :]
    pb = B[b[0] - p:b[0] + p + 1, b[1] - p:b[1] + p + 1, :]

    diff = pb - pa
    valid = np.sum(~np.isnan(diff))
    if valid == 0:
        return np.inf
    return np.nansum(diff * diff) / valid


def initialize(A, B, patch_size, seed=None):
    """
    Initializes a random NNF and NND
        - nnf[i,j] = (row, col) center of the best match in B for pixel (i,j) of A
        - nnd[i,j] = corresponding normalized MSE
        - A_pad: A padded with p pixels of NaNs to handle borders

    Centers in B are constrained to [p, BH-p-1] × [p, BW-p-1] so that the extracted patch stays inside B
    """
    if seed is not None:
        np.random.seed(seed)

    H, W = A.shape[:2]
    BH, BW = B.shape[:2]
    p = patch_size // 2

    nnf = np.zeros((H, W, 2), dtype=np.int32)
    nnf[:, :, 0] = np.random.randint(p, BH - p, size=(H, W))
    nnf[:, :, 1] = np.random.randint(p, BW - p, size=(H, W))

    A_pad = np.full((H + 2 * p, W + 2 * p, 3), np.nan, dtype=np.float64)
    A_pad[p:H + p, p:W + p, :] = A.astype(np.float64)

    B_f = B.astype(np.float64)
    nnd = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            a = np.array([i, j], dtype=np.int32)
            b = nnf[i, j]
            nnd[i, j] = patch_distance(a, b, A_pad, B_f, patch_size)

    return nnf, nnd, A_pad


def propagate(nnf, nnd, A_pad, B_f, patch_size, x, y, is_odd):
    """
    PatchMatch propagation 
    - on odd iterations it checks the neighbor above and to the left (shifted +1) 
    - on even iterations it checks the neighbor below and to the right (shifted −1).
    
    It updates nnf and nnd in-place
    """
    H = A_pad.shape[0] - patch_size + 1
    W = A_pad.shape[1] - patch_size + 1
    BH, BW = B_f.shape[:2]
    p = patch_size // 2

    best_d = nnd[x, y]
    a = np.array([x, y], dtype=np.int32)

    candidates = []

    if is_odd:
        if x > 0:
            c = nnf[x - 1, y].copy()
            c[0] = min(c[0] + 1, BH - p - 1)
            candidates.append(c)
        if y > 0:
            c = nnf[x, y - 1].copy()
            c[1] = min(c[1] + 1, BW - p - 1)
            candidates.append(c)
    else:
        if x < H - 1:
            c = nnf[x + 1, y].copy()
            c[0] = max(c[0] - 1, p)
            candidates.append(c)
        if y < W - 1:
            c = nnf[x, y + 1].copy()
            c[1] = max(c[1] - 1, p)
            candidates.append(c)

    for c in candidates:
        c[0] = np.clip(c[0], p, BH - p - 1)
        c[1] = np.clip(c[1], p, BW - p - 1)
        d = patch_distance(a, c, A_pad, B_f, patch_size)
        if d < best_d:
            best_d = d
            nnf[x, y] = c
            nnd[x, y] = d


def random_search(nnf, nnd, A_pad, B_f, patch_size, x, y, alpha=0.5, attempts=2):
    """
    Random search with an exponentially shrinking window (factor alpha)
    It generates random candidates in progressively smaller windows around the current match
    """
    BH, BW = B_f.shape[:2]
    p = patch_size // 2
    a = np.array([x, y], dtype=np.int32)

    bx, by = nnf[x, y]
    radius = max(BH, BW)

    while radius >= 1:
        r0 = int(max(bx - radius, p))
        r1 = int(min(bx + radius, BH - p - 1))
        c0 = int(max(by - radius, p))
        c1 = int(min(by + radius, BW - p - 1))

        if r0 < r1 and c0 < c1:
            for _ in range(attempts):
                rb = np.array(
                    [np.random.randint(r0, r1 + 1), np.random.randint(c0, c1 + 1)],
                    dtype=np.int32
                )
                d = patch_distance(a, rb, A_pad, B_f, patch_size)
                if d < nnd[x, y]:
                    nnd[x, y] = d
                    nnf[x, y] = rb
                    bx, by = int(rb[0]), int(rb[1])

        radius = int(radius * alpha)


def patchmatch(A, B, patch_size=5, iterations=5, alpha=0.5, attempts=2,
               seed=None, verbose=False):
    """
    Complete patchmtch: init + propagation iterations and random search
    Returns nnf (H,W,2) and nnd (H,W)
    """
    nnf, nnd, A_pad = initialize(A, B, patch_size, seed=seed)
    B_f = B.astype(np.float64)
    H, W = A.shape[:2]

    for it in range(1, iterations + 1):
        is_odd = (it % 2 == 1)
        rows = range(H) if is_odd else range(H - 1, -1, -1)
        cols = range(W) if is_odd else range(W - 1, -1, -1)

        for i in rows:
            for j in cols:
                propagate(nnf, nnd, A_pad, B_f, patch_size, i, j, is_odd)
                random_search(
                    nnf, nnd, A_pad, B_f, patch_size, i, j,
                    alpha=alpha, attempts=attempts
                )

        if verbose:
            print(f"  it {it}/{iterations}  "
                  f"MSE mean={nnd.mean():.4f}  RMS={np.sqrt(nnd.mean()):.4f}")

    return nnf, nnd


def patchmatch_history(A, B, patch_size=5, iterations=5, alpha=0.5, attempts=2,
                       seed=None):
    """
    Patchmatch version that saves history:
    - nnf_list
    - nnd_list
    - rms_list
    """
    nnf, nnd, A_pad = initialize(A, B, patch_size, seed=seed)
    B_f = B.astype(np.float64)
    H, W = A.shape[:2]

    nnf_list = [nnf.copy()]
    nnd_list = [nnd.copy()]
    rms_list = [float(np.sqrt(np.maximum(nnd.mean(), 0.0)))]

    for it in range(1, iterations + 1):
        is_odd = (it % 2 == 1)
        rows = range(H) if is_odd else range(H - 1, -1, -1)
        cols = range(W) if is_odd else range(W - 1, -1, -1)

        for i in rows:
            for j in cols:
                propagate(nnf, nnd, A_pad, B_f, patch_size, i, j, is_odd)
                random_search(
                    nnf, nnd, A_pad, B_f, patch_size, i, j,
                    alpha=alpha, attempts=attempts
                )

        nnf_list.append(nnf.copy())
        nnd_list.append(nnd.copy())
        rms_list.append(float(np.sqrt(np.maximum(nnd.mean(), 0.0))))

    return nnf_list, nnd_list, rms_list


def reconstruct(nnf, B):
    """Pixel wise reconstruction: for each (i,j) in A, copy B[nnf[i,j]]"""
    H, W = nnf.shape[:2]
    out = np.zeros((H, W, 3), dtype=B.dtype)
    for i in range(H):
        for j in range(W):
            bi, bj = nnf[i, j]
            out[i, j] = B[bi, bj]
    return out


def reconstruct_from_patches(nnf, B, patch_size=5):
    H, W = nnf.shape[:2]
    p = patch_size // 2

    acc = np.zeros((H, W, 3), dtype=np.float64)
    wgt = np.zeros((H, W), dtype=np.float64)

    for i in range(H):
        for j in range(W):
            br, bc = nnf[i, j]
            for di in range(-p, p + 1):
                for dj in range(-p, p + 1):
                    ai, aj = i + di, j + dj
                    bi, bj = br + di, bc + dj
                    if 0 <= ai < H and 0 <= aj < W and 0 <= bi < B.shape[0] and 0 <= bj < B.shape[1]:
                        acc[ai, aj] += B[bi, bj].astype(np.float64)
                        wgt[ai, aj] += 1.0

    out = np.zeros((H, W, 3), dtype=np.float64)
    valid = wgt > 1e-12
    out[valid] = acc[valid] / wgt[valid, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def visualize_nnf(nnf):
    """
    NNF visualization as HSV color map:
        - Hue = offset direction (angle)
        - Value = offset magnitude (normalized)
        - Saturation = 1
    """
    H, W = nnf.shape[:2]
    grid_y, grid_x = np.mgrid[:H, :W]
    dy = nnf[:, :, 0].astype(np.float64) - grid_y
    dx = nnf[:, :, 1].astype(np.float64) - grid_x

    angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag = mag / (mag.max() + 1e-8)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle * 179).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = (mag * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Inpainting
def create_rect_mask(shape, r0, c0, r1, c1):
    """Creates rectangular boolean mask"""
    mask = np.zeros(shape[:2], dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def _build_pyramid(img, mask, num_scales):
    """Gaussian pyramid (downscale x2) for img and mask"""
    imgs, masks = [img], [mask]
    for _ in range(1, num_scales):
        h = imgs[-1].shape[0] // 2
        w = imgs[-1].shape[1] // 2
        if h < 12 or w < 12:
            break
        imgs.append(cv2.resize(imgs[-1], (w, h), interpolation=cv2.INTER_AREA))
        masks.append(
            cv2.resize(masks[-1].astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST) > 0
        )
    return imgs[::-1], masks[::-1]


def _init_fill_nearest(img, mask):
    """Initial filling of the hole with the known nearest pixel (distance transform)"""
    if not mask.any():
        return img.copy()
    out = img.copy()
    _, idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
    out[mask] = img[idx[0][mask], idx[1][mask]]
    return out


def _valid_source_centers(mask, patch_size):
    """Returns boolean map: true where the center of a patch patch_sizexpatch_size falls entirely in the known region (c.ca mask)"""
    known = ~mask
    struct = np.ones((patch_size, patch_size), dtype=bool)
    valid = binary_erosion(known, structure=struct, border_value=False)
    return valid


def _patch_distance_masked(img, source, a_r, a_c, b_r, b_c, patch_size, conf):
    """
    Distance for inpainting: confidence weighted MSE, where known pixels weigh more
        - img: current image (hole filled progressively)
        - source: source image (known pixels + updated hole)
        - conf: confidence map in [0,1]
    """
    h, w = img.shape[:2]
    p = patch_size // 2

    ar0 = max(a_r - p, 0)
    ar1 = min(a_r + p + 1, h)
    ac0 = max(a_c - p, 0)
    ac1 = min(a_c + p + 1, w)

    dr0 = ar0 - (a_r - p)
    dc0 = ac0 - (a_c - p)
    dr1 = dr0 + (ar1 - ar0)
    dc1 = dc0 + (ac1 - ac0)

    br0 = b_r - p + dr0
    br1 = b_r - p + dr1
    bc0 = b_c - p + dc0
    bc1 = b_c - p + dc1

    if br0 < 0 or bc0 < 0 or br1 > h or bc1 > w:
        return np.inf

    pa = img[ar0:ar1, ac0:ac1].astype(np.float64)
    pb = source[br0:br1, bc0:bc1].astype(np.float64)
    wgt = conf[ar0:ar1, ac0:ac1].astype(np.float64)

    diff = np.sum((pa - pb) ** 2, axis=2)
    denom = np.sum(wgt)
    if denom < 1e-9:
        return float(np.mean(diff))
    return float(np.sum(diff * wgt) / denom)


def _init_nnf_inpainting(h, w, valid_centers, patch_size, seed=None):
    """Random NNF with centers chosen between the valid ones (known patch)"""
    if seed is not None:
        np.random.seed(seed)

    coords = np.argwhere(valid_centers)
    nnf = np.zeros((h, w, 2), dtype=np.int32)

    if len(coords) == 0:
        p = patch_size // 2
        nnf[:, :, 0] = np.random.randint(p, max(h - p, p + 1), size=(h, w))
        nnf[:, :, 1] = np.random.randint(p, max(w - p, p + 1), size=(h, w))
        return nnf

    idx = np.random.randint(0, len(coords), size=(h, w))
    nnf[:, :, 0] = coords[idx, 0]
    nnf[:, :, 1] = coords[idx, 1]
    return nnf


def _patchmatch_inpainting(img, source, mask, patch_size, pm_iters, conf,
                           alpha=0.5, attempts=2, seed=None):
    """Constrained patchmatch for inpainting: candidate centers must lie in valid_source_centers (known patch)"""
    h, w = img.shape[:2]
    p = patch_size // 2
    valid_centers = _valid_source_centers(mask, patch_size)

    nnf = _init_nnf_inpainting(h, w, valid_centers, patch_size, seed=seed)
    nnd = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            br, bc = nnf[i, j]
            nnd[i, j] = _patch_distance_masked(img, source, i, j, br, bc,
                                               patch_size, conf)

    def ok_center(r, c):
        if r < p or r >= h - p or c < p or c >= w - p:
            return False
        return bool(valid_centers[r, c])

    for it in range(1, pm_iters + 1):
        is_odd = (it % 2 == 1)
        rows = range(h) if is_odd else range(h - 1, -1, -1)
        cols = range(w) if is_odd else range(w - 1, -1, -1)

        for i in rows:
            for j in cols:
                best = nnd[i, j]
                br, bc = nnf[i, j]

                cand_list = []
                if is_odd:
                    if i > 0:
                        c = nnf[i - 1, j].copy()
                        c[0] = min(c[0] + 1, h - p - 1)
                        cand_list.append(c)
                    if j > 0:
                        c = nnf[i, j - 1].copy()
                        c[1] = min(c[1] + 1, w - p - 1)
                        cand_list.append(c)
                else:
                    if i < h - 1:
                        c = nnf[i + 1, j].copy()
                        c[0] = max(c[0] - 1, p)
                        cand_list.append(c)
                    if j < w - 1:
                        c = nnf[i, j + 1].copy()
                        c[1] = max(c[1] - 1, p)
                        cand_list.append(c)

                for c in cand_list:
                    rr = int(np.clip(c[0], p, h - p - 1))
                    cc = int(np.clip(c[1], p, w - p - 1))
                    if not ok_center(rr, cc):
                        continue
                    d = _patch_distance_masked(img, source, i, j, rr, cc,
                                               patch_size, conf)
                    if d < best:
                        best = d
                        br, bc = rr, cc

                radius = max(h, w)
                while radius >= 1:
                    r0 = int(max(br - radius, p))
                    r1 = int(min(br + radius, h - p - 1))
                    c0 = int(max(bc - radius, p))
                    c1 = int(min(bc + radius, w - p - 1))
                    if r0 < r1 and c0 < c1:
                        for _ in range(attempts):
                            rr = int(np.random.randint(r0, r1 + 1))
                            cc = int(np.random.randint(c0, c1 + 1))
                            if not ok_center(rr, cc):
                                continue
                            d = _patch_distance_masked(img, source, i, j, rr, cc,
                                                       patch_size, conf)
                            if d < best:
                                best = d
                                br, bc = rr, cc
                    radius = int(radius * alpha)

                nnf[i, j] = (br, bc)
                nnd[i, j] = best

    return nnf, nnd


def _vote(img, source, nnf, mask, patch_size, conf):
    """Weighted average of overlapping patches, which updates only the pixels inside the hole (mask=True)"""
    h, w = img.shape[:2]
    p = patch_size // 2

    acc = np.zeros((h, w, 3), dtype=np.float64)
    wgt = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            br, bc = nnf[i, j]
            weight_ij = float(conf[i, j])

            for di in range(-p, p + 1):
                for dj in range(-p, p + 1):
                    ai = i + di
                    aj = j + dj
                    bi = br + di
                    bj = bc + dj
                    if 0 <= ai < h and 0 <= aj < w and 0 <= bi < h and 0 <= bj < w:
                        acc[ai, aj] += source[bi, bj].astype(np.float64) * weight_ij
                        wgt[ai, aj] += weight_ij

    out = img.copy().astype(np.float64)
    sel = mask & (wgt > 1e-12)
    out[sel] = acc[sel] / wgt[sel, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def multiscale_inpainting(img, mask, patch_size=5, num_scales=3,
                          em_iters=6, pm_iters=3, verbose=False, seed=None):
    """
    Multiscale search + vote inpainting
    Parameters
        - img: RGB uint8 image
        - mask: boolean (H, W), True = pixels to fill
        - patch_size: patch side length (odd)
        - num_scales: gaussian pyramid levels
        - em_iters: EM (search + vote) iterations per scale
        - pm_iters: patchmatch iterations for each EM step
    """
    imgs, masks = _build_pyramid(img, mask, num_scales)
    result = None

    for s, (im_s, mk_s) in enumerate(zip(imgs, masks)):
        h, w = im_s.shape[:2]
        if verbose:
            print(f"[scale {s}] size={w}x{h}  hole_pixels={int(mk_s.sum())}")

        if result is None:
            cur = _init_fill_nearest(im_s, mk_s)
        else:
            up = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
            cur = im_s.copy()
            cur[mk_s] = up[mk_s]

        source = im_s.copy()
        source[mk_s] = cur[mk_s]

        for em in range(1, em_iters + 1):
            conf = np.ones((h, w), dtype=np.float64)
            conf[mk_s] = 0.1 + 0.9 * (em / em_iters)

            nnf, nnd = _patchmatch_inpainting(
                cur, source, mk_s, patch_size, pm_iters, conf,
                alpha=0.5, attempts=2, seed=seed
            )

            cur = _vote(cur, source, nnf, mk_s, patch_size, conf)
            source[mk_s] = cur[mk_s]

            if verbose:
                hole_mse = float(nnd[mk_s].mean()) if mk_s.any() else 0.0
                print(f"  EM {em}/{em_iters}  "
                      f"mean_MSE(hole)={hole_mse:.4f}  RMS={np.sqrt(hole_mse):.4f}")

        result = cur

    return result
