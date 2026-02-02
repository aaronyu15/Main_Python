#!/usr/bin/env python3
"""
Tiny event-slice -> SNN classifier prototype (N=9, 25 classes, no polarity).

This script tests the concept:
1) Build two event-count slices: E_{t-2} and E_{t-1} (counts of abs changes over two windows).
2) Trigger on pixels where E_{t-1}(y,x) >= threshold.
3) Extract 9x9 patches from both slices around (x,y).
4) Feed patches into a tiny spiking network and classify flow (vx,vy) in [-2..2]^2.

Dependencies: numpy, torch
Run: python3 test_event_slice_snn.py
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities: flow class mapping
# -----------------------------
VX_MIN, VX_MAX = -2, 2
VY_MIN, VY_MAX = -2, 2
GRID = 5  # 5x5 = 25 classes


def flow_to_class(vx: int, vy: int) -> int:
    if vx < VX_MIN or vx > VX_MAX or vy < VY_MIN or vy > VY_MAX:
        raise ValueError("vx/vy out of range for 25-class mapping.")
    return (vy - VY_MIN) * GRID + (vx - VX_MIN)


def class_to_flow(c: int) -> tuple[int, int]:
    vy = c // GRID + VY_MIN
    vx = c % GRID + VX_MIN
    return vx, vy


# -----------------------------
# Synthetic event-slice generator
# -----------------------------

import numpy as np
from dataclasses import dataclass

@dataclass
class SceneObj:
    kind: str               # "circle" or "rect"
    tex: np.ndarray         # (S,S) texture intensity
    mask: np.ndarray        # (S,S) alpha mask in [0,1]
    y: float
    x: float
    vy: float
    vx: float
    ang: float = 0.0
    vang: float = 0.0      # radians per microstep
    scale: float = 1.0

def make_base_texture(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    for _ in range(6):
        fy = int(rng.integers(8, 20))
        fx = int(rng.integers(8, 20))
        ky = np.hanning(fy).astype(np.float32)
        kx = np.hanning(fx).astype(np.float32)
        blob = np.outer(ky, kx)
        cy = int(rng.integers(0, h))
        cx = int(rng.integers(0, w))
        y0 = max(0, cy - fy // 2); y1 = min(h, y0 + fy)
        x0 = max(0, cx - fx // 2); x1 = min(w, x0 + fx)
        by0 = 0; bx0 = 0
        by1 = y1 - y0; bx1 = x1 - x0
        base[y0:y1, x0:x1] += float(rng.uniform(-6.0, 6.0)) * blob[by0:by1, bx0:bx1]

    base = 128.0 + 18.0 * base
    base = np.clip(base, 0.0, 255.0).astype(np.float32)
    return base

def make_textured_sprite(rng: np.random.Generator, S: int = 21) -> np.ndarray:
    tex = rng.normal(0.0, 1.0, size=(S, S)).astype(np.float32)
    for _ in range(3):
        tex = 0.25 * (np.roll(tex, 1, 0) + np.roll(tex, -1, 0) + np.roll(tex, 1, 1) + np.roll(tex, -1, 1))
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-6)
    tex = 140.0 + 80.0 * (tex - 0.5)
    return tex.astype(np.float32)

def make_mask(kind: str, S: int) -> np.ndarray:
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cy = (S - 1) / 2.0
    cx = (S - 1) / 2.0
    y = yy - cy
    x = xx - cx
    if kind == "circle":
        r = 0.45 * S
        m = (x*x + y*y) <= (r*r)
        mask = m.astype(np.float32)
    else:
        hw = 0.45 * S
        hh = 0.30 * S
        mask = ((np.abs(x) <= hw) & (np.abs(y) <= hh)).astype(np.float32)

    edge = 2.0
    d = np.minimum.reduce([
        yy, xx, (S - 1) - yy, (S - 1) - xx
    ])
    soft = np.clip(d / edge, 0.0, 1.0).astype(np.float32)
    return mask * soft

def rotate_sample(img: np.ndarray, angle: float) -> np.ndarray:
    S = img.shape[0]
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    cy = (S - 1) / 2.0
    cx = (S - 1) / 2.0
    y = yy - cy
    x = xx - cx
    ca = np.cos(angle).astype(np.float32)
    sa = np.sin(angle).astype(np.float32)
    xr =  ca * x + sa * y + cx
    yr = -sa * x + ca * y + cy
    x0 = np.floor(xr).astype(np.int32)
    y0 = np.floor(yr).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, S-1); x1c = np.clip(x1, 0, S-1)
    y0c = np.clip(y0, 0, S-1); y1c = np.clip(y1, 0, S-1)
    wx = xr - x0
    wy = yr - y0
    Ia = img[y0c, x0c]
    Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]
    Id = img[y1c, x1c]
    out = (Ia * (1-wx) * (1-wy) + Ib * wx * (1-wy) + Ic * (1-wx) * wy + Id * wx * wy).astype(np.float32)
    return out

def render_scene(base: np.ndarray, objs: list[SceneObj], cam_dy: float, cam_dx: float) -> np.ndarray:
    h, w = base.shape
    out = base.copy()

    int_cam_dy = int(round(cam_dy))
    int_cam_dx = int(round(cam_dx))
    if int_cam_dy != 0 or int_cam_dx != 0:
        shifted = np.zeros_like(out)
        ys0 = max(0, -int_cam_dy); ys1 = min(h, h - int_cam_dy)
        xs0 = max(0, -int_cam_dx); xs1 = min(w, w - int_cam_dx)
        yd0 = max(0, int_cam_dy); yd1 = min(h, h + int_cam_dy)
        xd0 = max(0, int_cam_dx); xd1 = min(w, w + int_cam_dx)
        shifted[yd0:yd1, xd0:xd1] = out[ys0:ys1, xs0:xs1]
        out = shifted

    for ob in objs:
        S = ob.tex.shape[0]
        tex = rotate_sample(ob.tex, ob.ang)
        msk = rotate_sample(ob.mask, ob.ang)

        oy = int(round(ob.y))
        ox = int(round(ob.x))
        y0 = oy - S // 2; y1 = y0 + S
        x0 = ox - S // 2; x1 = x0 + S
        if y1 <= 0 or y0 >= h or x1 <= 0 or x0 >= w:
            continue

        sy0 = max(0, -y0); sx0 = max(0, -x0)
        sy1 = min(S, h - y0); sx1 = min(S, w - x0)
        dy0 = max(0, y0); dx0 = max(0, x0)
        dy1 = dy0 + (sy1 - sy0); dx1 = dx0 + (sx1 - sx0)

        patch = tex[sy0:sy1, sx0:sx1]
        alpha = msk[sy0:sy1, sx0:sx1]
        out_region = out[dy0:dy1, dx0:dx1]
        out[dy0:dy1, dx0:dx1] = out_region * (1.0 - alpha) + patch * alpha

    return np.clip(out, 0.0, 255.0).astype(np.float32)

def dvs_event_counts(frames: list[np.ndarray], C: float = 0.15, sigma_read: float = 0.0, rng=None) -> np.ndarray:
    """
    Simplified DVS model using log intensity and contrast threshold.
    For each pixel, generate N events whenever |ΔlogI| accumulates beyond C.
    Polarity is ignored at the end by taking absolute counts.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    h, w = frames[0].shape
    E = np.zeros((h, w), dtype=np.uint16)

    eps = 1e-3
    last = np.log(frames[0] + eps).astype(np.float32)

    for t in range(1, len(frames)):
        cur = frames[t]
        if sigma_read > 0.0:
            cur = cur + rng.normal(0.0, sigma_read, size=cur.shape).astype(np.float32)
            cur = np.clip(cur, 0.0, 255.0)

        logI = np.log(cur + eps).astype(np.float32)
        d = logI - last

        n = np.floor(np.abs(d) / C).astype(np.int32)
        n = np.clip(n, 0, 20)
        E += n.astype(np.uint16)

        last = last + np.sign(d) * (n.astype(np.float32) * C)

    return E

def generate_scene_and_slices(cfg, rng):
    h, w = cfg.h, cfg.w
    base = make_base_texture(h, w, rng)

    vx = int(rng.integers(VX_MIN, VX_MAX + 1))
    vy = int(rng.integers(VY_MIN, VY_MAX + 1))

    S = int(rng.integers(17, 25))
    kind = "circle" if rng.random() < 0.6 else "rect"
    tex = make_textured_sprite(rng, S=S)
    mask = make_mask(kind, S=S)

    y = float(rng.integers(cfg.obj_size, h - cfg.obj_size))
    x = float(rng.integers(cfg.obj_size, w - cfg.obj_size))
    ang = float(rng.uniform(-0.5, 0.5))
    vang = float(rng.uniform(-0.06, 0.06))

    objs0 = [SceneObj(kind=kind, tex=tex, mask=mask, y=y, x=x, vy=float(vy), vx=float(vx), ang=ang, vang=vang)]

    if rng.random() < 0.4:
        kind2 = "rect" if kind == "circle" else "circle"
        S2 = int(rng.integers(15, 23))
        tex2 = make_textured_sprite(rng, S=S2)
        mask2 = make_mask(kind2, S=S2)
        y2 = float(rng.integers(cfg.obj_size, h - cfg.obj_size))
        x2 = float(rng.integers(cfg.obj_size, w - cfg.obj_size))
        vx2 = float(rng.uniform(-1.0, 1.0))
        vy2 = float(rng.uniform(-1.0, 1.0))
        ang2 = float(rng.uniform(-0.5, 0.5))
        vang2 = float(rng.uniform(-0.05, 0.05))
        objs0.append(SceneObj(kind=kind2, tex=tex2, mask=mask2, y=y2, x=x2, vy=vy2, vx=vx2, ang=ang2, vang=vang2))

    cam_vx = float(rng.uniform(-0.4, 0.4))
    cam_vy = float(rng.uniform(-0.4, 0.4))

    def make_window_frames(objs, cam_start_y, cam_start_x):
        frames = []
        cam_y = cam_start_y
        cam_x = cam_start_x
        for s in range(cfg.window_steps + 1):
            frame = render_scene(base, objs, cam_dy=cam_y, cam_dx=cam_x)
            frames.append(frame)
            for ob in objs:
                ob.y += ob.vy / cfg.window_steps
                ob.x += ob.vx / cfg.window_steps
                ob.ang += ob.vang
            cam_y += cam_vy
            cam_x += cam_vx
        return frames, cam_y, cam_x

    objs_a = [SceneObj(**vars(o)) for o in objs0]
    frames_tm2, cam_y_end, cam_x_end = make_window_frames(objs_a, cam_start_y=0.0, cam_start_x=0.0)

    objs_b = [SceneObj(**vars(o)) for o in objs_a]
    frames_tm1, _, _ = make_window_frames(objs_b, cam_start_y=cam_y_end, cam_start_x=cam_x_end)

    E_tm2 = dvs_event_counts(frames_tm2, C=0.15, sigma_read=0.8, rng=rng)
    E_tm1 = dvs_event_counts(frames_tm1, C=0.15, sigma_read=0.8, rng=rng)

    # Add a bit of background noise events to imitate sensor noise/hot pixels.
    noise_rate = float(rng.uniform(0.0, 0.03))
    if noise_rate > 0.0:
        noise1 = (rng.random((h, w)) < noise_rate).astype(np.uint16)
        noise2 = (rng.random((h, w)) < noise_rate).astype(np.uint16)
        E_tm1 = (E_tm1 + noise1).astype(np.uint16)
        E_tm2 = (E_tm2 + noise2).astype(np.uint16)

    return E_tm2.astype(np.float32), E_tm1.astype(np.float32), vx, vy

def generate_one_sample(cfg, rng):
    h, w = cfg.h, cfg.w
    N = cfg.patch_n
    half = N // 2

    for attempt in range(300):
        E_tm2, E_tm1, vx, vy = generate_scene_and_slices(cfg, rng)

        ys, xs = np.where(E_tm1 >= cfg.trigger_thresh)
        if len(ys) == 0:
            continue

        vals = E_tm1[ys, xs]
        K = min(500, len(ys))
        top_idx = np.argpartition(vals, -K)[-K:]
        pick = int(rng.integers(0, len(top_idx)))
        cy = int(ys[top_idx[pick]])
        cx = int(xs[top_idx[pick]])

        if cy - half < 0 or cy + half >= h or cx - half < 0 or cx + half >= w:
            continue

        p1 = E_tm1[cy - half: cy + half + 1, cx - half: cx + half + 1]
        p2 = E_tm2[cy - half: cy + half + 1, cx - half: cx + half + 1]

        p1n = p1.astype(np.float32) / float(cfg.window_steps)
        p2n = p2.astype(np.float32) / float(cfg.window_steps)

        x = np.stack([p1n, p2n], axis=0).astype(np.float32)
        y = flow_to_class(vx=int(vx), vy=int(vy))
        return x, y

    raise RuntimeError("Failed to generate a valid sample after many attempts.")



@dataclass
class SampleConfig:
    h: int = 64
    w: int = 64
    patch_n: int = 9
    obj_size: int = 16
    window_steps: int = 6
    event_thresh: float = 8.0
    trigger_thresh: int = 2
    max_tries: int = 50


class EventSliceDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int, cfg: SampleConfig, seed: int = 0):
        super().__init__()
        self.n = n_samples
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x, y = generate_one_sample(self.cfg, self.rng)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# -----------------------------
# Tiny spiking network (LIF + surrogate)
# -----------------------------
class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        s = torch.sigmoid(alpha * x)
        grad = alpha * s * (1.0 - s)
        return grad_output * grad, None


def spike(x, alpha=10.0):
    return SpikeFn.apply(x, alpha)


class LIFLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, tau: float = 2.0, thr: float = 1.0, alpha: float = 10.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=True)
        self.tau = tau
        self.thr = thr
        self.alpha = alpha

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (T, B, in_dim)
        returns spikes: (T, B, out_dim)
        """
        T, B, _ = x_seq.shape
        mem = torch.zeros((B, self.fc.out_features), device=x_seq.device, dtype=x_seq.dtype)

        decay = math.exp(-1.0 / self.tau)

        spk_out = []
        for t in range(T):
            cur = self.fc(x_seq[t])
            mem = mem * decay + cur
            spk = spike(mem - self.thr, alpha=self.alpha)
            mem = mem - spk * self.thr
            spk_out.append(spk)

        return torch.stack(spk_out, dim=0)


class TinyEventSNN(nn.Module):
    def __init__(self, patch_n: int = 9, hidden: int = 48, T: int = 8):
        super().__init__()
        self.patch_n = patch_n
        self.in_dim = 2 * patch_n * patch_n
        self.hidden = hidden
        self.T = T

        self.lif1 = LIFLayer(self.in_dim, hidden, tau=2.0, thr=1.0, alpha=10.0)
        self.readout = nn.Linear(hidden, 25, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, N, N)
        returns logits: (B, 25)
        """
        B = x.shape[0]
        v = x.view(B, -1)

        v = v / (v.abs().mean(dim=1, keepdim=True) + 1e-6)

        x_seq = v.unsqueeze(0).repeat(self.T, 1, 1)

        spk = self.lif1(x_seq)

        rate = spk.mean(dim=0)
        logits = self.readout(rate)
        return logits


# -----------------------------
# Train / eval
# -----------------------------
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            pred = torch.argmax(logits, dim=1)
            total += int(yb.numel())
            correct += int((pred == yb).sum().item())
            loss_sum += float(loss.item()) * int(yb.numel())
    return loss_sum / max(1, total), correct / max(1, total)
def generate_one_sample_with_meta(cfg: SampleConfig, rng: np.random.Generator):
    """
    Returns a diagnostic sample plus metadata for plotting.

    Expected: you have a function that returns (E_tm2, E_tm1, vx, vy) for a scene,
    where E_tm2 and E_tm1 are float32 (or castable) event-count images.

    The trigger point is chosen from among the top-K event pixels in E_tm1 so that
    the displayed patches and arrows usually come from a strong moving edge.
    """
    h, w = cfg.h, cfg.w
    N = cfg.patch_n
    half = N // 2

    for _ in range(300):
        # This should be your newer, more complex generator.
        # It must return slices and a single (vx,vy) label in [-2..2].
        E_tm2, E_tm1, vx, vy = generate_scene_and_slices(cfg, rng)

        E_tm2 = E_tm2.astype(np.float32, copy=False)
        E_tm1 = E_tm1.astype(np.float32, copy=False)

        ys, xs = np.where(E_tm1 >= cfg.trigger_thresh)
        if len(ys) == 0:
            continue

        vals = E_tm1[ys, xs]
        K = min(800, len(vals))
        top_idx = np.argpartition(vals, -K)[-K:]
        pick = int(rng.integers(0, len(top_idx)))
        cy = int(ys[top_idx[pick]])
        cx = int(xs[top_idx[pick]])

        if cy - half < 0 or cy + half >= h or cx - half < 0 or cx + half >= w:
            continue

        p1_raw = E_tm1[cy - half: cy + half + 1, cx - half: cx + half + 1].copy()
        p2_raw = E_tm2[cy - half: cy + half + 1, cx - half: cx + half + 1].copy()

        # Normalize similarly to training. If you changed training normalization, change this too.
        p1 = p1_raw / float(cfg.window_steps)
        p2 = p2_raw / float(cfg.window_steps)

        x_np = np.stack([p1, p2], axis=0).astype(np.float32)
        y_class = flow_to_class(vx=int(vx), vy=int(vy))

        meta = {
            "E_tm2": E_tm2,
            "E_tm1": E_tm1,
            "cy": cy,
            "cx": cx,
            "vx": int(vx),
            "vy": int(vy),
            "patch_tm1_raw": p1_raw,
            "patch_tm2_raw": p2_raw,
        }
        return x_np, y_class, meta

    raise RuntimeError("Failed to generate a valid diagnostic sample after many attempts.")

def show_diagnostic(
    model: nn.Module,
    device: torch.device,
    cfg: SampleConfig,
    seed: int = 123,
    arrow_scale: float = 8.0,
    show_overlay: bool = True,
):
    """
    Generates a diagnostic sample, runs the model, and plots:
    - E_{t-2} and E_{t-1} (full slices) with the trigger point
    - the 9x9 raw patches from both slices
    - a difference patch (t-1 - t-2)
    - GT and predicted flow arrows at the trigger point
    It also prints top-5 class probabilities and can overlay arrows across the slice.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    x_np, y_class, meta = generate_one_sample_with_meta(cfg, rng)

    xb = torch.from_numpy(x_np).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        pred_class = int(np.argmax(probs))

    gt_vx, gt_vy = class_to_flow(int(y_class))
    pr_vx, pr_vy = class_to_flow(pred_class)

    E2 = meta["E_tm2"]
    E1 = meta["E_tm1"]
    cy, cx = meta["cy"], meta["cx"]
    p1_raw = meta["patch_tm1_raw"]
    p2_raw = meta["patch_tm2_raw"]

    top5 = np.argsort(-probs)[:5]
    top5_lines = []
    for k in top5:
        vxk, vyk = class_to_flow(int(k))
        top5_lines.append(f"({vxk},{vyk}) : {probs[k]*100:.1f}%")

    fig = plt.figure(figsize=(13, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_title("Slice t-2 (event counts)")
    im1 = ax1.imshow(E2, cmap="gray")
    ax1.plot(cx, cy, "r.", markersize=8)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_axis_off()

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_title("Slice t-1 (event counts) with GT and Pred")
    im2 = ax2.imshow(E1, cmap="gray")
    ax2.plot(cx, cy, "r.", markersize=8)

    ax2.arrow(
        cx, cy,
        gt_vx * arrow_scale, gt_vy * arrow_scale,
        head_width=1.6, head_length=2.2, length_includes_head=True,
        color="lime"
    )
    ax2.arrow(
        cx, cy,
        pr_vx * arrow_scale, pr_vy * arrow_scale,
        head_width=1.6, head_length=2.2, length_includes_head=True,
        color="cyan"
    )

    ax2.text(
        0.02, 0.98,
        f"GT (vx,vy)=({gt_vx},{gt_vy})\nPred (vx,vy)=({pr_vx},{pr_vy})",
        transform=ax2.transAxes,
        va="top", ha="left",
        color="yellow",
        bbox=dict(facecolor="black", alpha=0.5, pad=4)
    )
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_axis_off()

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_title("Patch t-2 (raw counts)")
    im3 = ax3.imshow(p2_raw, cmap="gray")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_axis_off()

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("Patch t-1 (raw counts)")
    im4 = ax4.imshow(p1_raw, cmap="gray")
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.set_axis_off()

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("Patch difference (t-1 - t-2)")
    im5 = ax5.imshow(p1_raw.astype(np.float32) - p2_raw.astype(np.float32), cmap="gray")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    ax5.set_axis_off()

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("Top-5 class probabilities")
    ax6.text(0.05, 0.95, "\n".join(top5_lines), va="top", ha="left", fontsize=12)
    ax6.set_axis_off()

    plt.tight_layout()
    plt.show()

    if show_overlay:
        plt.figure(figsize=(8, 6))
        plt.title("Slice t-1 with sparse arrow overlay (GT lime, Pred cyan)")
        plt.imshow(E1, cmap="gray")
        plt.plot(cx, cy, "r.", markersize=8)

        # Draw GT and predicted arrows at the chosen point.
        plt.arrow(
            cx, cy,
            gt_vx * arrow_scale, gt_vy * arrow_scale,
            head_width=1.6, head_length=2.2, length_includes_head=True,
            color="lime"
        )
        plt.arrow(
            cx, cy,
            pr_vx * arrow_scale, pr_vy * arrow_scale,
            head_width=1.6, head_length=2.2, length_includes_head=True,
            color="cyan"
        )

        # Optionally overlay a few additional predicted arrows at other high-event points
        # to build intuition about spatial consistency, without turning it into clutter.
        ys, xs = np.where(E1 >= cfg.trigger_thresh)
        if len(ys) > 0:
            vals = E1[ys, xs]
            K = min(80, len(vals))
            top = np.argpartition(vals, -K)[-K:]
            rng2 = np.random.default_rng(seed + 1)
            picks = rng2.choice(top, size=min(20, len(top)), replace=False)

            N = cfg.patch_n
            half = N // 2
            for idx in picks:
                yy = int(ys[idx]); xx = int(xs[idx])
                if yy - half < 0 or yy + half >= cfg.h or xx - half < 0 or xx + half >= cfg.w:
                    continue
                p1 = E1[yy - half: yy + half + 1, xx - half: xx + half + 1] / float(cfg.window_steps)
                p2 = E2[yy - half: yy + half + 1, xx - half: xx + half + 1] / float(cfg.window_steps)
                x2_np = np.stack([p1, p2], axis=0).astype(np.float32)
                xb2 = torch.from_numpy(x2_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    pr = int(torch.argmax(model(xb2), dim=1).item())
                vx2, vy2 = class_to_flow(pr)
                plt.arrow(
                    xx, yy,
                    vx2 * (arrow_scale * 0.7), vy2 * (arrow_scale * 0.7),
                    head_width=1.1, head_length=1.6, length_includes_head=True,
                    color="cyan", alpha=0.6
                )

        plt.axis("off")
        plt.show()

    print(f"Diagnostic point (x,y)=({cx},{cy})")
    print(f"GT class={y_class} -> (vx,vy)=({gt_vx},{gt_vy})")
    print(f"Pred class={pred_class} -> (vx,vy)=({pr_vx},{pr_vy})")
    print("Top-5 probabilities:")
    for line in top5_lines:
        print("  " + line)

def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = SampleConfig(
        h=64, w=64,
        patch_n=9,
        obj_size=16,
        window_steps=6,
        event_thresh=8.0,
        trigger_thresh=2,
    )

    train_ds = EventSliceDataset(n_samples=1000, cfg=cfg, seed=1)
    test_ds = EventSliceDataset(n_samples=200, cfg=cfg, seed=999)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyEventSNN(patch_n=cfg.patch_n, hidden=48, T=8).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print(f"Device: {device}")
    print("Flow classes are (vx,vy) with vx,vy in [-2..2], mapped to 25 classes.")

    for epoch in range(1, 11):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        test_loss, test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | test loss {test_loss:.4f} | test acc {test_acc*100:.2f}%")

    xb, yb = next(iter(test_loader))
    xb = xb.to(device)
    yb = yb.to(device)
    with torch.no_grad():
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)

    for i in range(8):
        gt_c = int(yb[i].item())
        pr_c = int(pred[i].item())
        gt_vx, gt_vy = class_to_flow(gt_c)
        pr_vx, pr_vy = class_to_flow(pr_c)
        print(f"Example {i}: GT (vx,vy)=({gt_vx},{gt_vy}) | Pred (vx,vy)=({pr_vx},{pr_vy})")

        # Visual diagnostics: show a few random test cases
    for s in [1, 2, 3]:
        show_diagnostic(model, device, cfg, seed=1000 + s, arrow_scale=6.0)

if __name__ == "__main__":
    main()
