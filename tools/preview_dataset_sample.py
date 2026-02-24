"""
Interactive preview for OpticalFlowDataset samples.
Shows flow (Middlebury colors), valid mask, and an event-based active-region mask
with a slider to move through samples.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# Allow running from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snn.dataset import OpticalFlowDataset
from snn.utils.visualization import flow_to_color


def parse_args():
    parser = argparse.ArgumentParser(description="Preview samples from OpticalFlowDataset")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../blink_sim/output/train_set",
        help="Root folder containing sequences (e.g., ../blink_sim/output/train_set)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Dataset index to show initially (0-based)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Optional sequence name to jump to (overrides --sample-idx when paired with --frame)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Optional frame id within the sequence (overrides --sample-idx when provided with --sequence)",
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=1.0,
        help="Minimum summed events per pixel to mark as active in the event mask",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Temporal bins for event voxelization (passed to dataset)",
    )
    parser.add_argument(
        "--bin-interval-us",
        type=int,
        default=1000,
        help="Bin duration in microseconds (passed to dataset)",
    )
    parser.add_argument(
        "--use-polarity",
        action="store_true",
        help="Keep polarity-separated event channels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Image height for dataset voxelization",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Image width for dataset voxelization",
    )
    return parser.parse_args()


def resolve_index(dataset: OpticalFlowDataset, sequence: str, frame: int, fallback_idx: int) -> int:
    if sequence is not None and frame is not None:
        for i, sample in enumerate(dataset.samples):
            if sample.get("sequence") == sequence and sample.get("index") == frame:
                return i
        raise ValueError(f"No sample found for sequence={sequence}, frame={frame}")
    if fallback_idx < 0 or fallback_idx >= len(dataset):
        raise ValueError(f"sample-idx {fallback_idx} is out of range [0, {len(dataset)-1}]")
    return fallback_idx


def extract_masks(sample: dict, event_threshold: float):
    # Tensors come back on CPU from the dataset
    flow = sample["flow"].cpu().numpy()  # [2, H, W]
    valid = sample["valid_mask"].cpu().numpy()[0]  # [H, W]
    events = sample["input"].cpu().numpy()  # [T, C, H, W]

    flow_color = flow_to_color(flow) / 255.0
    event_sum = events.sum(axis=(0, 1))  # [H, W]
    event_mask = (event_sum > event_threshold).astype(np.float32)
    event_mask_valid = event_mask * valid

    meta = sample.get("metadata", {})
    return flow_color, valid, event_mask_valid, meta


def main():
    args = parse_args()

    dataset_config = {
        "data_root": args.data_root,
        "num_bins": args.num_bins,
        "bin_interval_us": args.bin_interval_us,
        "use_polarity": args.use_polarity,
        "data_size": (args.height, args.width),
    }
    dataset = OpticalFlowDataset(config=dataset_config)
    print(f"Loaded dataset with {len(dataset)} samples from {args.data_root}")

    try:
        start_idx = resolve_index(dataset, args.sequence, args.frame, args.sample_idx)
    except ValueError as exc:
        print(exc)
        return

    # Prepare first sample
    flow_img, valid_mask, event_mask, meta = extract_masks(dataset[start_idx], args.event_threshold)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im_flow = axes[0].imshow(flow_img)
    axes[0].set_title("Optical flow (Middlebury)")
    axes[0].axis("off")

    im_valid = axes[1].imshow(valid_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Valid mask")
    axes[1].axis("off")

    im_events = axes[2].imshow(event_mask, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title(f"Event-active mask (>{args.event_threshold} events)")
    axes[2].axis("off")

    fig.suptitle(f"Sample {start_idx} | seq={meta.get('sequence', 'n/a')} frame={meta.get('index', 'n/a')}", fontsize=12)
    plt.subplots_adjust(bottom=0.15)

    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_slider, "Sample idx", 0, len(dataset) - 1, valinit=start_idx, valstep=1)

    def update(idx: int):
        flow_img_u, valid_mask_u, event_mask_u, meta_u = extract_masks(dataset[idx], args.event_threshold)
        im_flow.set_data(flow_img_u)
        im_valid.set_data(valid_mask_u)
        im_events.set_data(event_mask_u)
        fig.suptitle(
            f"Sample {idx} | seq={meta_u.get('sequence', 'n/a')} frame={meta_u.get('index', 'n/a')}",
            fontsize=12,
        )
        fig.canvas.draw_idle()

    def on_change(val):
        update(int(val))

    slider.on_changed(on_change)
    plt.show()


if __name__ == "__main__":
    main()
