"""
Run DSEC test-set inference and export submission files.

This script runs a trained model on DSEC test samples and writes predicted
optical flow as 3-channel 16-bit PNGs in the format required by DSEC:

    R = flow_x * 128 + 2^15
    G = flow_y * 128 + 2^15
    B = 1

Output layout:
    submission_dir/
      interlaken_00_b/
        000820.png
        ...
      ...

Example:
    python test_dsec.py \
        --checkpoint checkpoints/11_dsec_5bin_20ms_1/best_model.pth \
        --config snn/configs/event_snn_lite_dsec.yaml \
        --data-root ../dsec/test \
        --submission-dir output/dsec_submission
"""

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from comparisons.dsec.dsec_dataset import DSECOpticalFlowDataset
from utils import get_model, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Run DSEC test inference and export submission files')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to model config YAML. If omitted, tries checkpoint["config"].')
    parser.add_argument('--data-root', type=str, default='../dsec/test',
                        help='Path to DSEC test root')
    parser.add_argument('--submission-dir', type=str, default='./output/dsec_submission',
                        help='Directory where submission folders and PNG files are written')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--strict-load', action='store_true',
                        help='Strict checkpoint loading')
    parser.add_argument('--integer-sim', action='store_true',
                        help='Use integer-only forward pass (requires quantized config/checkpoint)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of test samples for quick debugging')
    parser.add_argument('--zip', action='store_true',
                        help='Create submission zip after export')
    parser.add_argument('--zip-name', type=str, default='dsec_submission.zip',
                        help='Zip filename (created next to submission-dir)')
    parser.add_argument('--run-check', action='store_true',
                        help='Run dsec_check format checker after export')
    parser.add_argument('--checker-script', type=str,
                        default='../dsec_check/scripts/check_optical_flow_submission.py',
                        help='Path to check_optical_flow_submission.py')
    parser.add_argument('--flow-timestamps-dir', type=str, default=None,
                        help='Path to extracted flow timestamp CSV directory used by dsec_check')
    return parser.parse_args()


def _extract_state_dict(checkpoint_obj):
    if 'model_state_dict' in checkpoint_obj:
        return checkpoint_obj['model_state_dict']
    if 'state_dict' in checkpoint_obj:
        return checkpoint_obj['state_dict']
    return checkpoint_obj


def build_inference_model(args, device):
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.config is not None:
        config = load_config(args.config)
    else:
        config = checkpoint.get('config', None)

    if config is None:
        raise ValueError('No config found in checkpoint. Please provide --config.')

    model = get_model(config)
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)

    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    model = model.to(device)
    model.eval()
    model.disable_skip = True

    if args.integer_sim:
        if not config.get('quantized', False):
            raise ValueError('--integer-sim requires quantized checkpoint/config (quantized: true).')
        from snn.models.integer_inference import IntegerInferenceModel
        model = IntegerInferenceModel(model, config,
                                      accum_bit_width=config.get('accum_bit_width', 32))
        model = model.to(device)

    return model, config


def flow_to_dsec_png(pred_flow: np.ndarray) -> np.ndarray:
    """Convert flow [H, W, 2] float32 to DSEC 16-bit RGB PNG tensor [H, W, 3]."""
    if pred_flow.ndim != 3 or pred_flow.shape[2] != 2:
        raise ValueError(f'Expected flow shape [H, W, 2], got {pred_flow.shape}')

    encoded = np.round(pred_flow * 128.0 + 2**15).astype(np.int32)
    encoded = np.clip(encoded, 0, 65535).astype(np.uint16)

    png = np.zeros((pred_flow.shape[0], pred_flow.shape[1], 3), dtype=np.uint16)
    png[..., 0] = encoded[..., 0]  # R: flow_x
    png[..., 1] = encoded[..., 1]  # G: flow_y
    png[..., 2] = 1                # B: channel-order sanity marker
    return png


def write_flow_png(path: Path, png_data: np.ndarray):
    """Write uint16 RGB PNG in a way compatible with DSEC checker expectations."""
    try:
        imageio.imwrite(str(path), png_data, format='PNG-FI')
    except Exception:
        import cv2
        # OpenCV writes BGR; flip channels to preserve intended RGB values.
        cv2.imwrite(str(path), np.flip(png_data, axis=-1))


def maybe_run_checker(args, submission_dir: Path):
    if not args.run_check:
        return

    if args.flow_timestamps_dir is None:
        raise ValueError('--run-check requires --flow-timestamps-dir')

    checker = Path(args.checker_script)
    if not checker.exists():
        raise FileNotFoundError(f'Checker script not found: {checker}')

    cmd = [
        'python', str(checker),
        str(submission_dir),
        str(Path(args.flow_timestamps_dir))
    ]
    print('Running submission checker:')
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def maybe_zip_submission(args, submission_dir: Path):
    if not args.zip:
        return

    zip_path = submission_dir.parent / args.zip_name
    cmd = ['zip', '-r', str(zip_path), submission_dir.name]
    print('Creating zip archive:')
    print('  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=submission_dir.parent)
    print(f'Created zip: {zip_path}')


def main():
    args = parse_args()

    device = args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    print(f'Using device: {device}')

    model, config = build_inference_model(args, device)

    dataset_config = dict(config)
    dataset_config['data_root'] = args.data_root
    dataset_config['max_train_samples'] = args.max_samples

    dataset = DSECOpticalFlowDataset(config=dataset_config)
    if len(dataset) == 0:
        raise RuntimeError(f'No DSEC samples found in {args.data_root}')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda')
    )

    submission_dir = Path(args.submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    seq_counts = defaultdict(int)
    print(f'Exporting predictions for {len(dataset)} samples to {submission_dir}')

    with torch.no_grad():
        for batch in tqdm(loader, desc='DSEC test inference'):
            inputs = batch['input'].to(device, non_blocking=True)
            outputs = model(inputs)

            if isinstance(outputs, dict):
                pred_flow = outputs.get('flow', outputs.get('pred_flow', list(outputs.values())[0]))
            else:
                pred_flow = outputs

            pred_flow = pred_flow.detach().cpu().numpy()  # [B, 2, H, W]
            seq_names = batch['metadata']['sequence']
            file_indices = batch['metadata']['file_index']

            for i in range(pred_flow.shape[0]):
                seq = str(seq_names[i])
                file_index = int(file_indices[i].item())

                flow_hw2 = np.transpose(pred_flow[i], (1, 2, 0))
                png_data = flow_to_dsec_png(flow_hw2)

                seq_dir = submission_dir / seq
                seq_dir.mkdir(parents=True, exist_ok=True)
                out_path = seq_dir / f'{file_index:06d}.png'

                write_flow_png(out_path, png_data)
                seq_counts[seq] += 1

    print('\nExport summary:')
    for seq in sorted(seq_counts.keys()):
        print(f'  {seq}: {seq_counts[seq]} files')

    maybe_run_checker(args, submission_dir)
    maybe_zip_submission(args, submission_dir)

    print('\nDone. Submission directory is ready.')


if __name__ == '__main__':
    main()
