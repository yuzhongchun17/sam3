"""
Segment coloured wires from a .pcd file using SAM3.

Steps:
  1. Load PCD, scale mm -> metres
  2. Project to a synthetic colour + depth image (virtual pinhole camera)
  3. Run SAM3 with wire text prompts -> combined binary mask
  4. Back-project masked pixels -> filtered 3-D points
  5. Save wire_segmented.pcd  (metres, for AnyGrasp)
     Save wire_color.png / wire_depth.png  (masked RGBD for reference)

Run with:
  /home/g20/miniconda3/envs/sam3/bin/python segment_wire.py \
      --pcd  /home/g20/Downloads/Test_point_cloud_unorganized.pcd \
      --out  /home/g20/projects/anygrasp/anygrasp_sdk/grasp_detection/wire_output
"""

import argparse
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

import torch
import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--pcd',  default='/home/g20/Downloads/Test_point_cloud_unorganized.pcd')
parser.add_argument('--out',  default='/home/g20/projects/anygrasp/anygrasp_sdk/grasp_detection/wire_output')
parser.add_argument('--scale',       type=float, default=1000.0,
                    help='Divide PCD coordinates by this to get metres (default 1000 for mm input)')
parser.add_argument('--img_w',       type=int,   default=1280)
parser.add_argument('--img_h',       type=int,   default=1024)
parser.add_argument('--confidence',  type=float, default=0.35)
parser.add_argument('--prompts',     nargs='+',
                    default=['white wire', 'blue wire', 'red wire', 'green wire', 'yellow wire'],
                    help='SAM3 text prompts to segment')
parser.add_argument('--no_viz', action='store_true', help='Skip matplotlib visualisation')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ── Torch setup ────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast('cuda', dtype=torch.bfloat16).__enter__()

# ── Step 1: Load PCD ──────────────────────────────────────────────────────────
print(f'\n[1/4] Loading PCD: {args.pcd}')
pcd_raw = o3d.io.read_point_cloud(args.pcd)
pts  = np.asarray(pcd_raw.points, dtype=np.float64) / args.scale   # metres
cols = np.asarray(pcd_raw.colors, dtype=np.float32)                # [0,1]

valid = pts[:, 2] > 0
pts, cols = pts[valid], cols[valid]
print(f'  {len(pts):,} valid points')
print(f'  XYZ min {pts.min(axis=0)}  max {pts.max(axis=0)}')

# ── Step 2: Project to synthetic RGBD ─────────────────────────────────────────
print(f'\n[2/4] Projecting to {args.img_w}x{args.img_h} image ...')
X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]

BORDER = 0.05
u_norm, v_norm = X / Z, Y / Z
fx = args.img_w * (1 - 2 * BORDER) / (u_norm.max() - u_norm.min())
fy = args.img_h * (1 - 2 * BORDER) / (v_norm.max() - v_norm.min())
f  = min(fx, fy)
cx = args.img_w / 2.0 - f * (u_norm.min() + u_norm.max()) / 2.0
cy = args.img_h / 2.0 - f * (v_norm.min() + v_norm.max()) / 2.0
print(f'  Virtual camera: f={f:.1f}  cx={cx:.1f}  cy={cy:.1f}')

u_px = np.round(f * X / Z + cx).astype(np.int32)
v_px = np.round(f * Y / Z + cy).astype(np.int32)

in_bounds = (u_px >= 0) & (u_px < args.img_w) & (v_px >= 0) & (v_px < args.img_h)
u_s = u_px[in_bounds];  v_s = v_px[in_bounds]
Z_s = Z[in_bounds].astype(np.float32)
c_s = cols[in_bounds]

# Z-buffer: write far -> near so nearest point survives
order = np.argsort(-Z_s)
u_s, v_s, Z_s, c_s = u_s[order], v_s[order], Z_s[order], c_s[order]

depth_img = np.zeros((args.img_h, args.img_w), dtype=np.float32)
color_img = np.zeros((args.img_h, args.img_w, 3), dtype=np.float32)
depth_img[v_s, u_s] = Z_s
color_img[v_s, u_s] = c_s

color_uint8 = (color_img * 255).clip(0, 255).astype(np.uint8)

# Save raw RGBD for reference
Image.fromarray(color_uint8).save(os.path.join(args.out, 'color.png'))
depth_mm = (depth_img * 1000).astype(np.uint16)
Image.fromarray(depth_mm).save(os.path.join(args.out, 'depth.png'))
print('  Saved color.png, depth.png')

# ── Step 3: SAM3 segmentation ─────────────────────────────────────────────────
print(f'\n[3/4] Running SAM3 (confidence={args.confidence}) ...')
sam3_root = os.path.join(os.path.dirname(sam3.__file__), '..')
bpe_path  = os.path.join(sam3_root, 'assets', 'bpe_simple_vocab_16e6.txt.gz')
model     = build_sam3_image_model(bpe_path=bpe_path)
processor = Sam3Processor(model, confidence_threshold=args.confidence)

pil_image       = Image.fromarray(color_uint8)
inference_state = processor.set_image(pil_image)

H, W = args.img_h, args.img_w
combined_binary = np.zeros((H, W), dtype=np.float32)

for prompt in args.prompts:
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
    raw = inference_state.get('masks')
    if raw is None or len(raw) == 0:
        print(f'  ✗  {prompt}')
        continue
    m = raw.detach().cpu().numpy()
    if m.ndim == 4:
        m = m.squeeze(1)
    mask_2d = np.max(m, axis=0)
    combined_binary = np.maximum(combined_binary, mask_2d)
    print(f'  ✓  {prompt}  ({int(mask_2d.sum()):,} px)')

total_px = int(combined_binary.sum())
print(f'  Total wire pixels: {total_px:,} / {H*W:,}  ({total_px/(H*W)*100:.1f}%)')

if total_px == 0:
    raise RuntimeError('No wire pixels detected. Try adjusting --prompts or --confidence.')

# ── Step 4: Apply mask + save outputs ─────────────────────────────────────────
print(f'\n[4/4] Applying mask and saving outputs ...')
mask_bool = combined_binary > 0

# Masked RGBD
masked_color = color_uint8.copy();  masked_color[~mask_bool] = 0
masked_depth = depth_img.copy();    masked_depth[~mask_bool] = 0.0
Image.fromarray(masked_color).save(os.path.join(args.out, 'wire_color.png'))
Image.fromarray((masked_depth * 1000).astype(np.uint16)).save(os.path.join(args.out, 'wire_depth.png'))

# Back-project masked pixels -> 3-D points
u_all = np.round(f * X / Z + cx).astype(np.int32)
v_all = np.round(f * Y / Z + cy).astype(np.int32)
in_b  = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
in_m  = np.zeros(len(pts), dtype=bool)
in_m[in_b] = mask_bool[v_all[in_b], u_all[in_b]]

wire_pts  = pts[in_m].astype(np.float64)
wire_cols = cols[in_m].astype(np.float64)
print(f'  Wire points: {len(wire_pts):,}')

wire_pcd = o3d.geometry.PointCloud()
wire_pcd.points = o3d.utility.Vector3dVector(wire_pts)
wire_pcd.colors = o3d.utility.Vector3dVector(wire_cols)
out_pcd = os.path.join(args.out, 'wire_segmented.pcd')
o3d.io.write_point_cloud(out_pcd, wire_pcd)
print(f'  Saved wire_segmented.pcd -> {out_pcd}')
print(f'  Saved wire_color.png, wire_depth.png')

# ── Optional visualisation ─────────────────────────────────────────────────────
if not args.no_viz:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].imshow(color_uint8);           axes[0].set_title('Projected colour'); axes[0].axis('off')
    axes[1].imshow(combined_binary, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Wire mask');        axes[1].axis('off')
    axes[2].imshow(masked_color);          axes[2].set_title('Masked colour'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'segmentation_result.png'), dpi=100)
    print(f'  Saved segmentation_result.png')
    plt.show()

print('\nDone. Now run AnyGrasp with:')
print(f'  /home/g20/miniconda3/envs/anygrasp/bin/python '
      f'/home/g20/projects/anygrasp/anygrasp_sdk/grasp_detection/demo_pcd.py '
      f'--checkpoint_path /home/g20/projects/anygrasp/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar '
      f'--pcd_path {out_pcd} --scale 1.0 --debug')
