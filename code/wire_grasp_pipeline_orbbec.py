#!/usr/bin/env python
"""
wire_grasp_pipeline_orbbec.py
-------------------------------
Live, in-process merge of capture_orbbec_images.py + segment_orbbec_wires.py +
wire_grasp_pose_orbbec.py: on each trigger, capture a frame from the Orbbec
"eye" camera, segment it with SAM3, compute a top-down grasp pose for the
best wire, and publish it -- no round-trip through PNG/PCD files on disk.

The three source scripts stay useful on their own for offline
capture/tuning/debugging; this script is the "just run the whole thing"
path once the workspace bbox and wire colours are already dialed in.

Pipeline per trigger:
  1. capture_orbbec_images.OrbbecCapture.grab_rgb() -- latest colour+depth,
     cropped to the workspace bbox (same auto/manual/saved bbox workflow).
  2. SAM3 text-prompt segmentation per --colors (default: the whole bundle --
     tries "twisted wire" plus a few synonym phrasings in turn until one
     detects something, since a real twist doesn't always match the first
     wording; pass --colors to target individual colours instead),
     masked by valid depth, back-projected to camera-frame XYZ (mm) --
     segment_orbbec_wires.py's segment_one(), kept in memory instead of
     writing _mask_*.png/_pcd_*.pcd.
  3. Whichever segmented wire has the most 3-D points -- or --wire, forced --
     goes through wire_grasp_pose_orbbec.py's find_grasp_pose() (local PCA
     direction at the wire's centroid + top_down_frame()).
  4. Publish the grasp already in ROBOT BASE frame on --zmq_pub_addr (same
     port/schema grasp_executor.py listens on, "frame": "base"), plus an
     optional persistent ROS2 static TF for RViz sanity-checking.

cv2.imshow can't run in this process once SAM3 is loaded (confirmed segfault,
see stream_segment.py's module docstring), so interactive bbox picking ('s')
runs in a separate spawned process exactly like stream_segment.py's
_viewer_process -- SAM3 is loaded once at startup, after which this process
never touches cv2's GUI backend itself.

Prerequisite: pub_orbbec must be running on the Jetson for --camera_name's
camera (richtech-dex-open-cli run-plugin pub_orbbec -i <index>).

Stdin commands (all + Enter):
  <blank>   Capture -> segment -> grasp -> publish, once.
  s         Hand-pick the workspace crop box (drag on a live frame).
  a         Auto-detect the crop box (table plane + hue-variance twist).
  m         Reload the crop box saved at --bbox_file.
  q         Quit.

Run:
    conda activate sam3
    python wire_grasp_pipeline_orbbec.py
    python wire_grasp_pipeline_orbbec.py --auto_bbox --wire yellow_wire
    python wire_grasp_pipeline_orbbec.py --colors "yellow wire,red wire" --no_publish_tf
"""

import os

# Must be set before torch/sam3 (loaded lazily in load_sam3()) or cv2 (imported next) touch
# OpenMP: SAM3/torch pull in one OpenMP runtime and cv2's spawned viewer process pulls in
# another, and without this the duplicate-runtime abort doesn't always surface where you'd
# expect -- it reliably killed this script (SIGABRT, "terminate called without an active
# exception") during interpreter shutdown, only when SAM3 was loaded AND the OrbbecCapture
# zmq threads were still alive AND the spawned viewer process existed, all three required to
# reproduce it. See stream_segment.py's module docstring for the same underlying conflict.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import math
import multiprocessing as mp
import subprocess
import sys
import threading
import time

import cv2
import msgpack
import numpy as np
import zmq

import bbox_utils

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Known individual wire colours -- only used to look up a viz colour when
# --colors names one of these explicitly (see parse_colors_arg()). The default
# capture-time target is the whole twisted bundle, not any single colour.
WIRE_COLOR_PALETTE = [
    ('yellow_wire', 'yellow wire', [255, 215, 0]),
    ('red_wire',    'red wire',    [220, 50, 50]),
    ('blue_wire',   'blue wire',   [30, 144, 255]),
    ('white_wire',  'white wire',  [230, 230, 230]),
]

WIRES_DEFAULT = [
    ('twisted_wire', [
        'twisted wire',
        'twisted wires',
        'bundle of wires',
        'wire bundle',
        'coiled wire',
        'cable bundle',
    ], [200, 200, 200]),
]

DEFAULT_INTRINSICS_JSON = os.path.expanduser(
    "~/jetson_code/richtech-dex-open/code/cam_intrinsics.json")
DEFAULT_CHECKPOINT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--facebook--sam3/"
    "snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")
DEFAULT_CAMERA_EXTRINSIC_JSON = os.path.expanduser(
    "~/jetson_code/richtech-dex-open/code/camera_extrinsics/eye_camera.json")

DEFAULT_CAMERA_FRAME = 'eye_camera_optical_frame'
DEFAULT_GRASP_FRAME = 'wire_grasp'
ROS_SETUP_CMD = 'source /opt/ros/jazzy/setup.bash'

# Same port grasp_executor.py listens on by default -- this is a drop-in
# alternative source to anygrasp_sam3_stream.py, not something that runs
# alongside it on the same port.
DEFAULT_ZMQ_PUB_ADDR = 'tcp://*:5561'
ZMQ_SLOW_JOINER_DELAY_S = 0.5  # PUB/SUB: give a subscriber time to connect before the first send()

# grasp_executor.py's --ready_signal_addr default is 'tcp://*:5564' (it binds); this is the
# Jetson's IP, the host that script runs on (same machine as --jetson_ip's camera stream).
DEFAULT_READY_SIGNAL_ADDR = 'tcp://192.168.11.11:5564'

LOCAL_PCA_RADIUS_MM = 10.0
WIRE_WIDTH_MM = 3.0
SIDE_MARGIN_MM = 4.0

MASK_WINDOW_NAME = "wire_grasp_pipeline_orbbec: mask overlay"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--jetson_ip", default="192.168.11.11", help="IP of the Jetson (default: %(default)s)")
    p.add_argument("--color_port", default="10011", help="ZMQ colour port from pub_orbbec (default: %(default)s = eye)")
    p.add_argument("--depth_port", default="10013", help="ZMQ depth port from pub_orbbec (default: %(default)s = eye)")
    p.add_argument("--wait_timeout", type=float, default=10.0,
                   help="Seconds to wait for a colour/depth frame before giving up (default: %(default)s)")
    p.add_argument("--detect_retries", type=int, default=2,
                   help="Extra attempts (fresh capture + re-segment) if a cycle finds no wire/grasp "
                        "pose, before giving up on it -- SAM3 detection is flaky frame-to-frame "
                        "(default: %(default)s, i.e. 3 attempts total)")

    # Workspace bbox -- same workflow as capture_orbbec_images.py / stream_segment.py.
    p.add_argument("--bbox", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                   help="Crop box applied to every captured frame")
    p.add_argument("--bbox_file", default=None,
                   help="Where the crop box is stored (default: bbox.json next to this script)")
    p.add_argument("--select_bbox", action="store_true", help="Hand-pick the crop box before the main loop starts")
    p.add_argument("--auto_bbox", action="store_true", help="Auto-detect the crop box before the main loop starts")
    p.add_argument("--full", action="store_true", help="Ignore any saved crop box for this run")
    p.add_argument("--display_max", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Cap the bbox-picker window to this size (default: auto-detect the screen)")
    p.add_argument("--table_band", type=float, default=0.015,
                   help="Auto bbox: metres either side of the table plane still counted as table (default: %(default)s)")
    p.add_argument("--bbox_pad", type=float, default=0.6,
                   help="Auto bbox: padding around the detected twist, fraction of its longer side (default: %(default)s)")
    p.add_argument("--var_thresh", type=float, default=0.25,
                   help="Auto bbox: hue-mix threshold for twisted wire, 0-1 (default: %(default)s)")
    p.add_argument("--lift_mm", type=float, default=20.0,
                   help="Auto bbox: mm a candidate must stand above the table (default: %(default)s)")

    # Segmentation.
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to local SAM3 checkpoint .pt file")
    p.add_argument("--confidence", type=float, default=0.35)
    p.add_argument("--min_confidence", type=float, default=0.15,
                   help="Fallback confidence retried once per prompt (cheaply -- reuses the "
                        "cached image encoding) when nothing clears --confidence, so a wire "
                        "just under the normal threshold (lighting/angle) still gets found "
                        "(default: %(default)s)")
    p.add_argument("--colors", default=None,
                   help='Comma-separated SAM3 prompts to segment each capture with, e.g. '
                        '"yellow wire,red wire" (default: "twisted wire", the whole bundle '
                        'rather than any single colour)')
    p.add_argument("--wire", default=None,
                   help="Force the grasp target to this wire name (matches --colors, spaces -> underscores), "
                        "skipping the most-points comparison across colours")
    p.add_argument("--no_viz", action="store_true", help="Skip the mask-overlay window shown after each segmentation")

    # Camera geometry.
    p.add_argument("--intrinsics_json", default=DEFAULT_INTRINSICS_JSON,
                   help="cam_intrinsics.json to read fx/fy/cx/cy from (default: %(default)s)")
    p.add_argument("--camera_name", default="eye", help="camera_name in --intrinsics_json (default: %(default)s)")
    p.add_argument("--camera_extrinsic_json", default=DEFAULT_CAMERA_EXTRINSIC_JSON,
                   help="Fixed mounting extrinsic (xyz_mm/rpy_deg) for the camera (default: %(default)s)")

    # Grasp publishing.
    p.add_argument("--zmq_pub_addr", default=DEFAULT_ZMQ_PUB_ADDR,
                   help="ZMQ address to publish the grasp pose on, base frame (default: %(default)s)")
    p.add_argument("--no_publish", action="store_true", help="Skip the ZMQ grasp publish")
    p.add_argument("--camera_frame", default=DEFAULT_CAMERA_FRAME, help="ROS TF parent frame (default: %(default)s)")
    p.add_argument("--grasp_frame", default=DEFAULT_GRASP_FRAME, help="ROS TF child frame (default: %(default)s)")
    p.add_argument("--no_publish_tf", action="store_true", help="Skip publishing the ROS2 static TF")

    # Auto-trigger: replaces the blank-Enter keypress with a ZMQ signal from
    # grasp_executor.py (published once both its arms finish homing).
    p.add_argument("--ready_signal_addr", default=DEFAULT_READY_SIGNAL_ADDR,
                   help="ZMQ address to connect to for the ready-signal from grasp_executor.py "
                        "-- must point at the host running that script, not this one (default: %(default)s)")
    p.add_argument("--interactive", action="store_true",
                   help="Fall back to the manual blank-Enter/s/a/m/q stdin loop instead of waiting "
                        "for --ready_signal_addr")

    return p.parse_args()


# ── Capture: ZMQ colour+depth subscriber, ported from capture_orbbec_images.py ──

class OrbbecCapture:
    """Background colour+depth ZMQ subscriber -- same pattern as
    stream_segment.py's DualStreamViewer, decoupled from SAM3/the viewer."""

    def __init__(self, color_ip_port: str, depth_ip_port: str):
        self.context = zmq.Context()
        self.rgb_frame = None
        self.depth_frame = None
        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self._color_ip_port = color_ip_port
        self._depth_ip_port = depth_ip_port
        threading.Thread(target=self._color_thread, daemon=True).start()
        threading.Thread(target=self._depth_thread, daemon=True).start()

    @staticmethod
    def _decode(packed_message, is_depth):
        try:
            message = msgpack.unpackb(packed_message)
            if is_depth and 'depth_png' in message:
                arr = np.frombuffer(message['depth_png'], dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            for key in ('color_img', 'color_png', 'color_jpg'):
                if key in message:
                    arr = np.frombuffer(message[key], dtype=np.uint8)
                    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            arr = np.frombuffer(packed_message, dtype=np.uint8)
            flags = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
            return cv2.imdecode(arr, flags)
        return None

    def _color_thread(self):
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(f"tcp://{self._color_ip_port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        while True:
            try:
                frame = self._decode(sub.recv(zmq.NOBLOCK), is_depth=False)
                if frame is not None:
                    with self.rgb_lock:
                        self.rgb_frame = frame
            except zmq.error.Again:
                time.sleep(0.005)

    def _depth_thread(self):
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(f"tcp://{self._depth_ip_port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        while True:
            try:
                frame = self._decode(sub.recv(zmq.NOBLOCK), is_depth=True)
                if frame is not None:
                    with self.depth_lock:
                        self.depth_frame = frame
            except zmq.error.Again:
                time.sleep(0.005)

    def snapshot(self):
        with self.rgb_lock:
            color_bgr = self.rgb_frame.copy() if self.rgb_frame is not None else None
        with self.depth_lock:
            depth_mm = self.depth_frame.copy() if self.depth_frame is not None else None
        return color_bgr, depth_mm

    def grab_rgb(self, timeout=10.0):
        """One colour+depth grab -> (HxWx3 uint8 RGB, HxW uint16 depth_mm).
        Waits for BOTH streams -- see capture_orbbec_images.py's grab_rgb for why."""
        start = time.time()
        color_bgr = depth_mm = None
        while time.time() - start < timeout:
            color_bgr, depth_mm = self.snapshot()
            if color_bgr is not None and depth_mm is not None:
                return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB), depth_mm
            time.sleep(0.05)
        if color_bgr is not None:
            logger.warning(f"Got colour but no depth after {timeout:.0f}s -- proceeding without depth.")
            return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB), None
        raise TimeoutError(f"No colour frame received after {timeout:.0f}s -- "
                           f"is pub_orbbec running on the Jetson for this camera?")


# ── Bbox picking: interactive selection needs its own process, see module docstring ──

def _viewer_process(queue: mp.Queue, result_queue: mp.Queue, window_name: str):
    """Runs in its own spawned process (never imports torch/sam3) -- copy of
    stream_segment.py's _viewer_process. Handles two tagged messages:
      ('overlay', bgr_frame)              -- just display it
      ('pick_bbox', rgb_frame, disp_max)  -- interactive rubber-band select,
                                              result (x,y,w,h) or None on
                                              `result_queue`
    Polls with a short timeout and calls waitKey() every iteration regardless
    of whether a new frame arrived, so the window keeps pumping its event
    loop between frames -- see stream_segment.py's docstring for why."""
    import queue as _queue
    import cv2 as _cv2
    import bbox_utils as _bbox_utils
    _cv2.namedWindow(window_name, _cv2.WINDOW_NORMAL)
    while True:
        try:
            msg = queue.get(timeout=0.05)
            if msg is None:
                break
            kind = msg[0]
            if kind == 'overlay':
                _cv2.imshow(window_name, msg[1])
            elif kind == 'pick_bbox':
                _, rgb_frame, display_max = msg
                bbox = _bbox_utils.select_bbox_interactive(rgb_frame, display_max=display_max)
                result_queue.put(bbox)
        except _queue.Empty:
            pass
        _cv2.waitKey(1)
    _cv2.destroyAllWindows()


# ── Segmentation: SAM3 load + per-frame masking, ported from segment_orbbec_wires.py ──

def load_sam3(checkpoint, confidence):
    import torch
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast('cuda', dtype=torch.bfloat16).__enter__()

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), '..')
    bpe_path = os.path.join(sam3_root, 'assets', 'bpe_simple_vocab_16e6.txt.gz')
    if checkpoint and os.path.isfile(checkpoint):
        logger.info(f'Loading SAM3 from local checkpoint: {checkpoint}')
        model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint, load_from_HF=False)
    else:
        logger.info('Loading SAM3 from HuggingFace cache (facebook/sam3)...')
        model = build_sam3_image_model(bpe_path=bpe_path, load_from_HF=True)
    logger.info("SAM3 model ready.")
    return Sam3Processor(model, confidence_threshold=confidence)


def parse_colors_arg(colors_arg):
    if not colors_arg:
        return WIRES_DEFAULT
    palette = {name: color for name, _, color in WIRE_COLOR_PALETTE}
    wires = []
    for prompt in (c.strip() for c in colors_arg.split(',') if c.strip()):
        name = prompt.replace(' ', '_')
        wires.append((name, [prompt], palette.get(name, [200, 200, 200])))
    return wires


def segment_frame(processor, rgb, xyz, valid, wires, min_confidence=None):
    """Runs SAM3 on `rgb`, in memory -- no PNG/PCD written to disk (that's
    what segment_orbbec_wires.py is for, offline).

    Each wire in `wires` is (name, prompts, color) where `prompts` is a list
    of text-prompt phrasings tried in order, stopping at the first one SAM3
    detects anything for -- a twisted bundle doesn't always match "twisted
    wire" (lighting/angle can hide the twist), so WIRES_DEFAULT carries a few
    synonym phrasings as free retries against the same cached image encoding.

    If `min_confidence` is set (and lower than the processor's own threshold),
    each prompt that comes up empty at the normal confidence is retried once
    more at `min_confidence` before moving to the next phrasing -- same cheap
    fallback (reuses the cached image encoding) as
    wire_grasp_pipeline_orbbec_untwist.py's segment_frame()/stream_segment.py.

    Returns {wire_name: (mask_bool HxW, points_mm Nx3, colors_0to1 Nx3, prompt_used)}."""
    from PIL import Image
    state = processor.set_image(Image.fromarray(rgb))
    base_confidence = processor.confidence_threshold

    results = {}
    for name, prompts, _color in wires:
        raw_masks = matched_prompt = None
        for prompt in prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            candidate = state.get('masks')
            if (candidate is None or len(candidate) == 0) and min_confidence is not None \
                    and min_confidence < base_confidence:
                logger.info(f"  {name:15s} not detected at confidence={base_confidence:.2f} for "
                            f"'{prompt}', retrying at {min_confidence:.2f}")
                state = processor.set_confidence_threshold(min_confidence, state)
                candidate = state.get('masks')
                processor.set_confidence_threshold(base_confidence)  # restore for the next prompt
            if candidate is not None and len(candidate) > 0:
                raw_masks, matched_prompt = candidate, prompt
                break
        if raw_masks is None:
            tried = ', '.join(f"'{p}'" for p in prompts)
            logger.info(f'  {name:15s} not detected (tried: {tried})')
            continue
        masks_np = raw_masks.detach().cpu().numpy()
        if masks_np.ndim == 4:
            masks_np = masks_np.squeeze(1)
        mask = np.max(masks_np, axis=0) > 0

        sel = mask & valid
        n_pts = int(sel.sum())
        if n_pts == 0:
            logger.info(f'  {name:15s} detected but 0 valid 3-D pts (depth dropout?)')
            continue

        points_mm = xyz[sel] * 1000.0  # metres -> mm, matching wire_grasp_pose_orbbec.py's convention
        colors = rgb[sel].astype(np.float64) / 255.0
        suffix = f" (prompt: '{matched_prompt}')" if matched_prompt != prompts[0] else ""
        logger.info(f'  {name:15s} {n_pts:7,} pts{suffix}')
        results[name] = (mask, points_mm, colors, matched_prompt)

    return results


def show_mask_overlay(viz_queue, color_bgr, mask_bool, prompt, n_pixels,
                       mask_color=(0, 255, 0), alpha=0.5):
    colored_mask = np.zeros_like(color_bgr)
    colored_mask[mask_bool] = mask_color
    overlay = cv2.addWeighted(color_bgr, 1.0, colored_mask, alpha, 0)
    contours, _ = cv2.findContours((mask_bool.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, mask_color, 2)
    cv2.putText(overlay, f"'{prompt}'  ({n_pixels} px)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    viz_queue.put(('overlay', overlay))


# ── Grasp geometry, ported unchanged (frame-agnostic pure functions) from
# wire_grasp_pose_orbbec.py -- see that file's docstrings for full derivations. ──

def get_local_pca_direction(centre_point, points_mm, radius_mm=LOCAL_PCA_RADIUS_MM):
    distances = np.linalg.norm(points_mm - centre_point, axis=1)
    local_points = points_mm[distances <= radius_mm]
    if len(local_points) < 3:
        logger.warning(f"Only {len(local_points)} points within {radius_mm}mm of the grasp "
                        f"centre, cannot fit a local direction. Skipping this wire.")
        return None, None
    centroid = local_points.mean(axis=0)
    _, _, vt = np.linalg.svd(local_points - centroid)
    return local_points, vt[0]


def top_down_frame(centre, wire_dir, down):
    down = down / np.linalg.norm(down)
    z_axis = down
    x_axis = wire_dir - np.dot(wire_dir, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross([1.0, 0.0, 0.0], z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    T = np.eye(4)
    T[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    T[:3, 3] = centre
    return T


def find_grasp_pose(points_mm, down_cam, radius_mm=LOCAL_PCA_RADIUS_MM):
    if len(points_mm) == 0:
        return None
    centroid = points_mm.mean(axis=0)
    seed_pt = points_mm[np.argmin(np.linalg.norm(points_mm - centroid, axis=1))]
    local_pts, direction = get_local_pca_direction(seed_pt, points_mm, radius_mm)
    if local_pts is None:
        return None
    centre = local_pts.mean(axis=0)
    T = top_down_frame(centre, direction, down_cam)
    return T, local_pts


def _rpy_deg_to_matrix(roll_deg, pitch_deg, yaw_deg):
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def camera_down_vector(camera_extrinsic_json):
    with open(camera_extrinsic_json) as f:
        extr = json.load(f)
    R_base_to_optical = _rpy_deg_to_matrix(*extr['rpy_deg'])
    down_cam = R_base_to_optical.T @ np.array([0.0, 0.0, -1.0])
    return down_cam / np.linalg.norm(down_cam)


def camera_to_base_transform(camera_extrinsic_json):
    with open(camera_extrinsic_json) as f:
        extr = json.load(f)
    R_base_to_optical = _rpy_deg_to_matrix(*extr['rpy_deg'])
    xyz_m = np.array(extr['xyz_mm'], dtype=np.float64) / 1000.0
    T = np.eye(4)
    T[:3, :3] = R_base_to_optical
    T[:3, 3] = xyz_m
    return T


def rotation_matrix_to_quaternion_xyzw(R):
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return qx, qy, qz, qw


def publish_grasp_tf(T_best, camera_frame, grasp_frame, prev_proc=None):
    """Publish the grasp pose as a persistent ROS2 static TF. Kills any TF
    publisher this script started previously for the same grasp_frame first
    -- unlike wire_grasp_pose_orbbec.py's single-shot version, this runs in a
    loop, and leaving old publishers running would leave stale/conflicting
    transforms for the same child frame competing on /tf_static.

    Note prev_proc.terminate() alone is NOT enough: `ros2 run` execs the real
    static_transform_publisher binary as a *child*, not in its own place, so
    prev_proc (the `ros2 run` wrapper) dying leaves that child alive,
    reparented to init, still latching /tf_static forever -- pkill-by-frame-
    name reaches both regardless of that parent/child split."""
    if prev_proc is not None and prev_proc.poll() is None:
        prev_proc.terminate()
    subprocess.run(['pkill', '-f', f'static_transform_publisher.*--child-frame-id {grasp_frame}$'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    xyz = (T_best[:3, 3] / 1000.0).tolist()  # mm -> m
    qx, qy, qz, qw = rotation_matrix_to_quaternion_xyzw(T_best[:3, :3])
    cmd = (
        f"{ROS_SETUP_CMD} && ros2 run tf2_ros static_transform_publisher "
        f"--x {xyz[0]} --y {xyz[1]} --z {xyz[2]} "
        f"--qx {qx} --qy {qy} --qz {qz} --qw {qw} "
        f"--frame-id {camera_frame} --child-frame-id {grasp_frame}"
    )
    proc = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info(f"[ros tf] Publishing {camera_frame} -> {grasp_frame} (pid={proc.pid})")
    return proc


def build_grasp_payload(T_grasp_base, wire_name, frame_idx):
    """Same per-grasp schema as wire_grasp_pose_orbbec.py's publish_grasp() --
    "frame": "base" tells grasp_executor.py to skip its own camera-frame
    hand-eye transform (this pose is already in base frame, via
    camera_to_base_transform()). Built but not sent -- see publish_grasps()."""
    jaw_clearance_m = (WIRE_WIDTH_MM + 2 * SIDE_MARGIN_MM) / 1000.0
    return {
        "translation": T_grasp_base[:3, 3].tolist(),
        "rotation":    T_grasp_base[:3, :3].tolist(),
        "score":       1.0,
        "width":       jaw_clearance_m,
        "frame_idx":   frame_idx,
        "prompt":      wire_name,
        "frame":       "base",
        "stem":        f"live_{frame_idx:03d}",
    }


def publish_grasps(pub_socket, payloads, frame_idx):
    """Send every wire's grasp from this capture as ONE message, {"frame_idx":
    ..., "grasps": [payload, ...]} -- not one publish_grasp() call per wire.
    The SUB side (grasp_executor.py) runs with CONFLATE=1, which keeps only
    the latest message and silently drops the rest; back-to-back sends for
    the same capture would race against however fast the executor drains the
    socket, and a colour could be dropped before it's ever read. Bundling
    means one CONFLATE-kept message always carries every colour for that
    capture intact."""
    if not payloads:
        return
    pub_socket.send(msgpack.packb({"frame_idx": frame_idx, "grasps": payloads}))
    names = ", ".join(p["prompt"] for p in payloads)
    logger.info(f"[zmq pub] Published {len(payloads)} grasp(s) (base frame): {names} frame_idx={frame_idx}")


# ── Main loop ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    wires = parse_colors_arg(args.colors)
    wire_filter = args.wire

    bbox_file = args.bbox_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbox.json")
    display_max = tuple(args.display_max) if args.display_max else None

    logger.info(f"Connecting to Orbbec streams at {args.jetson_ip}:{args.color_port} (colour) / "
                f":{args.depth_port} (depth) ...")
    cap = OrbbecCapture(f"{args.jetson_ip}:{args.color_port}", f"{args.jetson_ip}:{args.depth_port}")
    img_rgb, _ = cap.grab_rgb(timeout=args.wait_timeout)
    logger.info(f"Connected: first frame {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    bbox = tuple(args.bbox) if args.bbox else (None if args.full else bbox_utils.load_bbox(str(bbox_file)))
    if bbox and not args.full:
        logger.info(f"Crop box: x={bbox[0]} y={bbox[1]} w={bbox[2]} h={bbox[3]}"
                    f"{'' if args.bbox else f' (from {bbox_file})'}")

    intrinsics = {}

    def get_intrinsics():
        if not intrinsics:
            fx, fy, cx, cy = bbox_utils.load_intrinsics(args.intrinsics_json, camera_name=args.camera_name)
            intrinsics.update(fx=fx, fy=fy, cx=cx, cy=cy)
        return intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

    down_cam = camera_down_vector(args.camera_extrinsic_json)
    logger.info(f"Gravity 'down' in camera-optical frame: [{down_cam[0]:+.3f}, {down_cam[1]:+.3f}, {down_cam[2]:+.3f}]")
    T_optical_to_base = camera_to_base_transform(args.camera_extrinsic_json)

    zmq_ctx = zmq.Context()

    grasp_pub = None
    if not args.no_publish:
        grasp_pub = zmq_ctx.socket(zmq.PUB)
        grasp_pub.bind(args.zmq_pub_addr)
        logger.info(f"Grasp publisher bound to {args.zmq_pub_addr}; waiting {ZMQ_SLOW_JOINER_DELAY_S}s "
                    f"for subscribers to connect...")
        time.sleep(ZMQ_SLOW_JOINER_DELAY_S)

    sig_sub = None
    if not args.interactive:
        sig_sub = zmq_ctx.socket(zmq.SUB)
        sig_sub.setsockopt(zmq.CONFLATE, 1)
        sig_sub.connect(args.ready_signal_addr)
        sig_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        logger.info(f"Waiting for ready-signal on {args.ready_signal_addr} (auto mode -- pass "
                    f"--interactive for the manual blank-Enter/s/a/m/q loop instead)")

    # Interactive bbox picking needs its own process -- see module docstring.
    mp_ctx = mp.get_context('spawn')
    viz_queue, viz_result_queue = mp_ctx.Queue(), mp_ctx.Queue()
    viz_proc = mp_ctx.Process(target=_viewer_process, args=(viz_queue, viz_result_queue, MASK_WINDOW_NAME), daemon=True)
    viz_proc.start()

    def pick_bbox(mode, preview_rgb, preview_depth_mm):
        nonlocal bbox
        if mode == "auto":
            if preview_depth_mm is None:
                print("  no depth frame available yet, can't auto-detect")
                return
            try:
                fx, fy, cx, cy = get_intrinsics()
            except Exception as e:
                print(f"  {e}")
                return
            if preview_depth_mm.shape[:2] != preview_rgb.shape[:2]:
                preview_depth_mm = cv2.resize(preview_depth_mm, (preview_rgb.shape[1], preview_rgb.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
            xyz = bbox_utils.depth_to_xyz(preview_depth_mm, fx, fy, cx, cy)
            picked = bbox_utils.auto_bbox(preview_rgb, xyz, band=args.table_band, pad_frac=args.bbox_pad,
                                          var_thresh=args.var_thresh, lift_mm=args.lift_mm)
            if picked is None:
                print("  no twisted wire found on the table, keeping previous setting")
                return
        else:
            viz_queue.put(('pick_bbox', preview_rgb, display_max))
            picked = viz_result_queue.get()  # blocks until confirmed/cancelled in the viewer process
            if picked is None:
                print("  no box selected, keeping previous setting")
                return
        bbox = picked
        bbox_utils.save_bbox(str(bbox_file), bbox, preview_rgb.shape)
        print(f"  crop box saved to {bbox_file}")

    if args.select_bbox or args.auto_bbox:
        print("Capturing a frame to set the crop box from ...")
        preview_rgb, preview_depth = cap.grab_rgb(timeout=args.wait_timeout)
        pick_bbox("auto" if args.auto_bbox else "manual", preview_rgb, preview_depth)

    processor = load_sam3(args.checkpoint, args.confidence)
    tf_procs = {}  # wire_name -> ros2 static_transform_publisher subprocess, one persistent TF per colour
    frame_idx = 0

    if args.interactive:
        print("\nPress Enter to run capture->segment->grasp->publish, 's' + Enter to hand-pick the "
              "crop box, 'a' + Enter to auto-detect it, 'm' + Enter to reload it, 'q' + Enter to quit.")
    else:
        print("\nAuto mode: waiting for the ready-signal from grasp_executor.py to run "
              "capture->segment->grasp->publish (Ctrl-C to quit).")

    while True:
        if args.interactive:
            key = input(f"[{frame_idx}] run pipeline? ").strip().lower()
            if key == "q":
                break
        else:
            sig_sub.recv()  # blocks until grasp_executor.py's home-both-arms signal arrives
            logger.info(f"[{frame_idx}] ready-signal received -- running pipeline")
            key = ""

        img_rgb, img_depth_mm = cap.grab_rgb(timeout=args.wait_timeout)

        if key in ("s", "a"):
            pick_bbox("auto" if key == "a" else "manual", img_rgb, img_depth_mm)
            continue
        if key == "m":
            loaded = bbox_utils.load_bbox(str(bbox_file))
            if loaded is None:
                print(f"  no saved bbox found at {bbox_file}")
            else:
                bbox = loaded
                print(f"  reloaded bbox from {bbox_file}: {bbox}")
            continue
        if key != "":
            print("  unrecognised input, use blank/s/a/m/q")
            continue

        if img_depth_mm is None:
            print("  no depth frame available yet, skipping this capture")
            continue

        print(f"=== frame {frame_idx} ===")
        wires_this_frame = [w for w in wires if wire_filter is None or w[0] == wire_filter]

        # SAM3 detection is flaky frame-to-frame (auto-exposure, IR glare, momentary blur) --
        # retry with a fresh capture up to --detect_retries times before giving up on this
        # cycle. Only the successful attempt's payloads get published, so this can never
        # double-publish: the loop breaks the moment one attempt produces a payload.
        payloads = None
        for attempt in range(args.detect_retries + 1):
            if attempt > 0:
                print(f"  retrying with a fresh capture (attempt {attempt + 1}/{args.detect_retries + 1})…")
                img_rgb, img_depth_mm = cap.grab_rgb(timeout=args.wait_timeout)
                if img_depth_mm is None:
                    print("  no depth frame available yet, skipping this attempt")
                    continue

            if img_depth_mm.shape[:2] != img_rgb.shape[:2]:
                img_depth_mm = cv2.resize(img_depth_mm, (img_rgb.shape[1], img_rgb.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

            clipped = bbox_utils.clamp_bbox(bbox, img_rgb.shape) if bbox is not None else None
            if bbox is not None and clipped is None:
                print(f"  crop box {bbox} lies outside the {img_rgb.shape[1]}x{img_rgb.shape[0]} frame, skipping")
                continue
            bx, by = (clipped[0], clipped[1]) if clipped else (0, 0)
            rgb = bbox_utils.crop(img_rgb, clipped)
            depth_mm = bbox_utils.crop(img_depth_mm, clipped)

            try:
                fx, fy, cx, cy = get_intrinsics()
            except Exception as e:
                print(f"  {e}")
                continue
            # cx/cy shifted to the crop's own origin -- pixel (0,0) of the crop is
            # pixel (bx,by) of the full frame, same fix as the other two scripts.
            xyz = bbox_utils.depth_to_xyz(depth_mm, fx, fy, cx - bx, cy - by)
            valid = np.isfinite(xyz).all(axis=-1)

            segmented = segment_frame(processor, rgb, xyz, valid, wires_this_frame,
                                      min_confidence=args.min_confidence)

            if not args.no_viz:
                for name, _prompts, _color in wires_this_frame:
                    if name in segmented:
                        mask, points_mm, _colors, matched_prompt = segmented[name]
                        show_mask_overlay(viz_queue, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, matched_prompt,
                                          int(mask.sum()))

            if not segmented:
                print("  no wire detected in this frame")
                continue

            # One grasp per detected colour -- each wire's own top (most-points) mask goes through
            # find_grasp_pose() and gets published independently, so "all of red/blue/yellow in the
            # scene" publishes three grasps, not just the single best one across colours.
            ranked = sorted(segmented.items(), key=lambda kv: -len(kv[1][1]))
            if len(ranked) > 1:
                print(f"  candidates by points: {', '.join(f'{n}={len(p):,}' for n, (_, p, _, _) in ranked)}")

            attempt_payloads = []
            for wire_name, (_mask, points_mm, _colors, _matched_prompt) in ranked:
                result = find_grasp_pose(points_mm, down_cam)
                if result is None:
                    print(f"  {wire_name}: not enough points near its centre for a grasp pose, skipping")
                    continue
                T_grasp_optical_mm, local_pts = result
                print(f"  {wire_name} ({len(points_mm):,} pts) grasp (camera-optical frame, mm): "
                      f"{np.round(T_grasp_optical_mm[:3, 3], 1).tolist()}")

                if not args.no_publish_tf:
                    grasp_frame = f"{args.grasp_frame}_{wire_name}"
                    tf_procs[wire_name] = publish_grasp_tf(T_grasp_optical_mm, args.camera_frame, grasp_frame,
                                                           prev_proc=tf_procs.get(wire_name))

                if grasp_pub is not None:
                    T_grasp_optical_m = T_grasp_optical_mm.copy()
                    T_grasp_optical_m[:3, 3] /= 1000.0
                    T_grasp_base = T_optical_to_base @ T_grasp_optical_m
                    print(f"  {wire_name} grasp (robot base frame, m): {np.round(T_grasp_base[:3, 3], 4).tolist()}")
                    attempt_payloads.append(build_grasp_payload(T_grasp_base, wire_name, frame_idx))

            if not attempt_payloads:
                print("  no candidate wire had enough points near its centre for a grasp pose")
                continue

            payloads = attempt_payloads
            break

        if payloads is None:
            print(f"  giving up on this cycle after {args.detect_retries + 1} attempt(s) -- nothing published")
        elif grasp_pub is not None:
            publish_grasps(grasp_pub, payloads, frame_idx)

        frame_idx += 1

    viz_queue.put(None)
    viz_proc.join(timeout=3)
    for wire_name, tf_proc in tf_procs.items():
        if tf_proc is not None and tf_proc.poll() is None:
            tf_proc.terminate()
        # tf_proc.terminate() alone only kills the `ros2 run` wrapper -- the real
        # static_transform_publisher binary it execs as a child survives that, reparented
        # to init -- see publish_grasp_tf()'s docstring.
        subprocess.run(['pkill', '-f', f'static_transform_publisher.*--child-frame-id {args.grasp_frame}_{wire_name}$'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if grasp_pub is not None:
        grasp_pub.close()
    print(f"\nDone -- {frame_idx} capture(s) processed.")


if __name__ == "__main__":
    main()
    # Skip normal interpreter teardown: once SAM3/CUDA has been loaded in this process and a
    # spawned cv2-GUI child process has actually called imshow() (the bbox picker / mask
    # overlay), letting Python's normal shutdown run aborts with SIGABRT ("terminate called
    # without an active exception") -- reproduced with a minimal repro isolating CUDA init +
    # zmq background threads + a spawned cv2 window; KMP_DUPLICATE_LIB_OK alone did not fix
    # it, only skipping teardown does. All cleanup that matters (publish sockets closed, TF
    # subprocess terminated, viewer process told to quit) already happened inside main().
    #
    # os._exit() skips stdio flushing too (unlike sys.exit()) -- when stdout isn't a tty it's
    # block-buffered, so without this the last print()s (e.g. "Done -- N capture(s)") can be
    # silently lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
