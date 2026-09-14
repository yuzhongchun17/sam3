#!/usr/bin/env python
"""
wire_grasp_pipeline_orbbec_untwist.py
--------------------------------------
COAXIAL variant of wire_grasp_pipeline_orbbec.py: same capture -> segment ->
grasp -> publish loop, but the grasp pose runs the gripper IN LINE with the
wire (approaching from beyond its free tip) instead of straight down onto
its middle, and each publish carries an extra "untwist_deg" field so
grasp_executor_untwist.py can spin the wrist to undo a counted twist after
closing the gripper. No bend step -- unlike SamRobotUntwisting.py (the
Zivid+ABB script this is ported from), the dex rig grasps the wire coaxially
as it lies.

Ported from SamRobotUntwisting.py (Zivid+ABB, EGM joint control):
  wire_tip_and_anchor        -> wire_tip_and_anchor_cam (same PCA-endpoint
                                 idea, but the tip/anchor choice is judged by
                                 distance to the ROBOT BASE origin instead of
                                 |y| in robot frame -- see its docstring)
  get_local_pca_direction    -> same function already in this file, now
                                 signed OUTWARD like the ABB version (needed
                                 for a coaxial approach; unsigned was fine
                                 for a top-down one)
  calculate_wire_orientation -> calculate_coaxial_orientation (same z = wire
                                 axis idea; x seeded from world "up" in
                                 camera-optical coordinates instead of a live
                                 EE quaternion, since this process never
                                 talks to the robot)
  bundle_pcd_from_rotation   -> bundle_points_from_rotation (same idea: the
                                 grasp is fitted to the whole twisted BUNDLE,
                                 not a single coloured strand -- a strand
                                 spirals around the bundle, so its own PCA
                                 direction is that helix's tangent, tilted off
                                 the bundle's true axis by roughly the twist
                                 angle. GRASP_FROM_BUNDLE is not ported as a
                                 toggle: per-colour segmentation only runs as
                                 a fallback when rotation counting fails
                                 outright, same as the ABB script's default)
Rotation counting is WireRotationCounter (segmentation/count_wire_rotations.py
in the barc_wire_sorting repo, a sibling of this workspace at the filesystem
root) run on the same crop this file already uses for segmentation -- no
second bbox file, unlike the ABB script's separate ROTATION_BBOX_FILE. Its own
wire mask (WIRE_PROMPTS: "colorful wire", "twisted wire", ...) is also this
file's grasp geometry -- see bundle_points_from_rotation -- so colour prompts
("red wire", ...) are only ever used for the per-strand rotation count, never
to pick grasp points.

cv2.imshow can't run in this process once SAM3 is loaded (confirmed segfault,
see stream_segment.py's module docstring), so interactive bbox picking ('s')
runs in a separate spawned process exactly like stream_segment.py's
_viewer_process -- SAM3 is loaded once at startup, after which this process
never touches cv2's GUI backend itself.

Prerequisite: pub_orbbec must be running on the Jetson for --camera_name's
camera (richtech-dex-open-cli run-plugin pub_orbbec -i <index>).

Stdin commands (all + Enter):
  <blank>   Capture -> segment -> count rotations -> grasp -> publish, once.
  s         Hand-pick the workspace crop box (drag on a live frame).
  a         Auto-detect the crop box (table plane + hue-variance twist).
  m         Reload the crop box saved at --bbox_file.
  q         Quit.

Run:
    conda activate sam3
    python wire_grasp_pipeline_orbbec_untwist.py
    python wire_grasp_pipeline_orbbec_untwist.py --auto_bbox --wire green_wire
    python wire_grasp_pipeline_orbbec_untwist.py --colors "green wire,black wire" --no_publish_tf
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

#WireRotationCounter lives in the barc_wire_sorting repo (a sibling directory of this
#workspace's root, not a dex-workspace submodule) -- reach into it directly rather than
#vendoring a copy, the same trick SamRobotUntwisting.py uses for the same module.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'barc_wire_sorting', 'segmentation'))
from count_wire_rotations import WireRotationCounter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

WIRES_DEFAULT = [
    ('black_wire', 'black wire', [30, 30, 30]),
    ('green_wire', 'green wire', [40, 180, 60]),
    ('white_wire', 'white wire', [230, 230, 230]),
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

# Same port grasp_executor_untwist.py listens on by default -- this is a drop-in
# alternative source to anygrasp_sam3_stream.py, not something that runs
# alongside it on the same port.
DEFAULT_ZMQ_PUB_ADDR = 'tcp://*:5561'
ZMQ_SLOW_JOINER_DELAY_S = 0.5  # PUB/SUB: give a subscriber time to connect before the first send()

# grasp_executor_untwist.py's --ready_signal_addr default is 'tcp://*:5564' (it binds); this
# is the Jetson's IP, the host that script runs on (same machine as --jetson_ip's camera stream).
DEFAULT_READY_SIGNAL_ADDR = 'tcp://192.168.11.11:5564'

LOCAL_PCA_RADIUS_MM = 30.0
WIRE_WIDTH_MM = 3.0
SIDE_MARGIN_MM = 4.0

# Depth-gap + DBSCAN cloud cleanup, ported from SAMSegmentationClass._filter_indices
# (barc_wire_sorting/sam_wire_grasp_loop) -- see filter_wire_points() below. None disables
# the respective stage, same as SAMSegmenter's own constructor defaults.
DEPTH_GAP_TOL_MM = 50.0
DBSCAN_EPS_MM = 3.0
DBSCAN_MIN_PTS = 10

MASK_WINDOW_NAME = "wire_grasp_pipeline_orbbec_untwist: mask overlay"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--jetson_ip", default="192.168.11.11", help="IP of the Jetson (default: %(default)s)")
    p.add_argument("--color_port", default="10011", help="ZMQ colour port from pub_orbbec (default: %(default)s = eye)")
    p.add_argument("--depth_port", default="10013", help="ZMQ depth port from pub_orbbec (default: %(default)s = eye)")
    p.add_argument("--wait_timeout", type=float, default=10.0,
                   help="Seconds to wait for a colour/depth frame before giving up (default: %(default)s)")

    # Workspace bbox -- same workflow as capture_orbbec_images.py / stream_segment.py. Also the
    # crop the rotation counter runs on, so grasp segmentation and twist counting always look
    # at the identical region.
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
                   help="Fallback confidence retried once per wire colour (cheaply -- reuses the "
                        "cached image encoding) when nothing clears --confidence, so a wire "
                        "changing lighting pushed just under the normal threshold still gets "
                        "found (default: %(default)s)")
    p.add_argument("--no_light_norm", action="store_true",
                   help="Skip white-balance/CLAHE normalization of SAM3's input frame (on by "
                        "default -- corrects colour-temperature/exposure drift between lighting "
                        "setups so colour prompts like 'red wire' stay reliable)")
    p.add_argument("--colors", default=None,
                   help='Comma-separated SAM3 prompts to segment each capture with, e.g. '
                        '"yellow wire,red wire" (default: the 4 standard wire colours)')
    p.add_argument("--wire", default=None,
                   help="Force the grasp target to this wire name (matches --colors, spaces -> underscores), "
                        "skipping the most-points comparison across colours")
    p.add_argument("--no_viz", action="store_true", help="Skip the mask-overlay window shown after each segmentation")
    p.add_argument("--debug_pcd", action="store_true",
                   help="Open a blocking Open3D window of the segmented point cloud (grey), the "
                        "tip/anchor endpoints (red/blue) and the local-PCA neighbourhood used for "
                        "wire_dir (green) before/whether or not a grasp pose is found -- for "
                        "spotting a stray outlier point (mixed wire/background pixel, depth noise) "
                        "that a min/max PCA projection can pick as the tip. Close the window to "
                        "continue to the next capture.")

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
    # grasp_executor_untwist.py (published once its right arm finishes parking).
    p.add_argument("--ready_signal_addr", default=DEFAULT_READY_SIGNAL_ADDR,
                   help="ZMQ address to connect to for the ready-signal from grasp_executor_untwist.py "
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
    palette = {name: color for name, _, color in WIRES_DEFAULT}
    wires = []
    for prompt in (c.strip() for c in colors_arg.split(',') if c.strip()):
        name = prompt.replace(' ', '_')
        wires.append((name, prompt, palette.get(name, [200, 200, 200])))
    return wires


def _find_depth_gap(z_vals, n_bins=50, min_gap_bins=5):
    """Z histogram -> threshold at the furthest significant gap (prevents bleed-over onto
    background behind the wire). None if no clear gap is found. Ported verbatim from
    SAMSegmentationClass._find_depth_gap (barc_wire_sorting/sam_wire_grasp_loop)."""
    hist, edges = np.histogram(z_vals, bins=n_bins)
    occupied = np.where(hist > 0)[0]
    if len(occupied) < 2:
        return None
    peak = np.argmax(hist)
    spacings = np.diff(occupied)
    valid_gaps = np.where((spacings > min_gap_bins) & (occupied[:-1] >= peak))[0]
    if len(valid_gaps) == 0:
        return None
    return edges[occupied[valid_gaps[-1]] + 1]


def filter_wire_points(points_mm, colors, label='wire',
                        depth_tol_mm=DEPTH_GAP_TOL_MM, dbscan_eps=DBSCAN_EPS_MM,
                        dbscan_min_pts=DBSCAN_MIN_PTS):
    """Depth-gap filter + DBSCAN: turns a raw back-projected SAM mask into one clean wire
    cloud, same two stages SamRobotUntwisting.py runs via SAMSegmenter.filter_wire_points
    (SAMSegmentationClass._filter_indices) before trusting a cloud for grasp geometry. Without
    this, a stray mixed wire/background edge pixel or depth-noise spike can hijack
    wire_tip_and_anchor_cam's min/max-along-the-PCA-axis tip pick -- see
    show_debug_pointcloud's docstring.
      depth filter: histogram gap between wire and background; falls back to a fixed
                    tolerance past the near percentile if no clear gap exists. None disables.
      DBSCAN:       keep the nearest cluster (smallest mean Z = closest to camera = the wire),
                    ignoring clusters under 40% of the largest cluster's size. None disables."""
    idx = np.arange(len(points_mm))
    if len(idx) == 0:
        return points_mm, colors

    if depth_tol_mm is not None:
        z = points_mm[:, 2]
        z_thresh = _find_depth_gap(z)
        if z_thresh is None:
            z_thresh = np.percentile(z, 5) + depth_tol_mm
        idx = idx[z[idx] <= z_thresh]

    if dbscan_eps is not None and len(idx) > dbscan_min_pts:
        import open3d as o3d
        pts = points_mm[idx]
        tmp = o3d.geometry.PointCloud()
        tmp.points = o3d.utility.Vector3dVector(pts)
        labels = np.array(tmp.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_pts))
        if labels.max() >= 0:
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            min_size = max(dbscan_min_pts, int(counts.max() * 0.4))
            valid = unique[counts >= min_size]
            mean_z = np.array([pts[labels == c, 2].mean() for c in valid])
            largest = valid[np.argmin(mean_z)]
            idx = idx[labels == largest]

    if len(idx) < len(points_mm):
        logger.info(f'  {label:15s} cloud filter: {len(points_mm)} -> {len(idx)} pts')
    return points_mm[idx], colors[idx]


def segment_frame(processor, rgb, xyz, valid, wires, min_confidence=None, light_norm=True):
    """Runs SAM3 once per wire prompt on `rgb`, in memory -- no PNG/PCD
    written to disk (that's what segment_orbbec_wires.py is for, offline).
    Returns {wire_name: (mask_bool HxW, points_mm Nx3, colors_0to1 Nx3)}.

    SAM3 itself only ever sees `sam_rgb` (white-balanced/CLAHE'd -- see
    bbox_utils.normalize_lighting) when light_norm is set; `rgb` stays
    untouched for the point-cloud colours pulled out below, so a lighting fix
    aimed at SAM3's colour-word prompts can't skew the actual grasp data."""
    from PIL import Image
    sam_rgb = bbox_utils.normalize_lighting(rgb, color_order='rgb') if light_norm else rgb
    state = processor.set_image(Image.fromarray(sam_rgb))
    base_confidence = processor.confidence_threshold

    results = {}
    for name, prompt, _color in wires:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        raw_masks = state.get('masks')
        if (raw_masks is None or len(raw_masks) == 0) and min_confidence is not None \
                and min_confidence < base_confidence:
            # Cheap retry against the cached image encoding -- see stream_segment.py's
            # identical fallback for why (changing lighting can push a real wire just
            # under the configured threshold).
            logger.info(f'  {name:15s} not detected at confidence={base_confidence:.2f}, '
                        f'retrying at {min_confidence:.2f}')
            state = processor.set_confidence_threshold(min_confidence, state)
            raw_masks = state.get('masks')
            processor.set_confidence_threshold(base_confidence)  # restore for the next prompt
        if raw_masks is None or len(raw_masks) == 0:
            logger.info(f'  {name:15s} not detected')
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
        logger.info(f'  {name:15s} {n_pts:7,} pts')
        points_mm, colors = filter_wire_points(points_mm, colors, label=name)
        results[name] = (mask, points_mm, colors)

    return results


def bundle_points_from_rotation(rotation_out, rgb, xyz, valid):
    """Back-projects rotation_counter.analyze()'s own bundle mask (WIRE_PROMPTS: 'colorful
    wire', 'twisted wire', ... -- see count_wire_rotations.py) to camera-frame 3-D points -- no
    extra SAM3 call, analyze() already built this mask while counting rotations. This is the
    grasp geometry for the whole twisted bundle, ported from SamRobotUntwisting.py's
    bundle_pcd_from_rotation() (see module docstring for why the bundle, not a single strand).
    Returns (mask_bool HxW, points_mm Nx3, colors_0to1 Nx3), or None if it back-projects to 0
    valid points."""
    mask = rotation_out['wire_mask']
    H, W = rgb.shape[:2]
    if mask.shape != (H, W):
        # analyze() downscales internally past its max_side default -- paste the mask back onto
        # this frame's own pixel grid (NEAREST: it's a label mask, interpolating would invent
        # edge pixels that back-project to the wrong depth) before it can index xyz/valid.
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    sel = mask & valid
    if not sel.any():
        return None
    points_mm = xyz[sel] * 1000.0  # metres -> mm, matching segment_frame()'s convention
    colors = rgb[sel].astype(np.float64) / 255.0
    points_mm, colors = filter_wire_points(points_mm, colors, label='bundle')
    return mask, points_mm, colors


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


# ── Grasp geometry: COAXIAL, ported from SamRobotUntwisting.py (see module docstring). All of
# it stays in CAMERA-OPTICAL frame (mm), same as the top-down version it replaces -- only the
# tip/anchor judgement below needs a peek at robot-base distance, and it converts just the two
# candidate points to do that rather than moving the whole pipeline into base frame. ──

def get_local_pca_direction(centre_point, points_mm, radius_mm=LOCAL_PCA_RADIUS_MM):
    """Local wire direction at centre_point, SIGNED OUTWARD -- away from the wire body and
    toward centre_point -- so a coaxial approach (z = -direction, see
    calculate_coaxial_orientation) always comes from beyond the free end, never from beyond the
    anchor. The top-down grasp this replaces didn't care about this sign (jaws land the same
    either way); a coaxial one very much does, so the anchoring is added here rather than left
    to whichever way SVD happens to return vt[0]."""
    distances = np.linalg.norm(points_mm - centre_point, axis=1)
    local_points = points_mm[distances <= radius_mm]
    if len(local_points) < 3:
        logger.warning(f"Only {len(local_points)} points within {radius_mm}mm of the grasp "
                        f"centre, cannot fit a local direction. Skipping this wire.")
        return None, None
    centroid = local_points.mean(axis=0)
    _, _, vt = np.linalg.svd(local_points - centroid)
    direction = vt[0]

    overall_centroid = points_mm.mean(axis=0)
    outward_ref = centre_point - overall_centroid
    if np.dot(direction, outward_ref) < 0:
        direction = -direction
    return local_points, direction


def wire_tip_and_anchor_cam(points_mm, T_optical_to_base):
    """Both ends of the wire along its principal axis, in CAMERA-OPTICAL frame (mm): the free
    TIP a coaxial grasp approaches from beyond, and the ANCHOR it pivots about (unused today,
    kept because it falls out of the same fit for free -- see SamRobotUntwisting.py).

    SamRobotUntwisting.py picked the tip by comparing |y| in ROBOT frame, which relied on
    knowing which way the fixture sits relative to the ABB's base. That axis convention doesn't
    carry over to the dex rig, so this instead picks whichever endpoint sits FARTHER FROM THE
    ROBOT BASE ORIGIN -- converting just the two candidate endpoints through T_optical_to_base,
    not the whole cloud, so the rest of this file's geometry stays in camera-optical frame.
    Confirmed against the dex fixture: the anchor (fixture clamp) sits closer to the base, the
    free tip swings out farther."""
    if points_mm.shape[0] < 20:
        return None, None
    centroid = points_mm.mean(axis=0)
    _, _, vt = np.linalg.svd(points_mm - centroid)
    direction = vt[0]
    t = (points_mm - centroid) @ direction
    # Actual data points at the extremes, NOT their projection onto the fitted line
    # (centroid + t.min()*direction): the bundle mask covers a whole twisted bundle, which can
    # bow/curve well past LOCAL_PCA_RADIUS_MM off a single straight-line fit, especially near a
    # loose free tip -- get_local_pca_direction's neighbourhood search would then centre on empty
    # space next to the real cloud and find 0 points. Indexing the real point guarantees at least
    # itself (and whatever real neighbours are actually there) falls inside that search.
    endpoint_1 = points_mm[int(np.argmin(t))]
    endpoint_2 = points_mm[int(np.argmax(t))]

    def dist_to_base_origin(p_mm):
        p_h = np.append(p_mm / 1000.0, 1.0)
        p_base = T_optical_to_base @ p_h
        return float(np.linalg.norm(p_base[:3]))

    if dist_to_base_origin(endpoint_1) > dist_to_base_origin(endpoint_2):
        return endpoint_1, endpoint_2
    return endpoint_2, endpoint_1


def calculate_coaxial_orientation(wire_dir, up):
    """COAXIAL grasp frame: tool z runs ALONG the wire, travelling from beyond the tip INTO it
    (ported from SamRobotUntwisting.py's calculate_wire_orientation). wire_dir is already
    signed outward by get_local_pca_direction, so the approach/insertion direction is its
    negation.
    x has no live EE orientation to seed it from roll-to-roll -- this process never talks to
    the robot, see the module docstring -- so it is built from `up` (world "up" expressed in
    camera-optical coordinates, i.e. -down_cam) instead. Any perpendicular is a valid x here:
    the grasp is rotationally symmetric about z."""
    pca_unit = wire_dir / np.linalg.norm(wire_dir)
    z_axis = -pca_unit

    x_axis = up - np.dot(up, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        #Wire is parallel to `up`: it has no component perpendicular to it, so fall back to
        #whichever camera axis isn't parallel to z either.
        x_axis = np.cross([1.0, 0.0, 0.0], z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)   #right-handed: z cross x = y
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def show_debug_pointcloud(points_mm, colors=None, tip=None, anchor=None, local_pts=None,
                           T_grasp=None, title="segmented point cloud"):
    """Blocking Open3D window of the cloud find_coaxial_grasp_pose() is about to trust, so a
    stray outlier (a mixed wire/background edge pixel, a depth-noise spike) can be SEEN before
    blaming "not enough points near the tip" on sensor dropout -- wire_tip_and_anchor_cam's
    min/max-along-the-PCA-axis pick is exactly the kind of thing one bad far-flung point can hijack.
    Runs in THIS process, unlike the mask-overlay window: the segfault documented at the top of
    this file is specific to cv2's Qt GUI backend after SAM3 loads, not GL/GLFW in general --
    SamRobotUntwisting.py opens Open3D windows the same way, in the same process as its own SAM3
    model.
      grey    = every segmented point
      red     = tip (wire_tip_and_anchor_cam's pick)
      blue    = anchor
      green   = the local-PCA neighbourhood get_local_pca_direction found near the tip (empty/
                too small green cluster around an isolated red point is the outlier signature)
      triad   = the fitted grasp frame, if one was found
    Close the window to continue to the next capture."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_mm)
    pcd.colors = (o3d.utility.Vector3dVector(colors) if colors is not None
                  else o3d.utility.Vector3dVector(np.full((len(points_mm), 3), 0.5)))
    geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)]  # camera-optical origin

    def marker(xyz, color, radius):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(xyz)
        s.paint_uniform_color(color)
        s.compute_vertex_normals()
        return s

    if tip is not None:
        geoms.append(marker(tip, [1.0, 0.0, 0.0], radius=3.0))
        print(f"  [debug pcd] tip (red):     {np.round(tip, 1).tolist()}")
    if anchor is not None:
        geoms.append(marker(anchor, [0.0, 0.0, 1.0], radius=3.0))
        print(f"  [debug pcd] anchor (blue): {np.round(anchor, 1).tolist()}")
    if local_pts is not None and len(local_pts):
        local_pcd = o3d.geometry.PointCloud()
        local_pcd.points = o3d.utility.Vector3dVector(local_pts)
        local_pcd.paint_uniform_color([0.0, 1.0, 0.0])
        geoms.append(local_pcd)
        print(f"  [debug pcd] {len(local_pts)} local pts (green) near the tip")
    if T_grasp is not None:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15.0)
        frame.transform(T_grasp)
        geoms.append(frame)

    print(f"  [debug pcd] {title} -- {len(points_mm):,} pts total, close the window to continue")
    o3d.visualization.draw_geometries(geoms, window_name=title)


def find_coaxial_grasp_pose(points_mm, up_cam, T_optical_to_base, radius_mm=LOCAL_PCA_RADIUS_MM,
                             debug=False, debug_colors=None, debug_title="segmented point cloud"):
    """tip/anchor -> local PCA direction at the tip -> coaxial frame, all in camera-optical
    frame (mm). Returns (T_grasp_cam_mm, local_pts, tip, anchor) or None.

    debug=True opens show_debug_pointcloud() on every path out of this function, success or
    failure alike -- a failure still has SOMETHING worth looking at (the raw cloud, or the raw
    cloud plus the tip that couldn't find neighbours), and that is the case debugging is for."""
    if len(points_mm) == 0:
        return None
    tip, anchor = wire_tip_and_anchor_cam(points_mm, T_optical_to_base)
    if tip is None:
        if debug:
            show_debug_pointcloud(points_mm, debug_colors,
                                  title=f"{debug_title} (too few points for a global PCA fit)")
        return None
    local_pts, wire_dir = get_local_pca_direction(tip, points_mm, radius_mm)
    if local_pts is None:
        if debug:
            show_debug_pointcloud(points_mm, debug_colors, tip=tip, anchor=anchor,
                                  title=f"{debug_title} (local PCA failed near the tip)")
        return None
    R = calculate_coaxial_orientation(wire_dir, up_cam)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = local_pts.mean(axis=0)
    if debug:
        show_debug_pointcloud(points_mm, debug_colors, tip=tip, anchor=anchor,
                              local_pts=local_pts, T_grasp=T, title=debug_title)
    return T, local_pts, tip, anchor


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


def camera_up_vector(camera_extrinsic_json):
    """World 'up' expressed in camera-optical coordinates -- the x-axis seed for
    calculate_coaxial_orientation. Same extrinsic the top-down version used for 'down'; this is
    just its negation, kept as its own function so the sign is named rather than inlined."""
    with open(camera_extrinsic_json) as f:
        extr = json.load(f)
    R_base_to_optical = _rpy_deg_to_matrix(*extr['rpy_deg'])
    up_cam = R_base_to_optical.T @ np.array([0.0, 0.0, 1.0])
    return up_cam / np.linalg.norm(up_cam)


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


def build_grasp_payload(T_grasp_base, wire_name, frame_idx, untwist_deg=0.0):
    """Same per-grasp schema as wire_grasp_pipeline_orbbec.py's build_grasp_payload(), plus
    "untwist_deg" -- the magnitude, in degrees, grasp_executor_untwist.py should spin the wrist
    by (in its own fixed, hardcoded direction -- see that script) to undo the counted twist.
    0.0 means no twist was counted (or none could be)."""
    jaw_clearance_m = (WIRE_WIDTH_MM + 2 * SIDE_MARGIN_MM) / 1000.0
    return {
        "translation":  T_grasp_base[:3, 3].tolist(),
        "rotation":     T_grasp_base[:3, :3].tolist(),
        "score":        1.0,
        "width":        jaw_clearance_m,
        "frame_idx":    frame_idx,
        "prompt":       wire_name,
        "frame":        "base",
        "stem":         f"live_{frame_idx:03d}",
        "untwist_deg":  float(untwist_deg),
    }


def publish_grasps(pub_socket, payloads, frame_idx):
    """Send every wire's grasp from this capture as ONE message, {"frame_idx":
    ..., "grasps": [payload, ...]} -- not one publish_grasp() call per wire.
    The SUB side (grasp_executor_untwist.py) runs with CONFLATE=1, which keeps only
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

    up_cam = camera_up_vector(args.camera_extrinsic_json)
    logger.info(f"World 'up' in camera-optical frame: [{up_cam[0]:+.3f}, {up_cam[1]:+.3f}, {up_cam[2]:+.3f}]")
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
    #Reuses `processor` (the same loaded SAM3 model/weights) instead of loading a second
    #checkpoint onto the GPU -- see SamRobotUntwisting.py's identical use of seg.processor.
    rotation_counter = WireRotationCounter(processor=processor)
    tf_procs = {}  # wire_name -> ros2 static_transform_publisher subprocess, one persistent TF per colour
    frame_idx = 0

    if args.interactive:
        print("\nPress Enter to run capture->segment->grasp->publish, 's' + Enter to hand-pick the "
              "crop box, 'a' + Enter to auto-detect it, 'm' + Enter to reload it, 'q' + Enter to quit.")
    else:
        print("\nAuto mode: waiting for the ready-signal from grasp_executor_untwist.py to run "
              "capture->segment->grasp->publish (Ctrl-C to quit).")

    while True:
        if args.interactive:
            key = input(f"[{frame_idx}] run pipeline? ").strip().lower()
            if key == "q":
                break
        else:
            sig_sub.recv()  # blocks until grasp_executor_untwist.py's right-arm-park signal arrives
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

        print(f"=== frame {frame_idx} ===")
        wires_this_frame = [w for w in wires if wire_filter is None or w[0] == wire_filter]

        # Grasp geometry comes from the whole twisted BUNDLE (rotation_counter's own wire mask),
        # not a single coloured strand -- see module docstring / bundle_points_from_rotation.
        # Colour only decides which strand's rotation count to apply; it never picks grasp points
        # in this, the common, path.
        rotation_colors = [name.split('_')[0] for name, _prompt, _clr in wires_this_frame]
        sam_rgb = bbox_utils.normalize_lighting(rgb, color_order='rgb') if not args.no_light_norm else rgb
        try:
            rotation_out = rotation_counter.analyze(sam_rgb, colors=rotation_colors)
        except RuntimeError as exc:
            print(f"  rotation count failed on this crop ({exc}) -- "
                  f"falling back to per-colour segmentation for grasp geometry")
            rotation_out = None

        bundle = bundle_points_from_rotation(rotation_out, rgb, xyz, valid) if rotation_out is not None else None

        candidates = []  # [(wire_label, points_mm, rotations, colors_0to1), ...]
        if bundle is not None:
            mask, points_mm, colors_arr = bundle
            if not args.no_viz:
                show_mask_overlay(viz_queue, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask,
                                  rotation_out['results']['wire_prompt'], int(mask.sum()))
            if wire_filter is not None:
                color = wire_filter.split('_')[0]  # always in strand_masks: color came from rotation_colors above
                strand_px = int(rotation_out['strand_masks'][color].sum())
                if strand_px == 0:
                    #The bundle WAS found -- only the requested colour was not in it. Grasp it
                    #anyway (matches SamRobotUntwisting.py's default REQUIRE_COLOR_STRAND=False):
                    #the untwist just comes back 0 since there is no strand to count rotations on.
                    print(f"  no '{color} wire' strand inside the bundle (found via "
                          f"'{rotation_out['results']['wire_prompt']}') -- grasping the bundle "
                          f"anyway, untwist will be 0 rotations")
                rotations = rotation_out['results']['per_strand_rotations'].get(color, 0.0)
                wire_label = wire_filter
            else:
                rotations = rotation_out['results']['final_rotations']
                wire_label = rotation_out['results']['wire_prompt'].replace(' ', '_')
            candidates.append((wire_label, points_mm, rotations, colors_arr))
        else:
            # No bundle mask (rotation counting failed outright, e.g. a mis-placed crop box) --
            # the one case the per-colour SAM3 pass is still worth paying for, and its cloud is
            # the only grasp geometry left. Same fallback as SamRobotUntwisting.py.
            segmented = segment_frame(processor, rgb, xyz, valid, wires_this_frame,
                                      min_confidence=args.min_confidence, light_norm=not args.no_light_norm)
            if not args.no_viz:
                for name, prompt, _color in wires_this_frame:
                    if name in segmented:
                        seg_mask, seg_points_mm, _ = segmented[name]
                        show_mask_overlay(viz_queue, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), seg_mask,
                                          prompt, int(seg_mask.sum()))
            if not segmented:
                print("  no wire detected in this frame")
                frame_idx += 1
                continue
            ranked = sorted(segmented.items(), key=lambda kv: -len(kv[1][1]))
            if len(ranked) > 1:
                print(f"  candidates by points: {', '.join(f'{n}={len(p):,}' for n, (_, p, _) in ranked)}")
            candidates = [(name, points_mm, 0.0, colors_arr)
                         for name, (_mask, points_mm, colors_arr) in ranked]

        published_any = False
        payloads = []
        for wire_name, points_mm, rotations, colors_arr in candidates:
            result = find_coaxial_grasp_pose(points_mm, up_cam, T_optical_to_base,
                                             debug=args.debug_pcd, debug_colors=colors_arr,
                                             debug_title=f"{wire_name} ({len(points_mm):,} pts)")
            if result is None:
                print(f"  {wire_name}: not enough points near a wire end for a coaxial grasp pose, skipping")
                continue
            published_any = True
            T_grasp_optical_mm, local_pts, tip, anchor = result
            untwist_deg = round(rotations) * 180.0

            print(f"  {wire_name} ({len(points_mm):,} pts) coaxial grasp (camera-optical frame, mm): "
                  f"{np.round(T_grasp_optical_mm[:3, 3], 1).tolist()}"
                  + (f", counted {rotations:.1f} rotations -> untwist {untwist_deg:.0f} deg"
                     if untwist_deg else ""))

            if not args.no_publish_tf:
                grasp_frame = f"{args.grasp_frame}_{wire_name}"
                tf_procs[wire_name] = publish_grasp_tf(T_grasp_optical_mm, args.camera_frame, grasp_frame,
                                                       prev_proc=tf_procs.get(wire_name))

            if grasp_pub is not None:
                T_grasp_optical_m = T_grasp_optical_mm.copy()
                T_grasp_optical_m[:3, 3] /= 1000.0
                T_grasp_base = T_optical_to_base @ T_grasp_optical_m
                print(f"  {wire_name} grasp (robot base frame, m): {np.round(T_grasp_base[:3, 3], 4).tolist()}")
                payloads.append(build_grasp_payload(T_grasp_base, wire_name, frame_idx, untwist_deg))

        if not published_any:
            print("  no candidate wire had enough points near an end for a coaxial grasp pose")
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
