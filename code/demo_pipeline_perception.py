#!/usr/bin/env python
"""
demo_pipeline_perception.py
----------------------------
Merge of wire_grasp_pipeline_orbbec.py (perpendicular/top-down grasp) and
wire_grasp_pipeline_orbbec_untwist.py (coaxial grasp + untwist counting) into
ONE persistent process: SAM3 loads exactly once at startup, the robot-facing
executor (demo_pipeline_executor.py, richtech-dex-open repo) then drives which
of the two detection paths runs on each trigger via a `mode` field, instead of
each variant needing its own separate SAM3-loading process.

Trigger protocol (replaces both source scripts' bare {'ready': True} signal
and their --interactive blank-Enter/s/a/m/q stdin loop -- this script is
demo-only, always auto-triggered): a ZMQ SUB on --trigger_addr receives
msgpack {'mode': 'perpendicular' | 'coaxial', ...} from
demo_pipeline_executor.py. 'perpendicular' runs wire_grasp_pipeline_orbbec.py's
segment_frame_perpendicular()+find_grasp_pose() (multi-prompt-per-wire, no
point-cloud cleanup, untwist_deg left at 0.0). 'coaxial' runs
wire_grasp_pipeline_orbbec_untwist.py's WireRotationCounter/bundle-mask path
+ find_coaxial_grasp_pose(), falling back to its own single-prompt
segment_frame_coaxial_fallback() (with filter_wire_points cleanup) when
rotation counting fails outright -- same as that script's own behaviour.
Either mode's cycle runs the same bounded --detect_retries retry loop (fresh
capture + re-detect, publish exactly once on whichever attempt succeeds) both
source scripts already had.

Everything else duplicated near-identically between the two source scripts
(OrbbecCapture, the spawned bbox-picker/mask-overlay viewer process, camera
intrinsics/extrinsics loaders, publish_grasp_tf/publish_grasps, load_sam3)
appears exactly once here.

Prerequisite: pub_orbbec must be running on the Jetson for --camera_name's
camera (richtech-dex-open-cli run-plugin pub_orbbec -i <index>), and
demo_pipeline_executor.py must be running on the host named by
--trigger_addr for anything to ever trigger a cycle.

Run:
    conda activate sam3
    python demo_pipeline_perception.py
    python demo_pipeline_perception.py --auto_bbox
"""

import os

# Must be set before torch/sam3 (loaded lazily in load_sam3()) or cv2 (imported next) touch
# OpenMP -- see wire_grasp_pipeline_orbbec(_untwist).py's identical comment / stream_segment.py's
# module docstring for the SIGABRT this avoids.
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

# WireRotationCounter lives in the barc_wire_sorting repo (a sibling directory of this
# workspace's root, not a dex-workspace submodule) -- reach into it directly rather than
# vendoring a copy, same trick wire_grasp_pipeline_orbbec_untwist.py / SamRobotUntwisting.py use.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'barc_wire_sorting', 'segmentation'))
from count_wire_rotations import WireRotationCounter

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Only used to look up a viz colour when --colors names one of these explicitly.
WIRE_COLOR_PALETTE = [
    ('yellow_wire', 'yellow wire', [255, 215, 0]),
    ('red_wire',    'red wire',    [220, 50, 50]),
    ('blue_wire',   'blue wire',   [30, 144, 255]),
    ('white_wire',  'white wire',  [230, 230, 230]),
]

# Perpendicular mode's default target is the whole twisted bundle -- (name, [prompts...], color).
WIRES_DEFAULT_PERPENDICULAR = [
    ('twisted_wire', [
        # Generic, shape-agnostic prompts FIRST -- a live-capture probe (2026-09-13) found
        # these hit reliably at the base confidence with a large, clean mask, while the
        # shape-specific prompts below often MISS outright (e.g. on a segment that isn't
        # visibly twisted/coiled/bundled) and, worse, when they DO hit it's usually only via
        # segment_frame_perpendicular's own per-prompt confidence fallback -- a much smaller,
        # noisier mask than the generic prompts give directly. So specific-first was actually
        # backwards: it wasn't producing a tighter mask, just a worse one that happened to
        # clear a lower bar. Kept below only in case a genuinely twisted/coiled bundle benefits
        # from the more descriptive prompt matching first.
        'wire',
        'cable',
        'wires',
        'twisted wire',
        'twisted wires',
        'bundle of wires',
        'wire bundle',
        'coiled wire',
        'cable bundle',
    ], [200, 200, 200]),
]

# Coaxial mode's default is the 4 standard per-colour prompts -- (name, prompt_str, color); only
# used for the rotation-count fallback path (segment_frame_coaxial_fallback) and for selecting
# which strand's rotation count to apply -- the bundle mask itself never depends on colour.
WIRES_DEFAULT_COAXIAL = [
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

# Same port both grasp_executor.py and grasp_executor_untwist.py listen on by default.
DEFAULT_ZMQ_PUB_ADDR = 'tcp://*:5561'
ZMQ_SLOW_JOINER_DELAY_S = 0.5  # PUB/SUB: give a subscriber time to connect before the first send()

# demo_pipeline_executor.py's --ready_signal_addr default is 'tcp://*:5564' (it binds); this is
# the Jetson's IP, the host that script runs on (same machine as --jetson_ip's camera stream).
DEFAULT_TRIGGER_ADDR = 'tcp://192.168.11.11:5564'

# Perpendicular (top-down) grasp geometry uses an unsigned local PCA direction at the wire's
# centroid -- a small neighbourhood is enough. Coaxial (in-line) grasp geometry needs a much
# wider neighbourhood at the wire's free TIP, signed outward -- see get_local_pca_direction_coaxial.
LOCAL_PCA_RADIUS_MM_PERPENDICULAR = 10.0
LOCAL_PCA_RADIUS_MM_COAXIAL = 30.0
WIRE_WIDTH_MM = 3.0
SIDE_MARGIN_MM = 4.0

# Depth-gap + DBSCAN cloud cleanup (coaxial path only), ported from
# SAMSegmentationClass._filter_indices (barc_wire_sorting/sam_wire_grasp_loop).
DEPTH_GAP_TOL_MM = 50.0
DBSCAN_EPS_MM = 3.0
DBSCAN_MIN_PTS = 10

MASK_WINDOW_NAME = "demo_pipeline_perception: mask overlay"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--jetson_ip", default="192.168.11.11", help="IP of the Jetson (default: %(default)s)")
    p.add_argument("--color_port", default="10011", help="ZMQ colour port from pub_orbbec (default: %(default)s = eye)")
    p.add_argument("--depth_port", default="10013", help="ZMQ depth port from pub_orbbec (default: %(default)s = eye)")
    p.add_argument("--wait_timeout", type=float, default=10.0,
                   help="Seconds to wait for a colour/depth frame before giving up (default: %(default)s)")
    p.add_argument("--detect_retries", type=int, default=2,
                   help="Extra attempts (fresh capture + re-detect) if a cycle finds no wire/grasp "
                        "pose, before giving up on it -- SAM3 detection is flaky frame-to-frame "
                        "(default: %(default)s, i.e. 3 attempts total)")

    # Workspace bbox -- same workflow as capture_orbbec_images.py / stream_segment.py. TWO
    # independent crops, not one: the perpendicular stage looks at the wire lying on the table,
    # the coaxial stage looks at the SAME wire after the right arm has picked it up and carried
    # it to --right_park_deg -- a completely different region of the camera's field of view. A
    # single shared bbox (the original single-mode scripts' assumption) is wrong for this merged
    # pipeline; each stage gets its own --bbox*/--bbox_file*/--select_bbox* below.
    p.add_argument("--bbox", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                   help="Perpendicular-stage crop box (wire on the table)")
    p.add_argument("--bbox_file", default=None,
                   help="Where the perpendicular-stage crop box is stored (default: "
                        "bbox_perpendicular.json next to this script -- same file "
                        "wire_grasp_pipeline_orbbec.py defaults to)")
    p.add_argument("--select_bbox", action="store_true",
                   help="Hand-pick the perpendicular-stage crop box before the main loop starts")
    p.add_argument("--auto_bbox", action="store_true",
                   help="Auto-detect the perpendicular-stage crop box before the main loop starts "
                        "(table-plane heuristic -- only meaningful for the on-table view)")
    p.add_argument("--bbox_coaxial", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                   help="Coaxial-stage crop box (wire held by the right arm at --right_park_deg)")
    p.add_argument("--bbox_file_coaxial", default=None,
                   help="Where the coaxial-stage crop box is stored (default: bbox_coaxial.json "
                        "next to this script -- same file wire_grasp_pipeline_orbbec_untwist.py "
                        "defaults to)")
    p.add_argument("--select_bbox_coaxial", action="store_true",
                   help="Hand-pick the coaxial-stage crop box before the main loop starts -- jog/hold the "
                        "right arm at --right_park_deg with a wire in its gripper first, so the preview "
                        "frame this captures actually shows the presented-wire view")
    p.add_argument("--full", action="store_true", help="Ignore any saved crop box (both stages) for this run")
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
                        "cached image encoding) when nothing clears --confidence, for both modes "
                        "(default: %(default)s)")
    p.add_argument("--no_light_norm", action="store_true",
                   help="Skip white-balance/CLAHE normalization of SAM3's input frame in coaxial "
                        "mode (on by default -- corrects colour-temperature/exposure drift so "
                        "colour prompts like 'red wire' stay reliable; perpendicular mode never "
                        "normalizes, matching wire_grasp_pipeline_orbbec.py)")
    p.add_argument("--colors", default=None,
                   help='Comma-separated SAM3 prompts to segment each capture with, e.g. '
                        '"yellow wire,red wire" -- applies to BOTH modes (default: perpendicular '
                        'targets the whole bundle, coaxial targets the 4 standard wire colours)')
    p.add_argument("--wire", default=None,
                   help="Force the grasp target to this wire name (matches --colors, spaces -> underscores), "
                        "skipping the most-points comparison across colours, in either mode")
    p.add_argument("--no_viz", action="store_true", help="Skip the mask-overlay window shown after each segmentation")
    p.add_argument("--debug_pcd", action="store_true",
                   help="Coaxial mode only: open a blocking Open3D window of the segmented point "
                        "cloud/tip/anchor/local-PCA neighbourhood before/whether or not a grasp "
                        "pose is found. Close the window to continue.")

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

    # Trigger: demo_pipeline_executor.py tells this process which mode to run each cycle.
    p.add_argument("--trigger_addr", default=DEFAULT_TRIGGER_ADDR,
                   help="ZMQ address to connect to for the {'mode': 'perpendicular'|'coaxial'} "
                        "trigger from demo_pipeline_executor.py -- must point at the host running "
                        "that script, not this one (default: %(default)s)")

    return p.parse_args()


# ── Capture: ZMQ colour+depth subscriber, ported from capture_orbbec_images.py ──

class OrbbecCapture:
    """Background colour+depth ZMQ subscriber -- same pattern as
    stream_segment.py's DualStreamViewer, decoupled from SAM3/the viewer."""

    def __init__(self, color_ip_port: str, depth_ip_port: str):
        self.context = zmq.Context()
        self.rgb_frame = None
        self.depth_frame = None
        self.rgb_ts = 0.0    # time.time() a frame last actually arrived, not just "was cached"
        self.depth_ts = 0.0  # -- see grab_rgb()'s freshness wait for why this matters
        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        self._color_ip_port = color_ip_port
        self._depth_ip_port = depth_ip_port
        self._stop_event = threading.Event()
        threading.Thread(target=self._color_thread, args=(self._stop_event,), daemon=True).start()
        threading.Thread(target=self._depth_thread, args=(self._stop_event,), daemon=True).start()

    def reconnect(self):
        """Tear down the current background subscriber threads/sockets and start fresh ones,
        discarding any cached frame -- for when "freshest available" isn't a strong enough
        guarantee and a capture needs to come from a brand-new connection with no possible
        carry-over at all (e.g. right before a stage where a stale-looking result keeps
        recurring despite the frame-level freshness/stability checks in grab_rgb() already
        passing). The old threads see their stop_event set and exit on their next loop tick;
        new ones get a fresh Event so they're never confused with the old generation."""
        self._stop_event.set()
        with self.rgb_lock:
            self.rgb_frame = None
            self.rgb_ts = 0.0
        with self.depth_lock:
            self.depth_frame = None
            self.depth_ts = 0.0
        self._stop_event = threading.Event()
        threading.Thread(target=self._color_thread, args=(self._stop_event,), daemon=True).start()
        threading.Thread(target=self._depth_thread, args=(self._stop_event,), daemon=True).start()

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

    def _color_thread(self, stop_event):
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(f"tcp://{self._color_ip_port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        while not stop_event.is_set():
            try:
                frame = self._decode(sub.recv(zmq.NOBLOCK), is_depth=False)
                if frame is not None:
                    with self.rgb_lock:
                        self.rgb_frame = frame
                        self.rgb_ts = time.time()
            except zmq.error.Again:
                time.sleep(0.005)
        sub.close()

    def _depth_thread(self, stop_event):
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(f"tcp://{self._depth_ip_port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        while not stop_event.is_set():
            try:
                frame = self._decode(sub.recv(zmq.NOBLOCK), is_depth=True)
                if frame is not None:
                    with self.depth_lock:
                        self.depth_frame = frame
                        self.depth_ts = time.time()
            except zmq.error.Again:
                time.sleep(0.005)
        sub.close()

    def snapshot(self):
        with self.rgb_lock:
            color_bgr, rgb_ts = self.rgb_frame, self.rgb_ts
            color_bgr = color_bgr.copy() if color_bgr is not None else None
        with self.depth_lock:
            depth_mm, depth_ts = self.depth_frame, self.depth_ts
            depth_mm = depth_mm.copy() if depth_mm is not None else None
        return color_bgr, depth_mm, rgb_ts, depth_ts

    def grab_rgb(self, timeout=10.0, min_ts=None, depth_stability_mm=15.0, depth_stability_gap_s=0.15):
        """One colour+depth grab -> (HxWx3 uint8 RGB, HxW uint16 depth_mm).
        Waits for BOTH streams -- see capture_orbbec_images.py's grab_rgb for why.

        Also waits for both to be FRESH: a background thread just caches whatever ZMQ message
        arrives, with no guarantee color and depth update at the same rate -- if depth stalls
        (sensor mode, IR interference, load during robot motion) while color keeps updating,
        the old "return whatever's non-None" version would happily hand back a stale depth
        frame paired with a brand-new color frame. That's exactly the failure mode where SAM3
        returns a good mask (correct, fresh color) but the depth backing it is from well before
        this call, e.g. a wire that visibly moved but a mask lifted into 3-D against the old
        table-only depth. min_ts (default: this call's own start time) is the cutoff -- both
        frames' arrival timestamps must be >= min_ts, i.e. actually arrived, not just present,
        since grab_rgb was invoked.

        Freshness alone isn't enough right after the SCENE itself changes (not just this call):
        a structured-light/active-stereo depth sensor needs a beat to re-settle once its target
        actually moves (new IR pattern lock, auto-exposure), so the very first frame that clears
        the freshness bar can still be mid-transition -- correctly "new", but not yet correct.
        Only reproduced this against a real scene change over a long-lived process, never a
        short, fresh single-shot capture, which is exactly consistent with a settling delay, not
        a one-off frame-staleness bug. So also require two consecutive fresh depth reads,
        depth_stability_gap_s apart, to agree (median abs diff <= depth_stability_mm over pixels
        valid in both) before trusting either -- if the sensor is still resettling, consecutive
        reads disagree and this keeps waiting; once it's stable, they match and we return."""
        start = time.time()
        min_ts = start if min_ts is None else min_ts
        color_bgr = depth_mm = None
        prev_depth_for_stability = None
        while time.time() - start < timeout:
            color_bgr, depth_mm, rgb_ts, depth_ts = self.snapshot()
            if color_bgr is not None and depth_mm is not None and rgb_ts >= min_ts and depth_ts >= min_ts:
                if prev_depth_for_stability is not None:
                    both_valid = (prev_depth_for_stability > 0) & (depth_mm > 0)
                    if not both_valid.any() or np.median(np.abs(
                            depth_mm[both_valid].astype(np.float64)
                            - prev_depth_for_stability[both_valid].astype(np.float64))) <= depth_stability_mm:
                        return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB), depth_mm
                # Not stable yet (or this is the first fresh candidate) -- remember it and demand
                # a genuinely NEWER frame next time, not an immediate re-read of this same one
                # (which would trivially "match" itself and short-circuit the stability check).
                prev_depth_for_stability = depth_mm.copy()
                min_ts = time.time()
                time.sleep(depth_stability_gap_s)
                continue
            time.sleep(0.01)
        if color_bgr is not None and depth_mm is None:
            logger.warning(f"Got colour but no depth after {timeout:.0f}s -- proceeding without depth.")
            return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB), None
        if color_bgr is not None:
            logger.warning(f"Got colour+depth but neither refreshed within {timeout:.0f}s of this "
                           f"call -- proceeding with what's cached (may be stale).")
            return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB), depth_mm
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


# ── Segmentation: SAM3 load, called exactly ONCE from main() before the trigger loop ──

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


def parse_colors_arg_perpendicular(colors_arg):
    """-> [(name, [prompts...], color), ...], matching WIRES_DEFAULT_PERPENDICULAR's shape --
    one prompt per --colors entry (no synonym list, that's only for the built-in default)."""
    if not colors_arg:
        return WIRES_DEFAULT_PERPENDICULAR
    palette = {name: color for name, _, color in WIRE_COLOR_PALETTE}
    wires = []
    for prompt in (c.strip() for c in colors_arg.split(',') if c.strip()):
        name = prompt.replace(' ', '_')
        wires.append((name, [prompt], palette.get(name, [200, 200, 200])))
    return wires


def parse_colors_arg_coaxial(colors_arg):
    """-> [(name, prompt_str, color), ...], matching WIRES_DEFAULT_COAXIAL's shape."""
    if not colors_arg:
        return WIRES_DEFAULT_COAXIAL
    palette = {name: color for name, _, color in WIRES_DEFAULT_COAXIAL}
    wires = []
    for prompt in (c.strip() for c in colors_arg.split(',') if c.strip()):
        name = prompt.replace(' ', '_')
        wires.append((name, prompt, palette.get(name, [200, 200, 200])))
    return wires


def segment_frame_perpendicular(processor, rgb, xyz, valid, wires, min_confidence=None):
    """Runs SAM3 on `rgb`, in memory. Each wire in `wires` is (name, prompts, color) where
    `prompts` is a list of text-prompt phrasings tried in order, stopping at the first one SAM3
    detects anything for -- a twisted bundle doesn't always match "twisted wire" (lighting/angle
    can hide the twist), so WIRES_DEFAULT_PERPENDICULAR carries a few synonym phrasings as free
    retries against the same cached image encoding.

    If `min_confidence` is set (and lower than the processor's own threshold), each prompt that
    comes up empty at the normal confidence is retried once more at `min_confidence` before
    moving to the next phrasing -- same cheap fallback as segment_frame_coaxial_fallback().

    No point-cloud cleanup (filter_wire_points) -- matches wire_grasp_pipeline_orbbec.py, which
    never applied it to the top-down path.

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

        points_mm = xyz[sel] * 1000.0  # metres -> mm
        colors = rgb[sel].astype(np.float64) / 255.0
        suffix = f" (prompt: '{matched_prompt}')" if matched_prompt != prompts[0] else ""
        logger.info(f'  {name:15s} {n_pts:7,} pts{suffix}')
        results[name] = (mask, points_mm, colors, matched_prompt)

    return results


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
    """Depth-gap filter + DBSCAN cleanup for the coaxial path (bundle_points_from_rotation and
    segment_frame_coaxial_fallback) -- see SamRobotUntwisting.py's SAMSegmenter.filter_wire_points."""
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


def segment_frame_coaxial_fallback(processor, rgb, xyz, valid, wires, min_confidence=None, light_norm=True):
    """Per-colour SAM3 pass used only when rotation counting fails outright (see
    bundle_points_from_rotation / the coaxial cycle below) -- ported from
    wire_grasp_pipeline_orbbec_untwist.py's segment_frame(). Single prompt per wire (no synonym
    list, unlike segment_frame_perpendicular), applies filter_wire_points cleanup, and can
    normalize lighting before SAM3 sees the frame (`rgb` itself stays untouched for the point-
    cloud colours).

    Returns {wire_name: (mask_bool HxW, points_mm Nx3, colors_0to1 Nx3)}."""
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

        points_mm = xyz[sel] * 1000.0
        colors = rgb[sel].astype(np.float64) / 255.0
        logger.info(f'  {name:15s} {n_pts:7,} pts')
        points_mm, colors = filter_wire_points(points_mm, colors, label=name)
        results[name] = (mask, points_mm, colors)

    return results


def bundle_points_from_rotation(rotation_out, rgb, xyz, valid):
    """Back-projects rotation_counter.analyze()'s own bundle mask to camera-frame 3-D points --
    no extra SAM3 call, analyze() already built this mask while counting rotations. This is the
    grasp geometry for the whole twisted bundle in coaxial mode -- see
    wire_grasp_pipeline_orbbec_untwist.py's module docstring for why the bundle, not a single
    strand. Returns (mask_bool HxW, points_mm Nx3, colors_0to1 Nx3), or None if it back-projects
    to 0 valid points."""
    mask = rotation_out['wire_mask']
    H, W = rgb.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    sel = mask & valid
    if not sel.any():
        return None
    points_mm = xyz[sel] * 1000.0
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


# ── Perpendicular (top-down) grasp geometry, ported unchanged from wire_grasp_pose_orbbec.py ──

def get_local_pca_direction(centre_point, points_mm, radius_mm=LOCAL_PCA_RADIUS_MM_PERPENDICULAR):
    """UNSIGNED local wire direction -- fine for a top-down grasp, which lands the jaws the same
    way regardless of sign. See get_local_pca_direction_coaxial for the signed version the
    coaxial path needs instead."""
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


def find_grasp_pose(points_mm, down_cam, radius_mm=LOCAL_PCA_RADIUS_MM_PERPENDICULAR):
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


def camera_down_vector(camera_extrinsic_json):
    with open(camera_extrinsic_json) as f:
        extr = json.load(f)
    R_base_to_optical = _rpy_deg_to_matrix(*extr['rpy_deg'])
    down_cam = R_base_to_optical.T @ np.array([0.0, 0.0, -1.0])
    return down_cam / np.linalg.norm(down_cam)


# ── Coaxial (in-line) grasp geometry, ported unchanged from wire_grasp_pipeline_orbbec_untwist.py
# (itself ported from SamRobotUntwisting.py -- see that file's module docstring). ──

def get_local_pca_direction_coaxial(centre_point, points_mm, radius_mm=LOCAL_PCA_RADIUS_MM_COAXIAL):
    """Local wire direction at centre_point, SIGNED OUTWARD -- away from the wire body and
    toward centre_point -- so a coaxial approach (z = -direction, see
    calculate_coaxial_orientation) always comes from beyond the free end, never from beyond the
    anchor. Distinct from get_local_pca_direction (perpendicular path), which is unsigned and
    uses a smaller default radius."""
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
    kept because it falls out of the same fit for free). Picks whichever endpoint sits FARTHER
    FROM THE ROBOT BASE ORIGIN as the tip -- see wire_grasp_pipeline_orbbec_untwist.py's
    docstring for the full derivation."""
    if points_mm.shape[0] < 20:
        return None, None
    centroid = points_mm.mean(axis=0)
    _, _, vt = np.linalg.svd(points_mm - centroid)
    direction = vt[0]
    t = (points_mm - centroid) @ direction
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
    """COAXIAL grasp frame: tool z runs ALONG the wire, travelling from beyond the tip INTO it.
    wire_dir is already signed outward by get_local_pca_direction_coaxial, so the
    approach/insertion direction is its negation. x is seeded from `up` (world "up" in
    camera-optical coordinates) since this process never talks to the robot for a live EE
    orientation -- any perpendicular is valid, the grasp is rotationally symmetric about z."""
    pca_unit = wire_dir / np.linalg.norm(wire_dir)
    z_axis = -pca_unit

    x_axis = up - np.dot(up, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross([1.0, 0.0, 0.0], z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)   # right-handed: z cross x = y
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def show_debug_pointcloud(points_mm, colors=None, tip=None, anchor=None, local_pts=None,
                           T_grasp=None, title="segmented point cloud"):
    """Blocking Open3D window of the cloud find_coaxial_grasp_pose() is about to trust -- see
    wire_grasp_pipeline_orbbec_untwist.py's docstring for why this runs in-process (unlike the
    mask-overlay window, which needs the spawned viewer process)."""
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


def find_coaxial_grasp_pose(points_mm, up_cam, T_optical_to_base, radius_mm=LOCAL_PCA_RADIUS_MM_COAXIAL,
                             debug=False, debug_colors=None, debug_title="segmented point cloud"):
    """tip/anchor -> local PCA direction at the tip -> coaxial frame, all in camera-optical
    frame (mm). Returns (T_grasp_cam_mm, local_pts, tip, anchor) or None."""
    if len(points_mm) == 0:
        return None
    tip, anchor = wire_tip_and_anchor_cam(points_mm, T_optical_to_base)
    if tip is None:
        if debug:
            show_debug_pointcloud(points_mm, debug_colors,
                                  title=f"{debug_title} (too few points for a global PCA fit)")
        return None
    local_pts, wire_dir = get_local_pca_direction_coaxial(tip, points_mm, radius_mm)
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


def camera_up_vector(camera_extrinsic_json):
    """World 'up' expressed in camera-optical coordinates -- the x-axis seed for
    calculate_coaxial_orientation. Same extrinsic camera_down_vector used; this is just its
    negation, kept as its own function so the sign is named rather than inlined."""
    with open(camera_extrinsic_json) as f:
        extr = json.load(f)
    R_base_to_optical = _rpy_deg_to_matrix(*extr['rpy_deg'])
    up_cam = R_base_to_optical.T @ np.array([0.0, 0.0, 1.0])
    return up_cam / np.linalg.norm(up_cam)


# ── Shared geometry/publish helpers ─────────────────────────────────────────

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
    """Publish the grasp pose as a persistent ROS2 static TF, killing any TF publisher this
    script started previously for the same grasp_frame first -- see
    wire_grasp_pipeline_orbbec_untwist.py's identical docstring for why prev_proc.terminate()
    alone is not enough (pkill-by-frame-name reaches the reparented static_transform_publisher
    child too)."""
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
    """Shared payload schema for BOTH modes -- perpendicular-mode callers simply never pass
    untwist_deg, leaving it at 0.0 (grasp_executor.py-side consumers of a plain grasp already
    ignore this field; grasp_executor_untwist.py-side consumers use it to spin the wrist)."""
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
    """Send every wire's grasp from this capture as ONE message, {"frame_idx": ..., "grasps":
    [payload, ...]} -- not one send per wire, so the SUB side's CONFLATE=1 can never drop one
    colour out of a capture while keeping another. See wire_grasp_pipeline_orbbec.py's identical
    docstring."""
    if not payloads:
        return
    pub_socket.send(msgpack.packb({"frame_idx": frame_idx, "grasps": payloads}))
    names = ", ".join(p["prompt"] for p in payloads)
    logger.info(f"[zmq pub] Published {len(payloads)} grasp(s) (base frame): {names} frame_idx={frame_idx}")


# ── Per-mode capture->detect->geometry cycles, each with its own bounded --detect_retries loop.
# Both return `payloads` (a non-empty list) on success, or None after exhausting every attempt --
# never partially built, never published more than once. ──

def run_perpendicular_cycle(cap, args, bbox, get_intrinsics, processor, down_cam, wires,
                             wire_filter, viz_queue, frame_idx):
    wires_this_frame = [w for w in wires if wire_filter is None or w[0] == wire_filter]

    for attempt in range(args.detect_retries + 1):
        if attempt > 0:
            print(f"  retrying with a fresh capture (attempt {attempt + 1}/{args.detect_retries + 1})…")
        # Reconnect from scratch every attempt, not just the cycle's first one -- same
        # rationale as run_coaxial_cycle's identical call: grab_rgb()'s freshness/stability
        # checks alone weren't a strong enough guarantee against a wrong-depth-under-a-correct-
        # mask result, and a retry deserves the same guarantee the first attempt gets, not a
        # weaker one from reusing a connection that's no longer brand-new.
        cap.reconnect()
        img_rgb, img_depth_mm = cap.grab_rgb(timeout=args.wait_timeout)
        if img_depth_mm is None:
            print("  no depth frame available yet, skipping this attempt")
            continue
        if img_depth_mm.shape[:2] != img_rgb.shape[:2]:
            # Color/depth can arrive at different native resolutions -- resize depth onto rgb's
            # own pixel grid BEFORE cropping with a bbox computed in rgb's coordinate space, or
            # the crop (and everything downstream: xyz, the SAM3 mask lining up with depth) reads
            # the wrong region. NEAREST: it's a range map, interpolating invents wrong depths at
            # edges. Both source scripts had this; it was dropped when they were merged here.
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
        xyz = bbox_utils.depth_to_xyz(depth_mm, fx, fy, cx - bx, cy - by)
        valid = np.isfinite(xyz).all(axis=-1)

        segmented = segment_frame_perpendicular(processor, rgb, xyz, valid, wires_this_frame,
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

        ranked = sorted(segmented.items(), key=lambda kv: -len(kv[1][1]))
        if len(ranked) > 1:
            print(f"  candidates by points: {', '.join(f'{n}={len(p):,}' for n, (_, p, _, _) in ranked)}")

        attempt_payloads = []
        for wire_name, (_mask, points_mm, _colors, _matched_prompt) in ranked:
            result = find_grasp_pose(points_mm, down_cam)
            if result is None:
                print(f"  {wire_name}: not enough points near its centre for a grasp pose, skipping")
                continue
            T_grasp_optical_mm, _local_pts = result
            print(f"  {wire_name} ({len(points_mm):,} pts) grasp (camera-optical frame, mm): "
                  f"{np.round(T_grasp_optical_mm[:3, 3], 1).tolist()}")

            if not args.no_publish_tf:
                grasp_frame = f"{args.grasp_frame}_{wire_name}"
                _CTX['tf_procs'][wire_name] = publish_grasp_tf(T_grasp_optical_mm, args.camera_frame, grasp_frame,
                                                                prev_proc=_CTX['tf_procs'].get(wire_name))

            T_grasp_optical_m = T_grasp_optical_mm.copy()
            T_grasp_optical_m[:3, 3] /= 1000.0
            T_grasp_base = _CTX['T_optical_to_base'] @ T_grasp_optical_m
            print(f"  {wire_name} grasp (robot base frame, m): {np.round(T_grasp_base[:3, 3], 4).tolist()}")
            attempt_payloads.append(build_grasp_payload(T_grasp_base, wire_name, frame_idx))

        if not attempt_payloads:
            print("  no candidate wire had enough points near its centre for a grasp pose")
            continue

        return attempt_payloads

    return None


def run_coaxial_cycle(cap, args, bbox, get_intrinsics, processor, rotation_counter, up_cam,
                      wires, wire_filter, viz_queue, frame_idx):
    wires_this_frame = [w for w in wires if wire_filter is None or w[0] == wire_filter]
    rotation_colors = [name.split('_')[0] for name, _prompt, _clr in wires_this_frame]

    for attempt in range(args.detect_retries + 1):
        if attempt > 0:
            print(f"  retrying with a fresh capture (attempt {attempt + 1}/{args.detect_retries + 1})…")
        # Reconnect from scratch every attempt, not just the cycle's first one: grab_rgb()'s
        # freshness/stability checks operate on frames from the SAME long-lived subscriber
        # threads, which is enough in isolation but a wrong-depth-under-a-right-looking-mask
        # result kept recurring here specifically (never in a short-lived, freshly-connected
        # test) even with those checks passing. Tearing down and restarting the ZMQ subscription
        # entirely removes any possible carry-over from the connection's own history, not just
        # from the cached frame -- and a retry deserves that same guarantee, not a weaker one
        # from reusing a connection that's no longer brand-new.
        cap.reconnect()
        img_rgb, img_depth_mm = cap.grab_rgb(timeout=args.wait_timeout)
        if img_depth_mm is None:
            print("  no depth frame available yet, skipping this attempt")
            continue
        if img_depth_mm.shape[:2] != img_rgb.shape[:2]:
            # Color/depth can arrive at different native resolutions -- resize depth onto rgb's
            # own pixel grid BEFORE cropping with a bbox computed in rgb's coordinate space, or
            # the crop (and everything downstream: xyz, the SAM3 mask lining up with depth) reads
            # the wrong region. NEAREST: it's a range map, interpolating invents wrong depths at
            # edges. Both source scripts had this; it was dropped when they were merged here.
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
        xyz = bbox_utils.depth_to_xyz(depth_mm, fx, fy, cx - bx, cy - by)
        valid = np.isfinite(xyz).all(axis=-1)

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
                color = wire_filter.split('_')[0]
                strand_px = int(rotation_out['strand_masks'][color].sum())
                if strand_px == 0:
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
            segmented = segment_frame_coaxial_fallback(processor, rgb, xyz, valid, wires_this_frame,
                                                        min_confidence=args.min_confidence,
                                                        light_norm=not args.no_light_norm)
            if not args.no_viz:
                for name, prompt, _color in wires_this_frame:
                    if name in segmented:
                        seg_mask, seg_points_mm, _ = segmented[name]
                        show_mask_overlay(viz_queue, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), seg_mask,
                                          prompt, int(seg_mask.sum()))
            if not segmented:
                print("  no wire detected in this frame")
                continue
            ranked = sorted(segmented.items(), key=lambda kv: -len(kv[1][1]))
            if len(ranked) > 1:
                print(f"  candidates by points: {', '.join(f'{n}={len(p):,}' for n, (_, p, _) in ranked)} "
                      f"-- using only the largest (this stage grasps ONE wire/bundle per cycle)")
            # Only the single best (most points) candidate -- this fallback used to publish one
            # grasp per detected colour, which made sense for an older colour-sorting workflow but
            # not here: the coaxial stage always presents/untwists exactly one wire per cycle.
            candidates = [(name, points_mm, 0.0, colors_arr)
                         for name, (_mask, points_mm, colors_arr) in ranked[:1]]

        attempt_payloads = []
        for wire_name, points_mm, rotations, colors_arr in candidates:
            result = find_coaxial_grasp_pose(points_mm, up_cam, _CTX['T_optical_to_base'],
                                             debug=args.debug_pcd, debug_colors=colors_arr,
                                             debug_title=f"{wire_name} ({len(points_mm):,} pts)")
            if result is None:
                print(f"  {wire_name}: not enough points near a wire end for a coaxial grasp pose, skipping")
                continue
            T_grasp_optical_mm, _local_pts, _tip, _anchor = result
            untwist_deg = round(rotations) * 180.0

            print(f"  {wire_name} ({len(points_mm):,} pts) coaxial grasp (camera-optical frame, mm): "
                  f"{np.round(T_grasp_optical_mm[:3, 3], 1).tolist()}"
                  + (f", counted {rotations:.1f} rotations -> untwist {untwist_deg:.0f} deg"
                     if untwist_deg else ""))

            if not args.no_publish_tf:
                grasp_frame = f"{args.grasp_frame}_{wire_name}"
                _CTX['tf_procs'][wire_name] = publish_grasp_tf(T_grasp_optical_mm, args.camera_frame, grasp_frame,
                                                                prev_proc=_CTX['tf_procs'].get(wire_name))

            T_grasp_optical_m = T_grasp_optical_mm.copy()
            T_grasp_optical_m[:3, 3] /= 1000.0
            T_grasp_base = _CTX['T_optical_to_base'] @ T_grasp_optical_m
            print(f"  {wire_name} grasp (robot base frame, m): {np.round(T_grasp_base[:3, 3], 4).tolist()}")
            attempt_payloads.append(build_grasp_payload(T_grasp_base, wire_name, frame_idx, untwist_deg))

        if not attempt_payloads:
            print("  no candidate wire had enough points near an end for a coaxial grasp pose")
            continue

        return attempt_payloads

    return None


# Small shared-state bag for the two cycle functions above (tf_procs dict, T_optical_to_base) --
# avoids threading two more parameters through both call sites just for TF-process bookkeeping
# that's naturally global-for-the-process-lifetime state, same as each source script's tf_procs.
_CTX = {'tf_procs': {}, 'T_optical_to_base': None}


# ── Main loop ────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Startup sweep: kill any static_transform_publisher left over from a previous run of this
    # script OR of wire_grasp_pipeline_orbbec.py/wire_grasp_pipeline_orbbec_untwist.py -- their
    # own per-frame pkill in publish_grasp_tf() only matches when a NEW pose reuses the exact
    # same wire_name (so, e.g., a leftover "wire_grasp_wire" from an old run won't get cleaned
    # up if this run instead detects "wire_grasp_colorful_wire" first), and an unclean exit
    # (kill -9, crash) skips each script's own end-of-run cleanup entirely. A stale broadcaster
    # under a frame name we never touch this run would otherwise sit in RViz indefinitely,
    # looking exactly like "the demo pipeline republished the old pose". Matches by prefix
    # (args.grasp_frame + "_"), not exact name, so it catches every wire_name, not just today's.
    if not args.no_publish_tf:
        killed = subprocess.run(
            ['pkill', '-f', f'static_transform_publisher.*--child-frame-id {args.grasp_frame}_'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if killed.returncode == 0:
            logger.info(f"[startup] cleared stale static_transform_publisher(s) matching "
                        f"'--child-frame-id {args.grasp_frame}_*' from a previous run")

    wires_perpendicular = parse_colors_arg_perpendicular(args.colors)
    wires_coaxial = parse_colors_arg_coaxial(args.colors)
    wire_filter = args.wire

    bbox_file = args.bbox_file or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbox_perpendicular.json")
    bbox_file_coaxial = args.bbox_file_coaxial or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bbox_coaxial.json")
    display_max = tuple(args.display_max) if args.display_max else None

    logger.info(f"Connecting to Orbbec streams at {args.jetson_ip}:{args.color_port} (colour) / "
                f":{args.depth_port} (depth) ...")
    cap = OrbbecCapture(f"{args.jetson_ip}:{args.color_port}", f"{args.jetson_ip}:{args.depth_port}")
    img_rgb, _ = cap.grab_rgb(timeout=args.wait_timeout)
    logger.info(f"Connected: first frame {img_rgb.shape[1]}x{img_rgb.shape[0]}")

    bbox_perpendicular = tuple(args.bbox) if args.bbox else (
        None if args.full else bbox_utils.load_bbox(str(bbox_file)))
    if bbox_perpendicular and not args.full:
        logger.info(f"Perpendicular crop box: x={bbox_perpendicular[0]} y={bbox_perpendicular[1]} "
                    f"w={bbox_perpendicular[2]} h={bbox_perpendicular[3]}"
                    f"{'' if args.bbox else f' (from {bbox_file})'}")

    bbox_coaxial = tuple(args.bbox_coaxial) if args.bbox_coaxial else (
        None if args.full else bbox_utils.load_bbox(str(bbox_file_coaxial)))
    if bbox_coaxial and not args.full:
        logger.info(f"Coaxial crop box: x={bbox_coaxial[0]} y={bbox_coaxial[1]} "
                    f"w={bbox_coaxial[2]} h={bbox_coaxial[3]}"
                    f"{'' if args.bbox_coaxial else f' (from {bbox_file_coaxial})'}")

    intrinsics = {}

    def get_intrinsics():
        if not intrinsics:
            fx, fy, cx, cy = bbox_utils.load_intrinsics(args.intrinsics_json, camera_name=args.camera_name)
            intrinsics.update(fx=fx, fy=fy, cx=cx, cy=cy)
        return intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

    down_cam = camera_down_vector(args.camera_extrinsic_json)
    up_cam = camera_up_vector(args.camera_extrinsic_json)
    logger.info(f"Gravity 'down' in camera-optical frame: [{down_cam[0]:+.3f}, {down_cam[1]:+.3f}, {down_cam[2]:+.3f}]")
    _CTX['T_optical_to_base'] = camera_to_base_transform(args.camera_extrinsic_json)

    zmq_ctx = zmq.Context()

    grasp_pub = None
    if not args.no_publish:
        grasp_pub = zmq_ctx.socket(zmq.PUB)
        grasp_pub.bind(args.zmq_pub_addr)
        logger.info(f"Grasp publisher bound to {args.zmq_pub_addr}; waiting {ZMQ_SLOW_JOINER_DELAY_S}s "
                    f"for subscribers to connect...")
        time.sleep(ZMQ_SLOW_JOINER_DELAY_S)

    sig_sub = zmq_ctx.socket(zmq.SUB)
    sig_sub.setsockopt(zmq.CONFLATE, 1)
    sig_sub.connect(args.trigger_addr)
    sig_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    logger.info(f"Waiting for {{'mode': ...}} triggers on {args.trigger_addr} from "
                f"demo_pipeline_executor.py (Ctrl-C to quit)")

    # Interactive bbox picking (--select_bbox/--auto_bbox at startup only -- no per-cycle
    # keypress loop in this demo-only script) needs its own process, see module docstring.
    mp_ctx = mp.get_context('spawn')
    viz_queue, viz_result_queue = mp_ctx.Queue(), mp_ctx.Queue()
    viz_proc = mp_ctx.Process(target=_viewer_process, args=(viz_queue, viz_result_queue, MASK_WINDOW_NAME), daemon=True)
    viz_proc.start()

    def pick_bbox(which, mode, preview_rgb, preview_depth_mm):
        """which: 'perpendicular' or 'coaxial' -- picks/saves into that stage's own bbox
        variable+file, independent of the other stage's."""
        nonlocal bbox_perpendicular, bbox_coaxial
        target_file = bbox_file if which == "perpendicular" else bbox_file_coaxial
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
        if which == "perpendicular":
            bbox_perpendicular = picked
        else:
            bbox_coaxial = picked
        bbox_utils.save_bbox(str(target_file), picked, preview_rgb.shape)
        print(f"  {which} crop box saved to {target_file}")

    if args.select_bbox or args.auto_bbox:
        print("Capturing a frame to set the perpendicular-stage crop box from (wire on the table) ...")
        preview_rgb, preview_depth = cap.grab_rgb(timeout=args.wait_timeout)
        pick_bbox("perpendicular", "auto" if args.auto_bbox else "manual", preview_rgb, preview_depth)

    if args.select_bbox_coaxial:
        print("Capturing a frame to set the coaxial-stage crop box from -- make sure the right arm "
              "is currently holding a wire at --right_park_deg before confirming this one ...")
        preview_rgb, preview_depth = cap.grab_rgb(timeout=args.wait_timeout)
        pick_bbox("coaxial", "manual", preview_rgb, preview_depth)

    # SAM3 loads exactly ONCE here, regardless of how many perpendicular/coaxial cycles follow --
    # the entire point of merging the two source scripts into one process.
    processor = load_sam3(args.checkpoint, args.confidence)
    rotation_counter = WireRotationCounter(processor=processor)
    frame_idx = 0

    print("\nWaiting for triggers (Ctrl-C to quit)...")

    while True:
        msg = sig_sub.recv()  # blocks until demo_pipeline_executor.py sends {'mode': ...}
        try:
            trigger = msgpack.unpackb(msg)
        except Exception as e:
            logger.warning(f"unparseable trigger message ({e}), ignoring")
            continue
        mode = trigger.get('mode') if isinstance(trigger, dict) else None

        print(f"=== frame {frame_idx} (mode={mode}) ===")
        if mode == 'perpendicular':
            payloads = run_perpendicular_cycle(cap, args, bbox_perpendicular, get_intrinsics, processor, down_cam,
                                               wires_perpendicular, wire_filter, viz_queue, frame_idx)
        elif mode == 'coaxial':
            payloads = run_coaxial_cycle(cap, args, bbox_coaxial, get_intrinsics, processor, rotation_counter,
                                         up_cam, wires_coaxial, wire_filter, viz_queue, frame_idx)
        else:
            logger.warning(f"unrecognised trigger mode {mode!r} (want 'perpendicular' or 'coaxial'), "
                           f"ignoring and waiting for the next trigger")
            continue

        if payloads is None:
            print(f"  giving up on this cycle after {args.detect_retries + 1} attempt(s) -- nothing published")
        elif grasp_pub is not None:
            publish_grasps(grasp_pub, payloads, frame_idx)

        frame_idx += 1


if __name__ == "__main__":
    main()
    # Skip normal interpreter teardown -- see wire_grasp_pipeline_orbbec.py's identical comment
    # for the SIGABRT this avoids (SAM3/CUDA + zmq threads + a spawned cv2 window all alive at
    # once during shutdown).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
