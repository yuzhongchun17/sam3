"""
stream_segment.py
-----------------
Headless SAM3 text-prompt segmenter for the DEX robot wrist camera.
Subscribes to the colour + depth ZMQ streams published by pub_orbbec on
the Jetson, and on demand runs SAM3 segmentation using a text prompt.
The segmented RGBD frame is then published over ZMQ for
anygrasp_sam3_stream.py to consume.

Prerequisites
-------------
- pub_orbbec must be running on the Jetson (provides colour on port 10031
  and depth on port 10033 by default).
- conda environment 'sam3' activated (Python 3.12, PyTorch 2.7, CUDA 12.6).
- SAM3 installed in editable mode (`pip install -e .` from the sam3 root).

Configuration
-------------
Edit the three constants at the bottom of this file before running:

    JETSON_IP  = "192.168.11.9"   # IP of the Jetson on your network
    COLOR_PORT = "10031"          # ZMQ port for colour stream (pub_orbbec)
    DEPTH_PORT = "10033"          # ZMQ port for depth stream  (pub_orbbec)

Usage
-----
    conda activate sam3
    python code/stream_segment.py

    # Use a local checkpoint to skip HuggingFace download:
    python code/stream_segment.py --checkpoint /path/to/sam3.pt

Stdin commands
--------------
  s (or Enter)  Freeze the current frame and enter the SAM3 prompt loop:
                  [SAM3] Enter text prompt (blank to cancel):
                Type a plain-English description of the object to segment,
                e.g.  "cable"  or  "orange wire"  or  "green connector".
                Re-prompts automatically if nothing is detected.
                On success the segmented RGBD is published.
  q             Quit

Output
------
Publishes segmented RGBD on:  tcp://127.0.0.1:5560

Payload format (msgpack dict, consumed by anygrasp_sam3_stream.py):
  color_img   : JPEG bytes  — BGR colour, pixels outside mask are black
  depth_raw   : raw bytes   — uint16 depth, tobytes(); zero outside mask
  depth_shape : [H, W]      — reshape depth_raw with this before use
  prompt      : str         — the text prompt that produced this mask
  timestamp   : float       — time.time()
"""

import os
import argparse
import cv2
import zmq
import msgpack
import numpy as np
import time
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# torch / sam3 are imported lazily inside _load_sam3() to avoid an OpenMP
# shared-library conflict with cv2 that causes a segfault at startup.


class DualStreamViewer:
    def __init__(self, color_ip_port: str, depth_ip_port: str, checkpoint: str = None):
        self.color_ip_port = color_ip_port
        self.depth_ip_port = depth_ip_port
        self._checkpoint_path = checkpoint

        self.context = zmq.Context()

        self.rgb_frame = None
        self.depth_frame = None
        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()

        self._sam3_model = None
        self._sam3_processor = None
        self._PIL_Image = None

        self._seg_pub = self.context.socket(zmq.PUB)
        self._seg_pub.bind(f"tcp://127.0.0.1:{SEG_PUB_PORT}")
        logger.info(f"Segmentation publisher bound to tcp://127.0.0.1:{SEG_PUB_PORT}")

    def _decode_payload(self, packed_message, is_depth=False):
        """Robust decoder that tries msgpack first, then raw bytes."""
        try:
            message = msgpack.unpackb(packed_message)
            if is_depth and 'depth_png' in message:
                arr = np.frombuffer(message['depth_png'], dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            else:
                for key in ('color_img', 'color_png', 'color_jpg'):
                    if key in message:
                        arr = np.frombuffer(message[key], dtype=np.uint8)
                        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
                raise KeyError(f"No colour key found in msgpack message: {list(message.keys())}")
        except Exception:
            arr = np.frombuffer(packed_message, dtype=np.uint8)
            flags = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
            return cv2.imdecode(arr, flags)

    def _color_subscriber_thread(self):
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(f"tcp://{self.color_ip_port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        logger.info(f"Connected to Color stream at {self.color_ip_port}")
        while True:
            try:
                msg = sub.recv(zmq.NOBLOCK)
                frame = self._decode_payload(msg, is_depth=False)
                if frame is not None:
                    with self.rgb_lock:
                        self.rgb_frame = frame
            except zmq.error.Again:
                time.sleep(0.005)

    def _depth_subscriber_thread(self):
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.connect(f"tcp://{self.depth_ip_port}")
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        logger.info(f"Connected to Depth stream at {self.depth_ip_port}")
        while True:
            try:
                msg = sub.recv(zmq.NOBLOCK)
                frame = self._decode_payload(msg, is_depth=True)
                if frame is not None:
                    with self.depth_lock:
                        self.depth_frame = frame
            except zmq.error.Again:
                time.sleep(0.005)

    # ------------------------------------------------------------------
    # SAM3 helpers
    # ------------------------------------------------------------------

    def _load_sam3(self):
        """Load torch + SAM3. Deferred import avoids cv2/OpenMP conflict at module load."""
        if self._sam3_model is not None:
            return

        import torch
        import sam3
        from PIL import Image as _Image
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._PIL_Image = _Image

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        if self._checkpoint_path is not None:
            logger.info(f"Loading SAM3 from local checkpoint: {self._checkpoint_path}")
            logger.info("Please wait — this may take 30–60 s on first load...")
            self._sam3_model = build_sam3_image_model(
                bpe_path=bpe_path,
                checkpoint_path=self._checkpoint_path,
                load_from_HF=False,
            )
        else:
            logger.info("Loading SAM3 from HuggingFace cache (facebook/sam3)...")
            logger.info("Please wait — this may take 30–60 s on first load...")
            self._sam3_model = build_sam3_image_model(bpe_path=bpe_path, load_from_HF=True)

        self._sam3_processor = Sam3Processor(self._sam3_model, confidence_threshold=0.35)
        logger.info("SAM3 model ready — streams starting.")

    def _run_segmentation(self, color_bgr, depth_u16):
        """Prompt loop: re-prompts until an object is found or the user cancels."""
        pil_image = self._PIL_Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        inference_state = self._sam3_processor.set_image(pil_image)

        while True:
            print("\n[SAM3] Enter text prompt (blank to cancel): ", end="", flush=True)
            prompt = input().strip()
            if not prompt:
                logger.info("Segmentation cancelled.")
                return

            logger.info(f"Running SAM3 with prompt: '{prompt}' ...")
            self._sam3_processor.reset_all_prompts(inference_state)
            inference_state = self._sam3_processor.set_text_prompt(
                state=inference_state, prompt=prompt
            )

            raw_masks = inference_state.get("masks")
            if raw_masks is None or len(raw_masks) == 0:
                print(f"[SAM3] No object detected for '{prompt}'. Try a different prompt.")
                continue

            masks_np = raw_masks.detach().cpu().numpy()
            if masks_np.ndim == 4:
                masks_np = masks_np.squeeze(1)
            combined_mask = np.max(masks_np, axis=0)
            mask_bool = combined_mask > 0
            n_pixels = int(mask_bool.sum())
            logger.info(f"Detected {masks_np.shape[0]} object(s), {n_pixels} px masked.")

            seg_rgb = np.zeros_like(color_bgr)
            seg_rgb[mask_bool] = color_bgr[mask_bool]

            seg_depth = None
            if depth_u16 is not None:
                dh, dw = depth_u16.shape[:2]
                mh, mw = mask_bool.shape
                if (mh, mw) != (dh, dw):
                    mask_uint8 = (combined_mask * 255).astype(np.uint8)
                    mask_depth = cv2.resize(mask_uint8, (dw, dh),
                                            interpolation=cv2.INTER_NEAREST) > 0
                else:
                    mask_depth = mask_bool
                seg_depth = np.zeros_like(depth_u16)
                seg_depth[mask_depth] = depth_u16[mask_depth]

            self._publish_segmented_rgbd(seg_rgb, seg_depth, prompt)
            print("[SAM3] Published. Press 's' + Enter to segment a new frame.")
            return

    def _publish_segmented_rgbd(self, seg_rgb, seg_depth, prompt):
        ok, color_buf = cv2.imencode('.jpg', seg_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            logger.error("Failed to JPEG-encode segmented colour — not publishing.")
            return

        if seg_depth is not None:
            depth_raw = seg_depth.astype(np.uint16).tobytes()
            depth_shape = list(seg_depth.shape[:2])
        else:
            depth_raw = b''
            depth_shape = [0, 0]

        payload = msgpack.packb({
            'color_img':   color_buf.tobytes(),
            'depth_raw':   depth_raw,
            'depth_shape': depth_shape,
            'prompt':      prompt,
            'timestamp':   time.time(),
        })

        self._seg_pub.send(payload)
        logger.info(f"Published segmented RGBD ({len(payload)//1024} KB) on port {SEG_PUB_PORT}")

    def _headless_loop(self):
        logger.info("Ready. Commands: 's' + Enter to segment, 'q' + Enter to quit.")
        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break

            if cmd == 'q':
                logger.info("Exiting...")
                break
            elif cmd == 's' or cmd == '':
                with self.rgb_lock:
                    snap_color = self.rgb_frame.copy() if self.rgb_frame is not None else None
                with self.depth_lock:
                    snap_depth = self.depth_frame.copy() if self.depth_frame is not None else None

                if snap_color is None:
                    logger.warning("No colour frame available yet — waiting for stream.")
                else:
                    logger.info("[FRAME CAPTURED] Launching SAM3 segmentation...")
                    self._run_segmentation(snap_color, snap_depth)
            else:
                print("Unknown command. Use 's' to segment or 'q' to quit.")

    def run(self):
        self._load_sam3()

        t_color = threading.Thread(target=self._color_subscriber_thread, daemon=True)
        t_depth = threading.Thread(target=self._depth_subscriber_thread, daemon=True)

        t_color.start()
        t_depth.start()

        self._headless_loop()


SEG_PUB_PORT = 5560  # local port — anygrasp subscribes to tcp://127.0.0.1:5560

if __name__ == "__main__":
    JETSON_IP  = "192.168.11.9"
    COLOR_PORT = "10031"
    DEPTH_PORT = "10033"

    parser = argparse.ArgumentParser(description="SAM3 streaming segmenter with ZMQ publisher")
    parser.add_argument("--checkpoint",
                        default=os.path.expanduser(
                            "~/.cache/huggingface/hub/models--facebook--sam3/"
                            "snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
                        ),
                        help="Path to local SAM3 checkpoint .pt file.")
    args = parser.parse_args()

    viewer = DualStreamViewer(
        color_ip_port=f"{JETSON_IP}:{COLOR_PORT}",
        depth_ip_port=f"{JETSON_IP}:{DEPTH_PORT}",
        checkpoint=args.checkpoint,
    )
    viewer.run()
