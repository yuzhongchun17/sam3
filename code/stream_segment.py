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

# --- 1. Standalone FPS Tracker (No Richtech imports needed) ---
class SimpleFPS:
    def __init__(self):
        self.prev_time = time.perf_counter()
        self.fps = 0.0

    def update(self):
        curr_time = time.perf_counter()
        diff = curr_time - self.prev_time
        if diff > 0:
            self.fps = (self.fps * 0.9) + ((1.0 / diff) * 0.1)
        self.prev_time = curr_time
        return self.fps

# --- 2. The Main Viewer Class ---
class DualStreamViewer:
    def __init__(self, color_ip_port: str, depth_ip_port: str, visualize: bool = True):
        self.color_ip_port = color_ip_port
        self.depth_ip_port = depth_ip_port
        self.visualize = visualize

        self.context = zmq.Context()
        
        # Data storage and locks for thread safety
        self.rgb_frame = None
        self.depth_frame = None
        self.rgb_lock = threading.Lock()
        self.depth_lock = threading.Lock()
        
        self.fps_color = SimpleFPS()
        self.fps_depth = SimpleFPS()
        self.fps_show = SimpleFPS()

        # Start in color mode
        self.display_mode = 'color'

        # SAM3 — lazy-loaded on first 's' press
        self._sam3_model = None
        self._sam3_processor = None
        self._PIL_Image = None

        # PUB socket — publishes segmented RGBD for anygrasp (or any other subscriber)
        self._seg_pub = self.context.socket(zmq.PUB)
        self._seg_pub.bind(f"tcp://127.0.0.1:{SEG_PUB_PORT}")
        logger.info(f"Segmentation publisher bound to tcp://127.0.0.1:{SEG_PUB_PORT}")

    def _decode_payload(self, packed_message, is_depth=False):
        """Robust decoder that tries msgpack first, then raw bytes."""
        try:
            # 1. Try msgpack (if Dex packs it as {'color_png': bytes} or {'depth_png': bytes})
            message = msgpack.unpackb(packed_message)
            if is_depth and 'depth_png' in message:
                arr = np.frombuffer(message['depth_png'], dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            else:
                # pub_orbbec sends 'color_img'; fallback to 'color_png' / 'color_jpg'
                for key in ('color_img', 'color_png', 'color_jpg'):
                    if key in message:
                        arr = np.frombuffer(message[key], dtype=np.uint8)
                        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
                raise KeyError(f"No colour key found in msgpack message: {list(message.keys())}")
        except Exception:
            # 2. Fallback: Raw bytes (if Dex just sends raw jpeg/png bytes directly)
            arr = np.frombuffer(packed_message, dtype=np.uint8)
            flags = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
            return cv2.imdecode(arr, flags)

    def _color_subscriber_thread(self):
        """Background thread listening ONLY to the color port."""
        sub = self.context.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1) # Keep only newest frame
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
                    self.fps_color.update()
            except zmq.error.Again:
                time.sleep(0.005)

    def _depth_subscriber_thread(self):
        """Background thread listening ONLY to the depth port."""
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
                    self.fps_depth.update()
            except zmq.error.Again:
                time.sleep(0.005)

    def _display_thread(self):
        """Main thread for rendering the OpenCV window."""
        logger.info("Display ready. Controls: 'c' (Color), 'd' (Depth), 's' (Segment), 'q' (Quit)")
        
        while True:
            display_img = None
            info_text = ""
            
            # --- 1. Grab the active frame ---
            if self.display_mode == 'color':
                with self.rgb_lock:
                    if self.rgb_frame is not None:
                        display_img = self.rgb_frame.copy()
                        info_text = "MODE: COLOR"
            
            elif self.display_mode == 'depth':
                with self.depth_lock:
                    if self.depth_frame is not None:
                        # Normalize 16-bit depth to 8-bit for viewing and apply Jet colormap
                        norm_img = cv2.normalize(self.depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        display_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
                        info_text = "MODE: DEPTH"

            # --- 2. Render the frame ---
            if display_img is not None:
                self.fps_show.update()
                
                # Draw text overlay
                cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_img, f"Show: {self.fps_show.fps:.1f} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if self.display_mode == 'color':
                    cv2.putText(display_img, f"Recv: {self.fps_color.fps:.1f} FPS", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, f"Recv: {self.fps_depth.fps:.1f} FPS", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Dex Dual Stream Viewer", display_img)

            # --- 3. Handle Keyboard Controls ---
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('c'):
                self.display_mode = 'color'
                logger.info("Switched to COLOR view")
            
            elif key == ord('d'):
                self.display_mode = 'depth'
                logger.info("Switched to DEPTH view")
                
            elif key == ord('s'):
                # Grab both frames atomically for the segmentation call
                with self.rgb_lock:
                    snap_color = self.rgb_frame.copy() if self.rgb_frame is not None else None
                with self.depth_lock:
                    snap_depth = self.depth_frame.copy() if self.depth_frame is not None else None

                if snap_color is None:
                    logger.warning("No colour frame available yet — waiting for stream.")
                else:
                    logger.info("[FRAME CAPTURED] Launching SAM3 segmentation...")
                    self._run_segmentation(snap_color, snap_depth)
                    
            elif key == ord('q'):
                logger.info("Exiting...")
                break

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # SAM3 helpers
    # ------------------------------------------------------------------

    def _load_sam3(self):
        """Lazy-load torch + SAM3 on first use (deferred to avoid cv2/OpenMP conflict)."""
        if self._sam3_model is not None:
            return
        logger.info("Loading SAM3 model — this takes ~30 s the first time...")

        import torch
        import sam3
        from PIL import Image as _Image
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Store PIL.Image reference for use in _run_segmentation
        self._PIL_Image = _Image

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
        self._sam3_model = build_sam3_image_model(bpe_path=bpe_path)
        self._sam3_processor = Sam3Processor(self._sam3_model, confidence_threshold=0.35)
        logger.info("SAM3 model ready.")

    def _run_segmentation(self, color_bgr, depth_u16):
        """
        Prompt loop: keeps asking for a text prompt (without reloading the model)
        until an object is found or the user leaves the prompt blank to cancel.
        Displays segmented RGB and segmented depth (zeroed outside the mask).
        """
        self._load_sam3()

        # set_image once per captured frame — prompts are reset between retries
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
                continue  # re-prompt without reloading model

            # Collapse [N,1,H,W] or [N,H,W] → 2-D binary mask
            masks_np = raw_masks.detach().cpu().numpy()
            if masks_np.ndim == 4:
                masks_np = masks_np.squeeze(1)
            combined_mask = np.max(masks_np, axis=0)   # float32 [H, W], values in [0,1]
            mask_bool = combined_mask > 0
            n_pixels = int(mask_bool.sum())
            logger.info(f"Detected {masks_np.shape[0]} object(s), {n_pixels} px masked.")

            # --- Segmented RGB: colour only inside the mask, black outside ---
            seg_rgb = np.zeros_like(color_bgr)
            seg_rgb[mask_bool] = color_bgr[mask_bool]
            # cv2.putText(seg_rgb, f"Prompt: {prompt}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # cv2.putText(seg_rgb, f"Pixels: {n_pixels}", (10, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # cv2.imshow("SAM3 - Segmented RGB", seg_rgb)

            # --- Segmented Depth: depth only inside the mask, zero outside ---
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
                # # Visualise as Jet colourmap for display
                # norm = cv2.normalize(seg_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # cv2.imshow("SAM3 - Segmented Depth", cv2.applyColorMap(norm, cv2.COLORMAP_JET))

            # --- Publish segmented RGBD over ZMQ ---
            self._publish_segmented_rgbd(seg_rgb, seg_depth if depth_u16 is not None else None, prompt)

            print("[SAM3] Done. Results shown. Press 's' again to segment a new frame.")
            return

    def _publish_segmented_rgbd(self, seg_rgb, seg_depth, prompt):
        """
        Pack and publish the segmented RGBD via ZMQ PUB.

        Format (msgpack dict):
          color_img  : JPEG bytes  — segmented BGR colour (uint8, black outside mask)
          depth_raw  : raw bytes   — segmented uint16 depth, tobytes() (zero outside mask)
          depth_shape: [H, W]      — needed to reshape depth_raw on the receiver side
          prompt     : str         — the text prompt that produced this mask
          timestamp  : float       — time.time()

        Receiver (anygrasp side) example:
          msg   = socket.recv()
          data  = msgpack.unpackb(msg)
          color = cv2.imdecode(np.frombuffer(data[b'color_img'], np.uint8), cv2.IMREAD_COLOR)
          h, w  = data[b'depth_shape']
          depth = np.frombuffer(data[b'depth_raw'], dtype=np.uint16).reshape(h, w)
        """
        # Encode colour as JPEG (lossy but compact; mask already zeroed bg so file is small)
        ok, color_buf = cv2.imencode('.jpg', seg_rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            logger.error("Failed to JPEG-encode segmented colour — not publishing.")
            return

        # Encode depth as raw uint16 bytes (lossless, required for accurate grasping)
        if seg_depth is not None:
            depth_raw = seg_depth.astype(np.uint16).tobytes()
            depth_shape = list(seg_depth.shape[:2])   # [H, W]
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
        """Stdin-driven loop used when --no-visualize is set. No cv2 window needed."""
        logger.info("Running headless. Commands: 's' + Enter to segment, 'q' + Enter to quit.")
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

    # ------------------------------------------------------------------

    def run(self):
        """Starts the native python threads."""
        t_color = threading.Thread(target=self._color_subscriber_thread, daemon=True)
        t_depth = threading.Thread(target=self._depth_subscriber_thread, daemon=True)

        t_color.start()
        t_depth.start()

        if self.visualize:
            # Display runs on the main thread so cv2.imshow works properly (required on Macs/Windows)
            self._display_thread()
        else:
            self._headless_loop()


SEG_PUB_PORT = 5560  # local port — anygrasp subscribes to tcp://127.0.0.1:5560

if __name__ == "__main__":
    # ---------------------------------------------------------
    # IMPORTANT: Update these to match your Jetson's IP & Ports
    # ---------------------------------------------------------
    JETSON_IP  = "192.168.11.9"
    COLOR_PORT = "10031"
    DEPTH_PORT = "10033"

    parser = argparse.ArgumentParser(description="SAM3 streaming segmenter with ZMQ publisher")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable the OpenCV viewer window; use stdin to trigger segmentation")
    args = parser.parse_args()

    viewer = DualStreamViewer(
        color_ip_port=f"{JETSON_IP}:{COLOR_PORT}",
        depth_ip_port=f"{JETSON_IP}:{DEPTH_PORT}",
        visualize=not args.no_visualize,
    )
    viewer.run()