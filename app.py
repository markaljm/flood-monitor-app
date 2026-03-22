import cv2
import av
import torch
import numpy as np
import streamlit as st
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Flood Monitor", page_icon="🌊", layout="centered")

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 384
ENCODER     = "mobilenet_v2"
DECODER_CH  = (128, 64, 32, 16, 8)
MODEL_PATH  = "models/flood_seg_best_fast.pt"
FLOOD_THR   = 5.0
CONF_THR    = 0.35
VEHICLE_IDS = {2, 3, 5, 7, 1}

def _make_tfm():
    ver = tuple(int(x) for x in A.__version__.split(".")[:2])
    resize = A.Resize(height=IMG_SIZE, width=IMG_SIZE) if ver >= (1,4) else A.Resize(IMG_SIZE, IMG_SIZE)
    return A.Compose([resize,
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()])

EVAL_TFM = _make_tfm()

@st.cache_resource
def load_models():
    seg = smp.Unet(encoder_name=ENCODER, encoder_weights=None,
                   decoder_channels=DECODER_CH, in_channels=3, classes=1, activation=None).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    seg.load_state_dict(sd)
    seg.eval()
    det = YOLO("yolov8n.pt")
    return seg, det

seg_model, det_model = load_models()

@torch.inference_mode()
def flood_mask(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = EVAL_TFM(image=rgb)["image"].unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
        prob = torch.sigmoid(seg_model(inp))[0, 0].cpu().numpy()
    h, w = bgr.shape[:2]
    return (cv2.resize(prob, (w, h)) > 0.45).astype(np.uint8)

class Processor(VideoProcessorBase):
    def __init__(self):
        self._n    = 0
        self._mask = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (480, 270))
        h, w = img.shape[:2]
        self._n += 1

        if self._n == 1 or self._n % 2 == 0:
            self._mask = flood_mask(img)

        mask      = self._mask if self._mask is not None else np.zeros((h,w), np.uint8)
        flood_pct = mask.sum() / (h*w) * 100
        flood     = flood_pct >= FLOOD_THR

        if flood:
            tint = img.copy()
            tint[mask > 0] = np.clip(
                tint[mask > 0] * 0.5 + np.array([160,40,0]) * 0.5, 0, 255).astype(np.uint8)
            cv2.addWeighted(tint, 0.5, img, 0.5, 0, img)

        res      = det_model(img, classes=list(VEHICLE_IDS), conf=CONF_THR, verbose=False)
        vehicles = 0
        if res and res[0].boxes is not None:
            vehicles = sum(1 for c in res[0].boxes.cls.cpu().numpy().astype(int) if c in VEHICLE_IDS)

        bar = img.copy()
        cv2.rectangle(bar, (0,0), (w,52), (0,0,0), -1)
        cv2.addWeighted(bar, 0.55, img, 0.45, 0, img)

        status_col = (0,60,220) if flood else (30,200,60)
        cv2.putText(img, "FLOOD" if flood else "OK", (10,26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_col, 2, cv2.LINE_AA)
        cv2.putText(img, f"Vehicles: {vehicles}", (10,46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🌊 Flood Monitor")
st.caption("Point your phone camera at the road. No data is stored.")

webrtc_streamer(
    key="flood",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Processor,
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": "environment"},
            "width":  {"ideal": 480},
            "height": {"ideal": 270},
            "frameRate": {"ideal": 12, "max": 15},
        },
        "audio": False,
    },
    async_processing=True,
)
