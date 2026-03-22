# 🌊 Live Flood Monitor

Real-time flood detection and vehicle tracking via your phone browser.  
Built with **U-Net + MobileNetV2** (flood segmentation) and **YOLOv8n + ByteTrack** (vehicle tracking).

---

## How it works

| Component | Details |
|-----------|---------|
| Flood model | U-Net / MobileNetV2 encoder, 384×384 input |
| Vehicle detector | YOLOv8n (COCO), ByteTrack tracker |
| Processing | 640×360, every 2nd frame for segmentation |
| Browser camera | WebRTC via `streamlit-webrtc` |

---

## Deploying to Streamlit Cloud (step by step)

### Step 1 — Install Git LFS (one-time)

The model file is large so it uses Git LFS.

```bash
# macOS
brew install git-lfs

# Ubuntu / WSL
sudo apt install git-lfs

# Then enable it
git lfs install
```

### Step 2 — Create a GitHub repo

1. Go to https://github.com/new
2. Name it: `flood-monitor-app`
3. Set to **Public** (required for free Streamlit Cloud)
4. Do **not** add README or .gitignore (you already have them)
5. Click **Create repository**

### Step 3 — Push this project

```bash
cd flood-monitor-app

git init
git lfs install
git lfs track "models/*.pt"

git add .
git commit -m "Initial commit — flood monitor app"

git remote add origin https://github.com/YOUR_USERNAME/flood-monitor-app.git
git branch -M main
git push -u origin main
```

> ⚠️ Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 4 — Upload your model file

```bash
# Copy your checkpoint into the models/ folder first
cp /path/to/flood_seg_best_fast.pt models/

git add models/flood_seg_best_fast.pt
git commit -m "Add flood segmentation checkpoint"
git push
```

Git LFS will upload the `.pt` file to GitHub's LFS storage (1 GB free).

### Step 5 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **New app**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/flood-monitor-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy**

Streamlit Cloud will install all dependencies from `requirements.txt` and `packages.txt` automatically. First deploy takes ~5 minutes.

### Step 6 — Open on your phone

1. Copy the public URL from Streamlit Cloud (e.g. `https://your-app.streamlit.app`)
2. Open it in your **phone browser** (Chrome or Safari)
3. Tap **START** and allow camera access
4. Mount phone on dashboard — rear camera faces road
5. Red overlay = flood zone · Red box border = vehicle in flood

---

## Folder structure

```
flood-monitor-app/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Python dependencies
├── packages.txt            ← System dependencies (apt)
├── .gitattributes          ← Git LFS config for .pt files
├── .gitignore
├── .streamlit/
│   └── config.toml         ← Dark theme + server settings
└── models/
    └── flood_seg_best_fast.pt   ← Your checkpoint (tracked by Git LFS)
```

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Performance tips

Since Streamlit Cloud runs on CPU only:

- Frames are resized to 640×360 before inference
- Flood segmentation runs every 2 frames (cached otherwise)
- YOLOv8n is the lightest YOLO variant
- Expect ~3–6 fps on Streamlit Cloud CPU

For better performance, run locally with a GPU.
