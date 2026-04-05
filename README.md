# 🌤️ Unveiling Atmospheric Secrets — Cloud Pattern Segmentation for Enhanced Weather Prediction

A deep-learning web application that segments satellite imagery into **4 distinct cloud pattern classes** (Fish, Flower, Gravel, Sugar) using a **UNet++ model with EfficientNet-B4 encoder**, then delivers **rule-based weather analysis** from the detected patterns.

---

## ✨ Features

- 🌍 **Interactive CesiumJS Globe** — click 4 points on a 3-D globe to define a bounding box, then fetch real satellite imagery for that region
- 🤖 **Cloud Segmentation (UNet++)** — pixel-level classification of satellite images into 4 cloud types using a pretrained deep-learning model
- ⛅ **Rule-Based Weather Analysis** — automatically derives weather insights (rainfall probability, atmospheric stability, etc.) from detected cloud patterns
-  **Dockerised** — one-command startup with Docker Compose; individual Dockerfiles for frontend and backend
- 💾 **Persistent Storage** — uploaded satellite images are stored in `data/` on the host machine

---

## 📁 Project Structure

```
.
├── docker-compose.yml            # Multi-container orchestration
├── .dockerignore                 # Files excluded from Docker builds
├── .gitignore
├── DOCKER.md                     # Docker-specific quick-start notes
│
├── backend/
│   ├── Dockerfile                # Python 3.12-slim image
│   ├── main.py                   # FastAPI application (all endpoints)
│   ├── inference2.py             # Custom UNet model loading & inference
│   ├── weather_rules.py          # Rule-based weather analysis
│   ├── requirements.txt          # Python dependencies
│   ├── Unet++/
│   │   ├── best_model (2).pth    # UNet++ pre-trained weights (~84 MB)
│   │   ├── inference.py          # UNet++ inference script
│   │   └── ml_project.py         # Training script (originally Google Colab)
│   └── UnetWithCls/
│       ├── best.pth              # UNetWithCls pre-trained weights (active model)
│       ├── initial_model.pth     # Initial checkpoint
│       └── inference.py          # UNetWithCls inference script
│
├── frontend/
│   ├── Dockerfile                # Nginx:alpine image
│   ├── nginx.conf                # Nginx configuration
│   ├── index.html                # Main page
│   ├── app.js                    # CesiumJS globe + API integration
│   └── style.css                 # Styling
│
└── data/                         # Saved satellite images (Docker volume)
```

---

## 🧩 Cloud Classes

| Class | Description | Weather Implication |
|-------|-------------|---------------------|
| **Fish** | Organised deep convection — fish-shaped cloud streets | Moderate to heavy rainfall |
| **Flower** | Open-cell convection — rosette-like clusters | Post-frontal environments, scattered showers |
| **Gravel** | Small uniform formations | Stable boundary layer, fair weather |
| **Sugar** | Thin, scattered shallow clouds | Trade-wind regions, fair weather |

---

## 🛠️ Prerequisites

Make sure the following are installed on your machine before proceeding:

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| **Git** | any | Clone the repository |
| **Python** | 3.12+ | Run the backend (local path) |
| **pip** | 23+ | Install Python packages |
| **Docker** | 24+ | Run via Docker (optional but recommended) |
| **Docker Compose** | v2 (`docker compose`) | Orchestrate containers (optional) |

> **Note:** A **CUDA-capable GPU** is optional but will significantly speed up model inference. The application runs on CPU by default.

---

## 🚀 Getting Started

### Option A — Docker (Recommended)

This is the easiest way to run the project. Docker handles all dependencies for you.

#### 1. Clone the Repository

```bash
git clone https://github.com/hamim-87/Cloud-Classification
cd Cloud-Classification
```
#### 2. Start All Services

```bash
docker compose up -d
```

This builds and starts:
- **Frontend**: [http://localhost:3000](http://localhost:3000) — CesiumJS satellite viewer
- **Backend API**: [http://localhost:8000](http://localhost:8000) — FastAPI server

#### 3. Verify Everything Is Running

```bash
docker compose ps          # Check container status
curl http://localhost:8000/ # Should return {"status":"ok",...}
```

#### 4. View Logs

```bash
docker compose logs -f backend   # Backend logs
docker compose logs -f frontend  # Frontend (Nginx) logs
```

#### 5. Stop All Services

```bash
docker compose down
```

#### Rebuild After Code Changes

```bash
docker compose up --build -d
```

---

### Option B — Local Development (Without Docker)

Follow these steps if you prefer to run the project without Docker.

#### 1. Clone the Repository

```bash
git clone https://github.com/hamim-87/Cloud-Classification
cd Cloud-Classification
```

#### 2. Create and Activate a Python Virtual Environment

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate it
# On macOS / Linux:
source .venv/bin/activate

# On Windows (Command Prompt):
.venv\Scripts\activate.bat

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

> You should see `(.venv)` prepended to your shell prompt.

#### 3. Install Backend Dependencies

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

> ⚠️ **PyTorch note:** The `requirements.txt` pulls the default (CPU) build of PyTorch. If you have a CUDA GPU and want GPU acceleration, install PyTorch manually first by following the instructions at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally), then run `pip install -r backend/requirements.txt`.



#### 4. Run the Backend Server

```bash
cd backend
python main.py
```

The API will be available at **http://localhost:8000**.

You should see output like:

```
INFO:     Loading cloud segmentation model …
INFO:     Segmentation model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 5. Serve the Frontend

Open a **new terminal** (keep the backend running), navigate back to the project root, and start a simple static file server:

```bash
cd frontend
python -m http.server 3000
```

Now open your browser and go to **http://localhost:3000**.

> **Alternatively**, you can open `frontend/index.html` directly in your browser (use `file://` path). The frontend auto-detects the backend URL based on the browser's hostname so it will connect to `http://localhost:8000`.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_PATH` | ❌ No (auto-detected) | Override path to model weights (default: `backend/UnetWithCls/best.pth`) |

A `.env` file is not required for the default setup.

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check — returns `{"status": "ok"}` |
| `POST` | `/save-image` | Upload and save a satellite image. Form fields: `image` (file), `filename` (string) |
| `GET` | `/images` | List all saved satellite images with their names and sizes |
| `POST` | `/predict-upload` | Upload an image, run UNet++ segmentation, and return masks + weather analysis |

### Example Requests

**Health check:**
```bash
curl http://localhost:8000/
```

**Upload and save an image:**
```bash
curl -X POST http://localhost:8000/save-image \
  -F "image=@/path/to/satellite.png" \
  -F "filename=my_image.png"
```

**Run cloud segmentation:**
```bash
curl -X POST http://localhost:8000/predict-upload \
  -F "image=@/path/to/satellite.png"
```

**List saved images:**
```bash
curl http://localhost:8000/images
```

Interactive API documentation is available at **http://localhost:8000/docs** (Swagger UI) and **http://localhost:8000/redoc** (ReDoc) when the backend is running.

---

## 🤖 Model Details

### Architecture

- **Active Model**: Custom `TimmUNetWithCls` — a UNet-style encoder–decoder with a shared classification head
- **Encoder**: `efficientnet_b1` via [timm](https://github.com/huggingface/pytorch-image-models) (ImageNet pre-trained)
- **Output classes**: 4 (Fish, Flower, Gravel, Sugar)
- **Input size**: 384 × 384 px (images are resized at inference time)
- **Active weights file**: `backend/UnetWithCls/best.pth`
- **Alternative (UNet++)**: weights at `backend/Unet++/best_model (2).pth`

### Inference Pipeline (`backend/inference2.py`)

1. Load the pre-trained UNet++ model weights
2. Resize and normalise the input satellite image
3. Forward pass through the model to obtain logits
4. Apply sigmoid activation → binary masks per class
5. Compute per-class confidence scores and pixel-coverage percentages
6. Encode masks as base64 PNG for the API response

### Weather Analysis (`backend/weather_rules.py`)

After segmentation, the detected cloud classes are fed into a deterministic rule engine that maps each class (or combination) to weather conditions such as:
- Rainfall probability
- Wind conditions
- Atmospheric stability indicators
- Visibility estimates

---

## 🧑‍🔬 Training

The training script is located at `backend/Unet++/ml_project.py`. It was originally developed in Google Colab and trains UNet++ on a labelled cloud-pattern satellite dataset.

### Steps to Retrain

1. **Prepare your dataset** — images labelled with binary masks for each of the 4 cloud classes (Fish, Flower, Gravel, Sugar). The dataset used originally is the [Understanding Clouds from Satellite Images](https://www.kaggle.com/c/understanding_cloud_organization) Kaggle competition dataset.

2. **Update data paths** inside `ml_project.py` to point to your local dataset.

3. **Run the training script** (GPU recommended):
   ```bash
   python backend/Unet++/ml_project.py
   ```

4. **Save the best checkpoint** into `backend/Unet++/` or `backend/UnetWithCls/` and update `MODEL_WEIGHTS` in `main.py` accordingly.

---

## 🐛 Troubleshooting

### Backend fails to start — model file not found

```
FileNotFoundError: backend/UnetWithCls/best.pth
```
**Solution:** Make sure the model weights file exists at `backend/UnetWithCls/best.pth`. The file may not be included in a shallow clone. Check with:
```bash
ls -lh "backend/UnetWithCls/best.pth"
```

---

### `pip install` fails for `torch` / `torchvision`

**Solution:** The default pip index may time out for large packages. Try:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
```

---

### "Connection refused" when the frontend tries to reach the backend

**Solution:**
1. Confirm the backend is running: `curl http://localhost:8000/`
2. Check that you started the backend before opening the frontend
3. Make sure no firewall rule is blocking port 8000

---

### Port already in use (Docker)

```
Error: bind: address already in use
```
**Solution:**
```bash
docker compose down        # Stop existing containers
docker compose up -d       # Restart
```
Or edit port mappings in `docker-compose.yml` (e.g., `"3001:80"` for the frontend).

---

### Data folder not writable (Docker)

```
PermissionError: [Errno 13] Permission denied: 'data/'
```
**Solution:**
```bash
chmod 777 data/   # Grant write permissions on the host data directory
```

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, Vanilla JavaScript, [CesiumJS](https://cesium.com/cesiumjs/) |
| **Backend** | Python 3.12, [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/) |
| **ML / Deep Learning** | [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), [Albumentations](https://albumentations.ai/) |
| **Image Processing** | [Pillow](https://pillow.readthedocs.io/), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/) |
| **Containerisation** | [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/), [Nginx](https://nginx.org/) |

---

## 📜 License

This project is provided for educational and research purposes. See [LICENSE](LICENSE) for details (or add your preferred license).

---

> **Version:** 2.0.0 &nbsp;|&nbsp; **Last Updated:** April 2026