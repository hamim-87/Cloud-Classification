"""
Eyes on Cloud — FastAPI Backend
Receives satellite images from the frontend, stores them in `data/`,
runs cloud-pattern segmentation via a trained UNet++ model,
and provides rule-based weather analysis.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference2 import CloudSegmentationModel
from weather_rules import compute_weather_fusion

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
# load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_WEIGHTS = BACKEND_DIR / "UnetWithCls" / "best.pth"

# ── Global references (filled at startup) ────────────────────────────────────
model: CloudSegmentationModel | None = None


# ── Lifespan: load model at startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading cloud segmentation model …")
    model = CloudSegmentationModel.get_instance(MODEL_WEIGHTS)
    logger.info("Segmentation model ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Eyes on Cloud API",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow frontend (served on different port or file://) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Eyes on Cloud API is running"}


# ── Save image ───────────────────────────────────────────────────────────────
@app.post("/save-image")
async def save_image(
    image: UploadFile = File(...),
    filename: str = Form(...),
):
    """
    Receive a satellite image from the frontend and save it to the data/ folder.
    """
    safe_name = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in filename
    )
    if not safe_name.lower().endswith((".png", ".jpg", ".jpeg")):
        safe_name += ".png"

    save_path = DATA_DIR / safe_name

    counter = 1
    original_stem = save_path.stem
    while save_path.exists():
        save_path = DATA_DIR / f"{original_stem}_{counter}{save_path.suffix}"
        counter += 1

    contents = await image.read()
    save_path.write_bytes(contents)

    return {
        "status": "success",
        "message": "Image saved successfully",
        "filename": save_path.name,
        "path": str(save_path),
        "size_bytes": len(contents),
    }


# ── List images ──────────────────────────────────────────────────────────────
@app.get("/images")
def list_images():
    """List all saved satellite images."""
    files = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            files.append({"name": f.name, "size_bytes": f.stat().st_size})
    return {"images": files, "count": len(files)}


# ── Predict (direct upload) ──────────────────────────────────────────────────
@app.post("/predict-upload")
async def predict_upload(
    image: UploadFile = File(...),
):
    """
    Upload a satellite image and run cloud segmentation immediately.

    The image is saved to data/ and then inference is performed.
    Returns per-class segmentation masks (base64 PNG), confidence scores,
    coverage percentages, detected cloud types, and rule-based weather analysis.
    """
    original_name = image.filename or "upload.png"
    safe_name = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in original_name
    )
    if not safe_name.lower().endswith((".png", ".jpg", ".jpeg")):
        safe_name += ".png"

    save_path = DATA_DIR / safe_name

    counter = 1
    original_stem = save_path.stem
    while save_path.exists():
        save_path = DATA_DIR / f"{original_stem}_{counter}{save_path.suffix}"
        counter += 1

    contents = await image.read()
    save_path.write_bytes(contents)

    logger.info("Running inference on uploaded image %s", save_path.name)
    result = model.predict(save_path)

    # ── 6-step structured fusion weather analysis ────────────────────────
    fusion = compute_weather_fusion(result.get("results", {}))

    logger.info(
        "Fusion probabilities: %s  →  forecast: %s (%s)",
        fusion["probabilities"],
        fusion["forecast"]["type"],
        fusion["forecast"]["primary_cloud"],
    )

    return {
        "status": "success",
        "filename": save_path.name,
        **result,
        **fusion,
        "weather_analysis": fusion["forecast"]["description"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
