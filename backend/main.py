"""
Eyes on Cloud — FastAPI Backend
Receives satellite images from the frontend, stores them in `data/`,
and runs cloud-pattern segmentation via a trained UNet++ model.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import CloudSegmentationModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_WEIGHTS = BACKEND_DIR / "model" / "best_model (2).pth"

# ── Global model reference (filled at startup) ──────────────────────────────
model: CloudSegmentationModel | None = None


# ── Lifespan: load model once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading cloud segmentation model …")
    model = CloudSegmentationModel.get_instance(MODEL_WEIGHTS)
    logger.info("Model ready.")
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

    - **image**: The PNG image file (multipart upload)
    - **filename**: Desired filename (e.g. satellite_2026-02-15_N12.34_S10.12.png)
    """
    # Sanitize filename — keep only safe chars
    safe_name = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in filename
    )
    if not safe_name.lower().endswith((".png", ".jpg", ".jpeg")):
        safe_name += ".png"

    save_path = DATA_DIR / safe_name

    # Avoid overwriting — append counter if file exists
    counter = 1
    original_stem = save_path.stem
    while save_path.exists():
        save_path = DATA_DIR / f"{original_stem}_{counter}{save_path.suffix}"
        counter += 1

    # Write the uploaded file
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


# ── Predict (by filename already in data/) ───────────────────────────────────
@app.post("/predict")
async def predict_by_filename(filename: str = Form(...)):
    """
    Run cloud segmentation on an image that is already saved in `data/`.

    - **filename**: Name of the image file inside the data/ folder.

    Returns per-class segmentation masks (base64 PNG), confidence scores,
    coverage percentages, and a list of detected cloud types.
    """
    image_path = DATA_DIR / filename

    if not image_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Image '{filename}' not found in data folder.",
        )

    if not image_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        raise HTTPException(
            status_code=400,
            detail="Only PNG / JPG images are supported.",
        )

    logger.info("Running inference on %s", filename)
    result = model.predict(image_path)

    return {
        "status": "success",
        "filename": filename,
        **result,
    }


# ── Predict (direct upload) ──────────────────────────────────────────────────
@app.post("/predict-upload")
async def predict_upload(image: UploadFile = File(...)):
    """
    Upload a satellite image and run cloud segmentation immediately.

    The image is saved to data/ and then inference is performed.
    Returns per-class segmentation masks (base64 PNG), confidence scores,
    coverage percentages, and a list of detected cloud types.
    """
    # Sanitize filename
    original_name = image.filename or "upload.png"
    safe_name = "".join(
        c if (c.isalnum() or c in "._-") else "_" for c in original_name
    )
    if not safe_name.lower().endswith((".png", ".jpg", ".jpeg")):
        safe_name += ".png"

    save_path = DATA_DIR / safe_name

    # Avoid overwriting
    counter = 1
    original_stem = save_path.stem
    while save_path.exists():
        save_path = DATA_DIR / f"{original_stem}_{counter}{save_path.suffix}"
        counter += 1

    # Save the upload
    contents = await image.read()
    save_path.write_bytes(contents)

    logger.info("Running inference on uploaded image %s", save_path.name)
    result = model.predict(save_path)

    return {
        "status": "success",
        "filename": save_path.name,
        **result,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
