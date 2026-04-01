"""
Eyes on Cloud — FastAPI Backend
Receives satellite images from the frontend and stores them in the `data/` folder.
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn

app = FastAPI(title="Eyes on Cloud API", version="1.0.0")

# Allow frontend (served on different port or file://) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data folder — sits alongside the backend folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Eyes on Cloud API is running"}


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
        "message": f"Image saved successfully",
        "filename": save_path.name,
        "path": str(save_path),
        "size_bytes": len(contents),
    }


@app.get("/images")
def list_images():
    """List all saved satellite images."""
    files = []
    for f in sorted(DATA_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            files.append({"name": f.name, "size_bytes": f.stat().st_size})
    return {"images": files, "count": len(files)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
