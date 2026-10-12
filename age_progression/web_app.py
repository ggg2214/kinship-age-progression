"""
FastAPI Interactive Web Application and REST API for Age Progression System
Provides interactive web dashboard, real-time prediction,
timelapse sequence generation, and kinship attention breakdown.
"""

import os
import sys

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import io
import base64
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch

from age_progression.config import AgeProgressionConfig
from age_progression.inference import AgeProgressionPipeline
from age_progression.visualize_samples import create_timeline_strip

app = FastAPI(
    title="Kinship-Guided Age Progression API",
    description="Deep Learning Age Progression System for Long-term Missing Children",
    version="1.0.0"
)

# Global pipeline instance
pipeline: Optional[AgeProgressionPipeline] = None
current_device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_file = "checkpoints/age_progression_latest.pt"


def get_pipeline() -> AgeProgressionPipeline:
    global pipeline
    if pipeline is None:
        config = AgeProgressionConfig(image_size=128, device=current_device)
        ckpt = checkpoint_file if os.path.exists(checkpoint_file) else None
        print(f"[INFO] Initializing AgeProgressionPipeline on {current_device} (ckpt: {ckpt})...")
        pipeline = AgeProgressionPipeline(config=config, checkpoint_path=ckpt, device=current_device)
    return pipeline


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def read_uploaded_image(file: Optional[UploadFile]) -> Optional[Image.Image]:
    if not file or not file.filename:
        return None
    contents = file.file.read()
    if len(contents) == 0:
        return None
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")


@app.on_event("startup")
def startup_event():
    get_pipeline()


@app.get("/api/status")
def get_status():
    p = get_pipeline()
    total_params = sum(param.numel() for param in p.model.parameters())
    return {
        "status": "online",
        "device": current_device,
        "checkpoint": checkpoint_file if os.path.exists(checkpoint_file) else "initialized (scratch)",
        "model_parameters": total_params,
        "image_size": p.config.image_size
    }


@app.post("/api/predict")
async def predict_aged_face(
    child_file: UploadFile = File(...),
    father_file: Optional[UploadFile] = File(None),
    mother_file: Optional[UploadFile] = File(None),
    target_age: float = Form(30.0),
    source_age: float = Form(5.0),
    enhance_face: bool = Form(False)
):
    """
    Synthesizes predicted face at target_age using child and optional parent photos.
    """
    p = get_pipeline()
    child_img = read_uploaded_image(child_file)
    if child_img is None:
        raise HTTPException(status_code=400, detail="Child face image is required.")

    father_img = read_uploaded_image(father_file)
    mother_img = read_uploaded_image(mother_file)

    aged_img, meta = p.predict(
        child=child_img,
        target_age=target_age,
        father=father_img,
        mother=mother_img,
        source_age=source_age,
        enhance_face=enhance_face
    )

    # Compute identity similarity between child and predicted face
    with torch.no_grad():
        c_tensor = p._prepare_tensor(child_img)
        pred_tensor = p._prepare_tensor(aged_img)
        v_c = p.model.id_encoder(c_tensor)
        v_pred = p.model.id_encoder(pred_tensor)
        id_sim = float(torch.sum(v_c * v_pred, dim=-1).item())

    # Parse attention weights if kinship was provided
    father_pct = 50.0
    mother_pct = 50.0
    attn = meta.get("attn_weights")
    has_parents = (father_img is not None or mother_img is not None)
    if attn is not None:
        w = attn.squeeze().cpu().numpy()
        if len(w) >= 2:
            father_pct = round(float(w[0]) * 100, 1)
            mother_pct = round(float(w[1]) * 100, 1)

    return JSONResponse({
        "status": "success",
        "predicted_image": pil_to_base64(aged_img),
        "target_age": target_age,
        "source_age": source_age,
        "identity_similarity": round(id_sim, 4),
        "has_kinship": has_parents,
        "father_attention_pct": father_pct if has_parents else None,
        "mother_attention_pct": mother_pct if has_parents else None,
        "kinship_vector_norm": round(meta.get("v_kin_norm", 0.0), 4)
    })


@app.post("/api/timelapse")
async def generate_timelapse(
    child_file: UploadFile = File(...),
    father_file: Optional[UploadFile] = File(None),
    mother_file: Optional[UploadFile] = File(None),
    source_age: float = Form(5.0),
    ages: str = Form("10,20,30,40,50,60")
):
    """
    Generates a progression sequence across specified ages and returns both
    individual frames and a combined banner strip.
    """
    p = get_pipeline()
    child_img = read_uploaded_image(child_file)
    if child_img is None:
        raise HTTPException(status_code=400, detail="Child face image is required.")

    father_img = read_uploaded_image(father_file)
    mother_img = read_uploaded_image(mother_file)

    age_list = [int(a.strip()) for a in ages.split(",") if a.strip().isdigit()]
    if not age_list:
        age_list = [10, 20, 30, 40, 50, 60]

    # Generate banner strip
    strip = create_timeline_strip(
        pipeline=p,
        child_img=child_img,
        ages=age_list,
        father_img=father_img,
        mother_img=mother_img,
        source_age=int(source_age)
    )

    # Generate individual age frames
    frames = []
    for a in age_list:
        aged, _ = p.predict(
            child=child_img,
            target_age=a,
            father=father_img,
            mother=mother_img,
            source_age=source_age
        )
        frames.append({
            "age": a,
            "image": pil_to_base64(aged)
        })

    return JSONResponse({
        "status": "success",
        "banner_strip": pil_to_base64(strip),
        "frames": frames
    })


# Static assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Kinship-Guided Age Progression Web UI</h1><p>static/index.html not found.</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("age_progression.web_app:app", host="127.0.0.1", port=8000, reload=False)
