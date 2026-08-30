"""
FastAPI service for the standalone hieroglyphics pipeline.

The original main.py remains unchanged and keeps the CLI commands.
This file exposes the same pipeline over HTTP:

    POST /api/hieroglyphics/translate
    GET  /health
    GET  /docs
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
UPLOAD_DIR = PROJECT_ROOT / "data" / "api_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Stage modules resolve their config relative to the project, while some of the
# existing paths are relative to the current working directory. Keep the API
# process rooted at the standalone project.
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("hieroglyphics-api")

app = FastAPI(
    title="Hieroglyphics Recognition and Translation API",
    description=(
        "Detection → Classification → Sequence Processing → Groq Translation"
    ),
    version="1.0.0",
)


class TranslationResponse(BaseModel):
    """Public response: final translation only, not intermediate stage outputs."""
    english_translation: str = ""
    arabic_translation: str = ""
    confidence: str = "low"
    caveats: str = ""


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(f"config.yaml not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_project_path(value: str | None) -> str | None:
    """Resolve a configured path without changing the YAML file format."""
    if not value:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def _validate_runtime_files(cfg: dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    missing: list[str] = []
    for key in ("stage1_weights", "stage2_weights", "stage3_weights", "class_map"):
        configured = paths.get(key)
        if not configured:
            missing.append(f"paths.{key} is not configured")
            continue
        resolved = _resolve_project_path(configured)
        if resolved and not os.path.exists(resolved):
            missing.append(f"{key}: {resolved}")

    if missing:
        raise FileNotFoundError(
            "Required model/data files are missing:\n- " + "\n- ".join(missing)
        )


def _save_upload(contents: bytes, original_name: str | None) -> Path:
    if not contents:
        raise ValueError("Uploaded image is empty")

    suffix = Path(original_name or "image.jpg").suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    if suffix not in allowed:
        raise ValueError("Supported image formats: jpg, jpeg, png, webp, bmp, tif, tiff")

    if len(contents) > 15 * 1024 * 1024:
        raise ValueError("Image is too large; maximum size is 15 MB")

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(contents)
    return target


def _run_full_pipeline(image_path: str) -> dict[str, Any]:
    """Run all four stages internally and return only the final translation."""
    # The original stage files use print() for CLI progress. Capture those
    # messages so the API response/log is not polluted with intermediate data.
    silent_stdout = io.StringIO()
    with contextlib.redirect_stdout(silent_stdout):
        from stage1_detection.detect import run_detection
        from stage2_classification.classify import HieroglyphClassifierInference
        from stage3_sequence.sequence import Stage3Pipeline
        from stage4_translation.translator import Stage4Pipeline

        cfg = _load_config()
        _validate_runtime_files(cfg)
        paths = cfg["paths"]
        detection_cfg = cfg.get("detection", {})

        stage1_weights = _resolve_project_path(paths.get("stage1_weights"))
        stage2_weights = _resolve_project_path(paths.get("stage2_weights"))

        detections = run_detection(
            image_path=image_path,
            weights_path=stage1_weights,
            conf=float(detection_cfg.get("conf_threshold", 0.05)),
            save=False,
        )
        if not detections:
            raise ValueError("No hieroglyphic symbols were detected in the image")

        classifier = HieroglyphClassifierInference(weights_path=stage2_weights)
        enriched = classifier.classify_detections(image_path, detections)

        sequence_pipeline = Stage3Pipeline(image_path=image_path)
        sequence = sequence_pipeline.process(enriched)

        translator = Stage4Pipeline(image_path=image_path)
        translation = translator.translate(sequence)

    result = translation.to_dict()
    return {
        "english_translation": result.get("english_translation", ""),
        "arabic_translation": result.get("arabic_translation", ""),
        "confidence": result.get("confidence", "low"),
        "caveats": result.get("caveats", ""),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hieroglyphics"}


@app.post("/api/hieroglyphics/translate", response_model=TranslationResponse)
async def translate_hieroglyphics(file: UploadFile = File(...)):
    """Run detection, classification, sequence processing, and translation."""
    image_path: Path | None = None
    try:
        contents = await file.read()
        image_path = _save_upload(contents, file.filename)
        return await run_in_threadpool(_run_full_pipeline, str(image_path))
    except (ValueError, FileNotFoundError) as exc:
        logger.exception("Hieroglyphics request could not be processed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Hieroglyphics pipeline failed")
        raise HTTPException(
            status_code=502,
            detail="Hieroglyphics pipeline failed; check the server log for the exact error",
        ) from exc
    finally:
        await file.close()
        if image_path is not None:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete temporary image: %s", image_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8001, reload=False)
