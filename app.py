from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import base64
import json
from pydantic import BaseModel
from typing import Dict, Any
from fastapi.middleware.cors import CORSMiddleware

from detectors import (
    detect_hough,
    detect_min_enclosing,
    detect_canny_hough,
    detect_yolo_segmentation
)
from eval_metrics import compute_edge_coverage, compute_hough_confidence

app = FastAPI(title="AI Model Web API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def img_to_base64(img_array):
    # OpenCV imencode implicitly assumes BGR and outputs standard PNG
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/api/detect")
async def detect_image(
    file: UploadFile = File(...),
    method: str = Form("method4"),
    compare_mode: bool = Form(False),
    params: str = Form("{}")
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    parsed_params = json.loads(params)

    methods_dict = {
        "method1": (detect_hough, parsed_params.get("method1", {})),
        "method2": (detect_min_enclosing, parsed_params.get("method2", {})),
        "method3": (detect_canny_hough, parsed_params.get("method3", {})),
        "method4": (detect_yolo_segmentation, parsed_params.get("method4", {}))
    }

    results = {}

    methods_to_run = list(methods_dict.keys()) if compare_mode else [method]

    for method_id in methods_to_run:
        func, current_params = methods_dict[method_id]
        img_copy = image.copy()
        res = func(gray_image, current_params)
        metrics = {}
        if res['success']:
            center = tuple(res['center'])
            radius = res['radius']
            cv2.circle(img_copy, center, radius, (0, 255, 0), 2)
            cv2.circle(img_copy, center, 2, (0, 0, 255), 3)

            try:
                e_map = res.get("debug", {}).get("edge_map")
                if e_map is None:
                    e_map = cv2.Canny(gray_image, 50, 150)
                cov = compute_edge_coverage(center, radius, e_map)
                conf = compute_hough_confidence(res.get("debug", {}), center, radius)
                metrics = {"edge_coverage": cov, "confidence": conf}
            except Exception as e:
                metrics = {"edge_coverage": 0.0, "confidence": 0.0}

        edge_map = res.get('debug', {}).get('edge_map', np.zeros_like(gray_image))
        results[method_id] = {
            "success": res['success'],
            "message": res['diagnostics']['message'],
            "elapsed_ms": res['diagnostics']['elapsed_ms'],
            "center": res.get('center', [0, 0]),
            "radius": res.get('radius', 0),
            "metrics": metrics,
            "result_image": f"data:image/png;base64,{img_to_base64(img_copy)}",
            "edge_image": f"data:image/png;base64,{img_to_base64(edge_map)}"
        }

    return results