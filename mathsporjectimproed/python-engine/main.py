from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from visualizers.math_engines import (
    animated_surface,
    default_animated_surface,
    expression_plot,
    implicit_plot,
    lorenz_attractor,
    parametric_plot,
    polar_plot,
    surface_plot,
)

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"

app = FastAPI(title="Algebra Graphing Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlotRequest(BaseModel):
    plot_type: str = "expression"
    expression: str = "sin(x)"
    x_expression: str = "cos(t)"
    y_expression: str = "sin(t)"
    x_min: float = -10
    x_max: float = 10
    y_min: float = -10
    y_max: float = 10
    t_min: float = 0
    t_max: float = 6.28318530718
    points: int = Field(default=1000, ge=20, le=2500)
    frames: int = Field(default=360, ge=60, le=600)


class LorenzRequest(BaseModel):
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 2.6666666667
    dt: float = 0.008
    steps: int = Field(default=3500, ge=500, le=10000)
    frame_size: int = Field(default=8, ge=2, le=50)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/plot")
def plot(req: PlotRequest):
    try:
        p = req.plot_type.lower()
        # Equations and expressions involving y require a two-dimensional plot.
        if p == "expression" and ("=" in req.expression or "y" in req.expression):
            p = "implicit"
        if p == "expression":
            return expression_plot(req.expression, req.x_min, req.x_max, req.points)
        if p == "implicit":
            return implicit_plot(req.expression, req.x_min, req.x_max, req.y_min, req.y_max, min(req.points, 350))
        if p == "parametric":
            return parametric_plot(req.x_expression, req.y_expression, req.t_min, req.t_max, req.points)
        if p == "polar":
            return polar_plot(req.expression, req.t_min, req.t_max, req.points)
        if p == "surface":
            return surface_plot(req.expression, req.x_min, req.x_max, req.y_min, req.y_max, min(req.points, 120))
        if p == "animated_surface":
            if req.expression.strip() in {"", "default"}:
                return default_animated_surface(req.frames)
            return animated_surface(
                req.expression, req.frames, req.x_min, req.x_max, req.y_min, req.y_max, min(req.points, 60)
            )
        raise ValueError(f"Unsupported plot type: {req.plot_type}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/lorenz")
def lorenz(req: LorenzRequest):
    try:
        return lorenz_attractor(req.sigma, req.rho, req.beta, req.dt, req.steps, req.frame_size)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Serve the page as a fallback for browser navigation.
@app.get("/{path:path}")
def frontend_fallback(path: str):
    candidate = (FRONTEND / path).resolve()
    if FRONTEND.resolve() in candidate.parents and candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(FRONTEND / "index.html")
