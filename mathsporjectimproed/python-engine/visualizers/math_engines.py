"""Numerical math engines used by the FastAPI backend."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr as sympy_parse_expr,
    standard_transformations,
)


_TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)
_SUPERSCRIPT_TRANSLATION = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


def _safe_locals() -> dict[str, Any]:
    return {
        "x": sp.Symbol("x"),
        "y": sp.Symbol("y"),
        "t": sp.Symbol("t"),
        "pi": sp.pi,
        "E": sp.E,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
        "abs": sp.Abs,
    }


def parse_expr(expression: str):
    expression = expression.strip()
    expression = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+",
        lambda match: "**" + match.group(0).translate(_SUPERSCRIPT_TRANSLATION),
        expression,
    )
    if not expression:
        raise ValueError("Expression cannot be empty.")
    if expression.count("=") > 1:
        raise ValueError("An equation can contain only one equals sign.")
    if "=" in expression:
        left, right = expression.split("=", 1)
        expression = f"({left}) - ({right})"
    return sympy_parse_expr(
        expression,
        local_dict=_safe_locals(),
        transformations=_TRANSFORMATIONS,
        evaluate=True,
    )


def clean_array(values):
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def _lambdify(expr, variables):
    """Create a NumPy function only when every variable is an input."""
    if not isinstance(expr, sp.Expr):
        raise ValueError("Enter a complete expression, such as sin(x).")
    variable_names = {variable.name for variable in variables}
    unknown_names = sorted(
        symbol.name for symbol in expr.free_symbols
        if symbol.name not in variable_names
    )
    if unknown_names:
        names = ", ".join(unknown_names)
        expected = ", ".join(variable_names)
        raise ValueError(f"Unknown variable(s): {names}. Use {expected}.")
    return sp.lambdify(variables, expr, modules=["numpy"])


def expression_plot(expression: str, x_min=-10, x_max=10, points=1000):
    x = sp.Symbol("x")
    expr = parse_expr(expression)
    fn = _lambdify(expr, (x,))
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = clean_array(fn(xs))
    return {
        "type": "expression",
        "data": [{"x": xs.tolist(), "y": ys.tolist(), "name": expression}],
        "layout": {"xaxis": {"title": "x"}, "yaxis": {"title": "f(x)"}},
    }


def implicit_plot(expression: str, x_min=-6, x_max=6, y_min=-6, y_max=6, points=180):
    x, y = sp.symbols("x y")
    expr = parse_expr(expression)
    fn = _lambdify(expr, (x, y))
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = np.linspace(float(y_min), float(y_max), int(points))
    X, Y = np.meshgrid(xs, ys)
    Z = clean_array(fn(X, Y))
    return {
        "type": "implicit",
        "data": [{"x": xs.tolist(), "y": ys.tolist(), "z": Z.tolist(), "levels": [0]}],
        "layout": {"xaxis": {"title": "x"}, "yaxis": {"title": "y"}},
    }


def parametric_plot(x_expression: str, y_expression: str, t_min=0, t_max=6.28318530718, points=1200):
    t = sp.Symbol("t")
    fx = _lambdify(parse_expr(x_expression), (t,))
    fy = _lambdify(parse_expr(y_expression), (t,))
    ts = np.linspace(float(t_min), float(t_max), int(points))
    xs = clean_array(fx(ts))
    ys = clean_array(fy(ts))
    return {
        "type": "parametric",
        "data": [{"x": xs.tolist(), "y": ys.tolist(), "name": f"({x_expression}, {y_expression})"}],
        "layout": {"xaxis": {"title": "x"}, "yaxis": {"title": "y"}, "t": ts.tolist()},
    }


def polar_plot(expression: str, t_min=0, t_max=6.28318530718, points=1200):
    t = sp.Symbol("t")
    expr = parse_expr(expression).subs(sp.Symbol("x"), t)
    fr = _lambdify(expr, (t,))
    ts = np.linspace(float(t_min), float(t_max), int(points))
    r = clean_array(fr(ts))
    xs = r * np.cos(ts)
    ys = r * np.sin(ts)
    return {
        "type": "polar",
        "data": [{"x": xs.tolist(), "y": ys.tolist(), "r": r.tolist(), "theta": ts.tolist(), "name": expression}],
        "layout": {"polar": True, "title": f"r = {expression}"},
    }


def surface_plot(expression: str, x_min=-5, x_max=5, y_min=-5, y_max=5, points=100):
    x, y = sp.symbols("x y")
    expr = parse_expr(expression)
    fn = _lambdify(expr, (x, y))
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = np.linspace(float(y_min), float(y_max), int(points))
    X, Y = np.meshgrid(xs, ys)
    Z = clean_array(fn(X, Y))
    return {
        "type": "surface",
        "data": [{"x": xs.tolist(), "y": ys.tolist(), "z": Z.tolist(), "name": expression}],
        "layout": {"title": f"z = {expression}"},
    }


def animated_surface(expression: str, frames=360, x_min=-5, x_max=5, y_min=-5, y_max=5, points=55):
    """
    Returns a compact collection of surface frames.
    The browser owns playback, so the animation can loop forever without
    repeatedly hitting the backend.
    """
    x, y, t = sp.symbols("x y t")
    expr = parse_expr(expression)
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = np.linspace(float(y_min), float(y_max), int(points))
    X, Y = np.meshgrid(xs, ys)
    fn = _lambdify(expr, (x, y, t))

    frame_list = []
    for i in range(int(frames)):
        time_factor = i * 0.1
        Z = clean_array(fn(X, Y, time_factor))
        frame_list.append({
            "x": xs.tolist(),
            "y": ys.tolist(),
            "z": Z.tolist(),
            "camera": {
                "eye": {
                    "x": 1.55 * math.cos(i / frames * 2 * math.pi),
                    "y": 1.55 * math.sin(i / frames * 2 * math.pi),
                    "z": 1.2,
                }
            }
        })
    return {
        "type": "animated_surface",
        "frames": frame_list,
        "frame_count": len(frame_list),
        "fps": 60,
        "expression": expression,
    }


def default_animated_surface(frames=360):
    # Based on the uploaded reference: a time-morphing 3D wave with
    # synchronized camera motion and bounded vertical scale.
    x = np.linspace(-5, 5, 55)
    y = np.linspace(-5, 5, 55)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2) + 1e-8
    out = []
    for i in range(frames):
        time_factor = i * 0.1
        Z = np.sin(R - time_factor) * np.cos(X * 0.2) / (R * 0.5 + 1)
        out.append({
            "x": x.tolist(),
            "y": y.tolist(),
            "z": clean_array(Z).tolist(),
            "camera": {
                "eye": {
                    "x": 1.7 * math.cos(i / frames * 2 * math.pi),
                    "y": 1.7 * math.sin(i / frames * 2 * math.pi),
                    "z": 1.25 + 0.15 * math.sin(time_factor * 0.5),
                }
            }
        })
    return {
        "type": "animated_surface",
        "frames": out,
        "frame_count": frames,
        "fps": 60,
        "expression": "default morphing surface",
    }


def lorenz_attractor(
    sigma=10.0,
    rho=28.0,
    beta=2.6666666667,
    dt=0.008,
    steps=3500,
    frame_size=8,
):
    steps = int(steps)
    sigma, rho, beta, dt = map(float, (sigma, rho, beta, dt))
    arr = np.empty((steps, 3), dtype=float)
    arr[0] = [0.1, 0.0, 0.0]
    for i in range(1, steps):
        x0, y0, z0 = arr[i - 1]
        dx = sigma * (y0 - x0)
        dy = x0 * (rho - z0) - y0
        dz = x0 * y0 - beta * z0
        arr[i] = arr[i - 1] + dt * np.array([dx, dy, dz])
    frame_size = max(1, int(frame_size))
    frames = []
    for end in range(frame_size, steps + 1, frame_size):
        segment = arr[:end]
        frames.append({
            "x": segment[:, 0].tolist(),
            "y": segment[:, 1].tolist(),
            "z": segment[:, 2].tolist(),
        })
    if not frames:
        frames = [{"x": arr[:,0].tolist(), "y": arr[:,1].tolist(), "z": arr[:,2].tolist()}]
    return {
        "type": "lorenz",
        "frames": frames,
        "frame_count": len(frames),
        "fps": 60,
        "parameters": {"sigma": sigma, "rho": rho, "beta": beta, "dt": dt},
    }
