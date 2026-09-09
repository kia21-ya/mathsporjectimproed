"""Complex number and imaginary plotting engine for mathematical visualizations."""

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


def _safe_locals_complex() -> dict[str, Any]:
    """Return safe namespace for complex number evaluations."""
    return {
        "x": sp.Symbol("x"),
        "y": sp.Symbol("y"),
        "t": sp.Symbol("t"),
        "i": sp.I,
        "j": sp.I,  # Alternative for j notation
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
        "re": sp.re,
        "im": sp.im,
        "arg": sp.arg,
        "conjugate": sp.conjugate,
    }


def parse_complex_expr(expression: str):
    """Parse expression that may contain complex numbers and imaginary units."""
    expression = expression.strip()
    expression = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+",
        lambda match: "**" + match.group(0).translate(_SUPERSCRIPT_TRANSLATION),
        expression,
    )
    # Support both 'i' and 'j' for imaginary unit
    expression = expression.replace(" i", " *I").replace("(i", "(*I")
    expression = expression.replace(" j", " *I").replace("(j", "(*I")
    if not expression:
        raise ValueError("Expression cannot be empty.")
    if expression.count("=") > 1:
        raise ValueError("An equation can contain only one equals sign.")
    if "=" in expression:
        left, right = expression.split("=", 1)
        expression = f"({left}) - ({right})"
    return sympy_parse_expr(
        expression,
        local_dict=_safe_locals_complex(),
        transformations=_TRANSFORMATIONS,
        evaluate=True,
    )


def clean_array(values):
    """Clean array by replacing non-finite values with NaN."""
    arr = np.asarray(values, dtype=complex)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def complex_plot_2d(expression: str, x_min=-10, x_max=10, points=1000):
    """
    Plot complex function f(x) where x is real.
    Shows magnitude (color), real part (x-axis), imaginary part (y-axis).
    """
    x = sp.Symbol("x")
    expr = parse_complex_expr(expression)
    fn = sp.lambdify(x, expr, modules=["numpy"])
    xs = np.linspace(float(x_min), float(x_max), int(points))
    
    ys = clean_array(fn(xs))
    real_parts = np.real(ys)
    imag_parts = np.imag(ys)
    magnitudes = np.abs(ys)
    
    return {
        "type": "complex_2d",
        "data": [{
            "x": real_parts.tolist(),
            "y": imag_parts.tolist(),
            "mode": "markers",
            "marker": {
                "size": 4,
                "color": magnitudes.tolist(),
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "|f(x)|"}
            },
            "name": expression
        }],
        "layout": {
            "xaxis": {"title": "Real Part"},
            "yaxis": {"title": "Imaginary Part"},
            "title": f"Complex Function: {expression}"
        }
    }


def complex_domain_coloring(expression: str, x_min=-2, x_max=2, y_min=-2, y_max=2, points=300):
    """
    Domain coloring visualization of complex function f(z).
    Each pixel represents a complex number z = x + iy.
    Color represents arg(f(z)), brightness represents |f(z)|.
    """
    x, y = sp.symbols("x y")
    expr = parse_complex_expr(expression)
    
    # Create complex variable z = x + iy
    z = x + sp.I * y
    expr_z = expr.subs([(sp.Symbol("x"), x), (sp.Symbol("y"), y), 
                        (sp.Symbol("t"), x)])
    
    fn = sp.lambdify((x, y), expr_z, modules=["numpy"])
    
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = np.linspace(float(y_min), float(y_max), int(points))
    X, Y = np.meshgrid(xs, ys)
    
    Z = clean_array(fn(X, Y))
    
    # Compute hue (argument) and brightness (magnitude)
    magnitude = np.abs(Z)
    phase = np.angle(Z)
    
    # Normalize for visualization
    hue = (phase + np.pi) / (2 * np.pi)
    brightness = magnitude / (magnitude + 1)  # Normalize to [0, 1]
    
    return {
        "type": "complex_domain_coloring",
        "data": [{
            "x": xs.tolist(),
            "y": ys.tolist(),
            "z": hue.tolist(),
            "colorscale": "Twilight",
            "showscale": True,
            "colorbar": {"title": "arg(f(z))"},
            "type": "heatmap"
        }],
        "layout": {
            "xaxis": {"title": "Real axis"},
            "yaxis": {"title": "Imaginary axis"},
            "title": f"Domain Coloring: {expression}"
        }
    }


def mandelbrot_set(max_iter=100, x_min=-2.5, x_max=1, y_min=-1.25, y_max=1.25, points=400):
    """
    Generate Mandelbrot set visualization.
    Each pixel's color represents iteration count to divergence.
    """
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = np.linspace(float(y_min), float(y_max), int(points))
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y
    
    Z = np.zeros_like(C, dtype=complex)
    M = np.zeros(C.shape, dtype=int)
    
    for i in range(int(max_iter)):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] = i
    
    return {
        "type": "mandelbrot",
        "data": [{
            "x": xs.tolist(),
            "y": ys.tolist(),
            "z": M.tolist(),
            "colorscale": "Hot",
            "showscale": True,
            "colorbar": {"title": "Iterations to divergence"},
            "type": "heatmap"
        }],
        "layout": {
            "xaxis": {"title": "Real axis"},
            "yaxis": {"title": "Imaginary axis"},
            "title": "Mandelbrot Set"
        }
    }


def julia_set(c_real=-0.7, c_imag=0.27015, max_iter=100, x_min=-1.5, x_max=1.5, y_min=-1.5, y_max=1.5, points=400):
    """
    Generate Julia set visualization.
    c is a complex parameter; different values generate different Julia sets.
    """
    c = complex(float(c_real), float(c_imag))
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = np.linspace(float(y_min), float(y_max), int(points))
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    
    M = np.zeros(Z.shape, dtype=int)
    
    for i in range(int(max_iter)):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + c
        M[mask] = i
    
    return {
        "type": "julia",
        "data": [{
            "x": xs.tolist(),
            "y": ys.tolist(),
            "z": M.tolist(),
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {"title": "Iterations to divergence"},
            "type": "heatmap"
        }],
        "layout": {
            "xaxis": {"title": "Real axis"},
            "yaxis": {"title": "Imaginary axis"},
            "title": f"Julia Set (c = {c_real} + {c_imag}i)"
        }
    }


def riemann_sphere_stereographic(expression: str, x_min=-2, x_max=2, y_min=-2, y_max=2, points=300):
    """
    Stereographic projection of complex function onto Riemann sphere.
    Visualizes the extended complex plane including the point at infinity.
    """
    x, y = sp.symbols("x y")
    expr = parse_complex_expr(expression)
    
    # Create complex variable z = x + iy
    z = x + sp.I * y
    expr_z = expr.subs([(sp.Symbol("x"), x), (sp.Symbol("y"), y), 
                        (sp.Symbol("t"), x)])
    
    fn = sp.lambdify((x, y), expr_z, modules=["numpy"])
    
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = np.linspace(float(y_min), float(y_max), int(points))
    X, Y = np.meshgrid(xs, ys)
    
    Z = clean_array(fn(X, Y))
    
    # Compute stereographic projection
    r_squared = np.abs(Z)**2
    X_sphere = 2 * np.real(Z) / (r_squared + 1)
    Y_sphere = 2 * np.imag(Z) / (r_squared + 1)
    Z_sphere = (r_squared - 1) / (r_squared + 1)
    
    return {
        "type": "riemann_sphere",
        "data": [{
            "x": X_sphere.tolist(),
            "y": Y_sphere.tolist(),
            "z": Z_sphere.tolist(),
            "type": "surface",
            "colorscale": "Viridis",
            "showscale": True
        }],
        "layout": {
            "scene": {
                "xaxis": {"title": "X"},
                "yaxis": {"title": "Y"},
                "zaxis": {"title": "Z"}
            },
            "title": f"Riemann Sphere: {expression}"
        }
    }


def trig_complex_plot(expression: str, x_min=-10, x_max=10, y_min=-10, y_max=10, points=150):
    """
    Plot real and imaginary parts of trigonometric function over 2D domain.
    Useful for visualizing functions like sin(x+iy), cos(x+iy), etc.
    """
    x, y = sp.symbols("x y")
    expr = parse_complex_expr(expression)
    
    fn = sp.lambdify((x, y), expr, modules=["numpy"])
    
    xs = np.linspace(float(x_min), float(x_max), int(points))
    ys = np.linspace(float(y_min), float(y_max), int(points))
    X, Y = np.meshgrid(xs, ys)
    
    Z = clean_array(fn(X, Y))
    real_parts = np.real(Z)
    imag_parts = np.imag(Z)
    
    return {
        "type": "trig_complex",
        "data": [
            {
                "x": xs.tolist(),
                "y": ys.tolist(),
                "z": real_parts.tolist(),
                "type": "surface",
                "colorscale": "Blues",
                "name": "Real part"
            },
            {
                "x": xs.tolist(),
                "y": ys.tolist(),
                "z": imag_parts.tolist(),
                "type": "surface",
                "colorscale": "Reds",
                "name": "Imaginary part",
                "visible": False
            }
        ],
        "layout": {
            "scene": {
                "xaxis": {"title": "x"},
                "yaxis": {"title": "y"},
                "zaxis": {"title": "Value"}
            },
            "title": f"Trigonometric Complex Function: {expression}",
            "showlegend": True
        }
    }
