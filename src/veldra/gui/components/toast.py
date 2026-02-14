"""Toast notification component."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

def toast_container() -> html.Div:
    """Create a container for toasts."""
    return html.Div(
        id="toast-container",
        style={
            "position": "fixed",
            "top": "20px",
            "right": "20px",
            "zIndex": "9999",
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px"
        }
    )

def make_toast(message: str, header: str = "Notification", icon: str = "info", duration: int = 4000) -> dbc.Toast:
    """Create a toast notification."""
    color = "primary"
    if icon == "success": color = "success"
    if icon == "danger": color = "danger"
    if icon == "warning": color = "warning"
    
    return dbc.Toast(
        message,
        header=header,
        duration=duration,
        icon=color,
        dismissable=True,
        is_open=True,
        style={"minWidth": "300px", "opacity": "0.95"}
    )
