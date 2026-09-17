"""Explicit WSGI entrypoint for Gunicorn/Render."""
from app import create_app

app = create_app()
