"""
main.py — Entry point
Gunicorn imports `app` from here. 
For local run: python main.py
"""
import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Trading bot starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
