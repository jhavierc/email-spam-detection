"""Run the development server:  python -m app  [--host 127.0.0.1] [--port 8000]"""
import argparse

from . import create_app

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    create_app().run(host=a.host, port=a.port, debug=a.debug, use_reloader=False)
