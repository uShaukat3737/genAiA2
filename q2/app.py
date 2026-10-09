"""Flask demo: upload or capture a face/sketch, auto-detect its domain, convert it."""
import argparse
import base64
import io
import logging

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from inference import DIRECTIONS, is_sketch, load_generators, pick_device, translate

log = logging.getLogger("cyclegan.app")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
MODES = ("auto",) + DIRECTIONS
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
                               "style-src 'self'; script-src 'self'; frame-ancestors *",
    "X-Content-Type-Options": "nosniff",
    "Strict-Transport-Security": "max-age=31536000",
    "Referrer-Policy": "no-referrer",
}
# X-Frame-Options deliberately omitted: Colab serves the app inside an iframe.


def to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def read_upload(file):
    try:
        img = Image.open(file.stream)
        img.verify()  # verify() invalidates the handle, so reopen to decode
        file.stream.seek(0)
        img = Image.open(file.stream)
        if img.width * img.height > MAX_PIXELS:
            return None
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        return None


def create_app(generators, image_size, device):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.after_request
    def add_headers(resp):
        resp.headers.update(SECURITY_HEADERS)
        return resp

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/convert")
    def convert():
        file = request.files.get("image")
        mode = request.form.get("mode", "auto")
        if file is None:
            return jsonify(error="no image uploaded"), 400
        if mode not in MODES:
            return jsonify(error="invalid mode"), 400
        img = read_upload(file)
        if img is None:
            return jsonify(error="file is not a supported image"), 400
        detected = "sketch" if is_sketch(img) else "photo"
        direction = mode if mode != "auto" else ("sketch2photo" if detected == "sketch" else "photo2sketch")
        try:
            out = translate(generators[direction], img, image_size, device)
        except RuntimeError:
            log.exception("translation failed direction=%s size=%s", direction, img.size)
            return jsonify(error="conversion failed"), 500
        return jsonify(detected=detected, direction=direction, image=to_data_url(out))

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="image too large (max 10 MB)"), 413

    return app


def main():
    p = argparse.ArgumentParser(description="CycleGAN face <-> sketch web demo")
    p.add_argument("--checkpoint", default="./checkpoints/latest.pth")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to expose on your LAN")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = pick_device(args.device)
    gens, size = load_generators(args.checkpoint, device)
    create_app(gens, size, device).run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
