import argparse
import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageOps, ImageDraw


def extract_images(pdf_path: Path, min_side: int = 120):
    doc = fitz.open(pdf_path)
    out = []
    seen = set()
    for pno in range(len(doc)):
        page = doc[pno]
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen:
                continue
            seen.add(xref)
            base = doc.extract_image(xref)
            if not base:
                continue
            data = base.get("image")
            if not data:
                continue
            im = Image.open(io.BytesIO(data)).convert("RGB")
            if min(im.size) < min_side:
                continue
            out.append((im.width * im.height, pno, im))
    doc.close()
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def make_montage(images, out_path: Path, title: str, cols: int = 4, rows: int = 3, tile: int = 300):
    n = min(len(images), cols * rows)
    canvas_w = cols * tile + 60
    canvas_h = rows * tile + 120
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 24, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), title, fill=(240, 240, 240))

    for i in range(n):
        _, pno, im = images[i]
        thumb = ImageOps.fit(im, (tile - 10, tile - 10), method=Image.Resampling.LANCZOS)
        r = i // cols
        c = i % cols
        x = 30 + c * tile
        y = 60 + r * tile
        canvas.paste(thumb, (x, y))
        draw.text((x + 6, y + tile - 26), f"p{pno+1}", fill=(255, 210, 120))

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Create APRICOT sample-image montage from a PDF containing APRICOT figures.")
    parser.add_argument("--pdf", default="apricot_supplemental.pdf")
    parser.add_argument("--out", default="slide_visuals_apricot/apricot_dataset_samples.png")
    parser.add_argument("--title", default="APRICOT Dataset Visual Samples")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images = extract_images(pdf_path)
    if not images:
        raise RuntimeError(f"No suitable images found in {pdf_path}")
    make_montage(images, out_path, args.title)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
