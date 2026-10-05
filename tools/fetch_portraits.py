"""Fetch freely licensed portrait photos of reference actors from Wikimedia.

Reads /Voxa/.voxa-spec/out/reference/actors.json, downloads the Wikipedia
page image plus up to 4 Commons search images per name, keeps JPEG/PNG files
>=300 px wide, and writes photos/index.json with source URLs.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

REF_DIR = Path("/Voxa/.voxa-spec/out/reference")
PHOTO_DIR = REF_DIR / "photos"
USER_AGENT = "VoxaCharacterAnalysis/1.0 (portrait-fetch; research use)"

SLEEP = 0.5


def _request(url, timeout):
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as exc:
            last = exc
            time.sleep(5.0 * (attempt + 1))
    raise last


def _get_json(url):
    with _request(url, 30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_bytes(url):
    with _request(url, 60) as resp:
        return resp.read()


def _image_dims(data):
    """Return (width, height) for JPEG/PNG bytes, or None if not parseable."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            if marker == 0xC0:  # SOF
                h = int.from_bytes(data[i + 5 : i + 7], "big")
                w = int.from_bytes(data[i + 7 : i + 9], "big")
                return w, h
            size = int.from_bytes(data[i + 2 : i + 4], "big")
            i += 2 + size
    return None


def _wiki_page_image(name):
    url = (
        "https://en.wikipedia.org/w/api.php?action=query&titles="
        + urllib.request.quote(name)
        + "&prop=pageimages&piprop=original%7Cthumbnail&format=json&redirects=1"
    )
    data = _get_json(url)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        return (page.get("thumbnail", {}).get("source")
                or page.get("original", {}).get("source"))
    return None


def _commons_images(name):
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        "&gsrsearch=" + urllib.request.quote(name)
        +         "&gsrnamespace=6&gsrlimit=8&prop=imageinfo"
        "&iiprop=url|size|mime&iiurlwidth=500&format=json"
    )
    data = _get_json(url)
    pages = data.get("query", {}).get("pages", {})
    out = []
    for page in pages.values():
        for info in page.get("imageinfo", []):
            if info.get("mime") in ("image/jpeg", "image/png"):
                out.append(info.get("thumburl") or info.get("url"))
    return out[:4]


def main():
    actors = json.loads((REF_DIR / "actors.json").read_text())
    index_path = REF_DIR / "photos" / "index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
    have = {k.rsplit("/", 1)[0] for k in index}
    for group, names in actors.items():
        for name in names:
            dest_dir = PHOTO_DIR / group / name.replace("/", "-")
            if str(dest_dir) in have:
                continue
            dest_dir.mkdir(parents=True, exist_ok=True)
            urls = []
            try:
                page_img = _wiki_page_image(name)
                if page_img:
                    urls.append(page_img)
            except Exception:
                page_img = None
            try:
                urls.extend(_commons_images(name))
            except Exception:
                pass
            if not urls:
                for suffix in (" (actor)", " (actress)"):
                    try:
                        page_img = _wiki_page_image(name + suffix)
                        if page_img:
                            urls.append(page_img)
                            break
                    except Exception:
                        continue
            n = 0
            for src in urls:
                time.sleep(SLEEP)
                try:
                    data = _get_bytes(src)
                except Exception:
                    continue
                dims = _image_dims(data)
                if dims is None or dims[0] < 300:
                    continue
                n += 1
                ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
                path = dest_dir / f"{n}{ext}"
                path.write_bytes(data)
                index[str(path)] = src
            (REF_DIR / "photos" / "index.json").write_text(json.dumps(index, indent=1))
    print(f"fetched {len(index)} photos")


if __name__ == "__main__":
    main()
