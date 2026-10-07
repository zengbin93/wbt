"""Cache the report's existing CDN dependencies for offline browser regression."""

import base64
import re
import sys
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

target = Path(sys.argv[1])
target.mkdir(parents=True, exist_ok=True)
resources = {
    "fonts.css": "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap",
    "bootstrap.css": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
    "icons.css": "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css",
    "bootstrap.js": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js",
}
cache = {}


def fetch(url):
    if url not in cache:
        with urlopen(url, timeout=60) as response:
            cache[url] = response.read()
    return cache[url]


for name, url in resources.items():
    content = fetch(url).decode()
    if name.endswith(".css"):
        for asset in re.findall(r'url\([\'\"]?([^\)\'\"]+)', content):
            asset_url = urljoin(url, asset)
            mime = "font/woff2" if ".woff2" in asset_url else "font/woff"
            content = content.replace(asset, f"data:{mime};base64," + base64.b64encode(fetch(asset_url)).decode())
    (target / name).write_text(content)
