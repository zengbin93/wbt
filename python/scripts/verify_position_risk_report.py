"""Optional browser regression check for the offline position-risk report."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/benchmarks/position_risk/report.html"))
    parser.add_argument("--screenshot", type=Path, default=Path("target/position-risk-report.png"))
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1500})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.report.resolve().as_uri())
        assert page.title() == "WBT · Position Risk 性能报告"
        assert page.locator("section.scroll tr").count() > 1
        assert page.locator("body").evaluate("(node) => node.scrollWidth") == 1500
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(args.screenshot))
        page.locator("details summary").first.click()
        assert page.locator("details").first.get_attribute("open") is not None
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.locator("body").evaluate("(node) => node.scrollWidth") == 390
        assert not errors, errors
        browser.close()
    print("HTML verified: result rows, desktop/mobile width, expandable phase table, no page errors.")


if __name__ == "__main__":
    main()
