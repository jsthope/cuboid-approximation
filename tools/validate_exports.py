"""Validate a GLB with Khronos and exercise the offline viewer in Chromium."""

import argparse
import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    output = args.run.resolve() / "textured"
    script = """
const fs = require('fs');
const validator = require('gltf-validator');
validator.validateBytes(new Uint8Array(fs.readFileSync(process.argv[1])))
  .then(r => { console.log(JSON.stringify(r)); process.exitCode = r.issues.numErrors ? 1 : 0; })
  .catch(e => { console.error(e); process.exitCode = 1; });
"""
    result = subprocess.run(
        ["node", "-e", script, str(output / "cuboids_textured.glb")],
        check=True,
        capture_output=True,
        text=True,
    )
    validation = json.loads(result.stdout)
    (args.run / "gltf-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport=dict(width=1280, height=800))
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: (
                errors.append(message.text)
                if "GL_INVALID" in message.text or message.type == "error"
                else None
            ),
        )
        page.goto((output / "cuboids_textured_3d.html").as_uri())
        page.wait_for_function("window.viewerState && window.viewerState.ready")
        assert page.evaluate("window.viewerState.vertices") == 0
        assert page.evaluate("window.viewerState.glError") == 0
        count = int(page.locator("#step").get_attribute("max"))
        ends = page.evaluate("data.box_vertex_ends || Array.from({length:data.report.cuboids+1},(_,i)=>i*36)")
        assert len(ends) == count + 1
        for prefix in range(1, count + 1):
            page.locator("#step").fill(str(prefix))
            page.locator("#step").dispatch_event("input")
            page.wait_for_function("expected => window.viewerState.vertices === expected", arg=ends[prefix])
            assert page.evaluate("window.viewerState.glError") == 0
        for mode in ("texture", "cloud", "overlay", "confidence", "compare"):
            page.select_option("#mode", mode)
            page.wait_for_function("mode => window.viewerState.mode === mode", arg=mode)
            assert page.evaluate("window.viewerState.glError") == 0
        assert page.evaluate("window.viewerState.vertices") == ends[-1]
        for button in ("#front", "#back", "#reset"):
            page.click(button)
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.screenshot))
        assert not errors, errors
        browser.close()
    print(
        f"GLB: {validation['issues']['numErrors']} errors, {validation['issues']['numWarnings']} warnings; viewer controls passed."
    )


if __name__ == "__main__":
    main()
