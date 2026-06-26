#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote


DEFAULT_RUN_DIR = Path.home() / "data" / "benchmarks" / "waybill_ocr" / "run_full_640x480"
DEFAULT_MANIFEST = DEFAULT_RUN_DIR / "current_manifest.json"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "review"


def json_dumps(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True)


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise ValueError(f"manifest must contain a samples list: {path}")
    return data


def clean_ground_truth(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("destination_floor", "destination_room", "destination_dong"):
        raw = value.get(key)
        if raw not in (None, ""):
            result[key] = str(raw)
    return result


def relative_file_url(path_value: Any, *, base_dir: Path) -> str | None:
    if not path_value:
        return None
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    try:
        relative = Path(os.path.relpath(path, start=base_dir.resolve())).as_posix()
    except ValueError:
        relative = path.as_posix()
    return quote(relative, safe="/:@._-+()[]")


def review_item(sample: dict[str, Any], *, index: int, out_dir: Path) -> dict[str, Any]:
    ground_truth = clean_ground_truth(sample.get("ground_truth"))
    image_path = sample.get("image_path")
    original_image_path = sample.get("original_image_path")
    return {
        "index": index,
        "sample_id": str(sample.get("sample_id") or ""),
        "carrier": sample.get("carrier"),
        "condition": sample.get("condition"),
        "benchmark_status": sample.get("benchmark_status"),
        "annotation_source": sample.get("benchmark_annotation_source"),
        "has_ground_truth": bool(ground_truth),
        "ground_truth": ground_truth,
        "image_path": str(image_path or ""),
        "original_image_path": str(original_image_path or ""),
        "image_url": relative_file_url(image_path, base_dir=out_dir),
        "original_image_url": relative_file_url(original_image_path, base_dir=out_dir),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "sample_id",
        "carrier",
        "condition",
        "benchmark_status",
        "annotation_source",
        "destination_floor",
        "destination_room",
        "destination_dong",
        "image_path",
        "original_image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            truth = row.get("ground_truth") if isinstance(row.get("ground_truth"), dict) else {}
            writer.writerow(
                {
                    "index": row.get("index"),
                    "sample_id": row.get("sample_id"),
                    "carrier": row.get("carrier"),
                    "condition": row.get("condition"),
                    "benchmark_status": row.get("benchmark_status"),
                    "annotation_source": row.get("annotation_source"),
                    "destination_floor": truth.get("destination_floor"),
                    "destination_room": truth.get("destination_room"),
                    "destination_dong": truth.get("destination_dong"),
                    "image_path": row.get("image_path"),
                    "original_image_path": row.get("original_image_path"),
                }
            )


def embedded_json(payload: Any) -> str:
    return (
        json_dumps(payload)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("</", "<\\/")
    )


def build_html(*, title: str, manifest_path: Path, items: list[dict[str, Any]]) -> str:
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "title": title,
        "generated_at": generated_at,
        "manifest_path": str(manifest_path),
        "items": items,
    }
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #697586;
      --line: #d7dde5;
      --accent: #0b6bcb;
      --warn: #b54708;
      --ok: #067647;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      min-height: 100vh;
    }}
    .viewer {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      min-width: 0;
      border-right: 1px solid var(--line);
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    .title {{
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    h1 {{
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      font-weight: 700;
    }}
    .sub {{
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 72vw;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    button, select, input, textarea {{
      font: inherit;
    }}
    button {{
      min-height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    button.primary {{
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }}
    button:disabled {{
      opacity: 0.45;
      cursor: default;
    }}
    .image-wrap {{
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }}
    .image-frame {{
      width: min(100%, 1120px);
      height: min(78vh, calc(100vh - 96px));
      min-height: 420px;
      background: #111827;
      border: 1px solid #111827;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }}
    aside {{
      min-width: 0;
      background: var(--panel);
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      height: 100vh;
    }}
    .side-head, .side-foot {{
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }}
    .side-foot {{
      border-top: 1px solid var(--line);
      border-bottom: 0;
    }}
    .meta {{
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr);
      gap: 8px 10px;
      align-items: start;
      margin-top: 10px;
    }}
    .meta dt {{
      color: var(--muted);
      font-size: 12px;
    }}
    .meta dd {{
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .gt-current {{
      display: grid;
      gap: 4px;
      margin-top: 12px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
    }}
    .gt-current span {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .gt-current strong {{
      font-size: 22px;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }}
    .body {{
      min-height: 0;
      overflow: auto;
      padding: 14px;
    }}
    .fields {{
      display: grid;
      gap: 10px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    input, select, textarea {{
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      color: var(--ink);
      background: #fff;
      font-size: 14px;
    }}
    textarea {{
      resize: vertical;
      min-height: 72px;
      line-height: 1.4;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .status {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 999px;
      padding: 2px 9px;
      background: #eef4ff;
      color: #1849a9;
      font-size: 12px;
      font-weight: 700;
    }}
    .status.edited {{
      background: #ecfdf3;
      color: var(--ok);
    }}
    .status.warn {{
      background: #fff6ed;
      color: var(--warn);
    }}
    .list {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(42px, 1fr));
      gap: 5px;
      margin-top: 12px;
    }}
    .list button {{
      min-height: 28px;
      padding: 3px 0;
      border-radius: 5px;
      font-size: 12px;
    }}
    .list button.active {{
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
    }}
    .list button.edited {{
      border-color: var(--ok);
      color: var(--ok);
    }}
    .actions {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .actions button {{
      flex: 1 1 120px;
    }}
    @media (max-width: 900px) {{
      .app {{
        grid-template-columns: 1fr;
      }}
      .viewer {{
        border-right: 0;
      }}
      aside {{
        height: auto;
        min-height: 60vh;
      }}
      .image-frame {{
        height: 58vh;
        min-height: 300px;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <main class="viewer">
      <header>
        <div class="title">
          <h1 id="title">{html.escape(title)}</h1>
          <div class="sub" id="path"></div>
        </div>
        <div class="controls">
          <button id="prev" type="button" title="previous">&lt;</button>
          <strong id="counter">0 / 0</strong>
          <button id="next" type="button" title="next">&gt;</button>
          <button id="toggleImage" type="button">원본</button>
        </div>
      </header>
      <section class="image-wrap">
        <div class="image-frame">
          <img id="image" alt="">
        </div>
      </section>
    </main>
    <aside>
      <section class="side-head">
        <div class="controls">
          <select id="filter">
            <option value="all">전체</option>
            <option value="manual_jsonl">수동 GT</option>
            <option value="source_manifest">원본 GT</option>
            <option value="edited">수정됨</option>
          </select>
          <span class="status" id="editState">saved</span>
        </div>
        <dl class="meta">
          <dt>sample</dt><dd id="sampleId"></dd>
          <dt>carrier</dt><dd id="carrier"></dd>
          <dt>condition</dt><dd id="condition"></dd>
          <dt>source</dt><dd id="source"></dd>
          <dt>status</dt><dd id="benchStatus"></dd>
        </dl>
        <div class="gt-current">
          <span>현재 GT</span>
          <strong id="gtNow">-</strong>
        </div>
        <div class="list" id="sampleList"></div>
      </section>
      <section class="body">
        <div class="fields">
          <label>review status
            <select id="status">
              <option value="verified">verified</option>
              <option value="low_confidence">low_confidence</option>
              <option value="unreadable">unreadable</option>
              <option value="no_destination_detail">no_destination_detail</option>
              <option value="exclude">exclude</option>
            </select>
          </label>
          <div class="row">
            <label>floor
              <input id="floor" autocomplete="off">
            </label>
            <label>dong
              <input id="dong" autocomplete="off">
            </label>
          </div>
          <label>room
            <input id="room" autocomplete="off">
          </label>
          <label>note
            <textarea id="note"></textarea>
          </label>
        </div>
      </section>
      <section class="side-foot">
        <div class="actions">
          <button id="save" type="button" class="primary">저장</button>
          <button id="reset" type="button">되돌림</button>
          <button id="export" type="button">export</button>
        </div>
      </section>
    </aside>
  </div>
  <script id="review-data" type="application/json">{embedded_json(payload)}</script>
  <script>
    const payload = JSON.parse(document.getElementById("review-data").textContent);
    const items = payload.items;
    const storeKey = "indory_ocr_review:" + payload.manifest_path;
    let edits = JSON.parse(localStorage.getItem(storeKey) || "{{}}");
    let current = 0;
    let filter = "all";
    let showingOriginal = false;
    let lastSaveWasRemote = false;

    const $ = (id) => document.getElementById(id);

    function formatDestination(gt) {{
      const parts = [];
      if (gt.destination_dong) parts.push(gt.destination_dong);
      if (gt.destination_floor) parts.push(gt.destination_floor);
      if (gt.destination_room) parts.push(gt.destination_room);
      return parts.length ? parts.join(" ") : "-";
    }}

    function baseRecord(item) {{
      const gt = item.ground_truth || {{}};
      return {{
        sample_id: item.sample_id,
        status: item.benchmark_status || "verified",
        ground_truth: {{
          destination_floor: gt.destination_floor || "",
          destination_room: gt.destination_room || "",
          destination_dong: gt.destination_dong || ""
        }},
        review_note: ""
      }};
    }}

    function activeRecord(item) {{
      return edits[item.sample_id] || baseRecord(item);
    }}

    function filteredIndexes() {{
      return items.map((item, index) => [item, index]).filter(([item]) => {{
        if (filter === "all") return true;
        if (filter === "edited") return Boolean(edits[item.sample_id]);
        return item.annotation_source === filter;
      }}).map(([, index]) => index);
    }}

    function updateList() {{
      const indexes = filteredIndexes();
      const list = $("sampleList");
      list.innerHTML = "";
      indexes.forEach((index) => {{
        const item = items[index];
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = String(index + 1);
        button.className = [
          index === current ? "active" : "",
          edits[item.sample_id] ? "edited" : ""
        ].filter(Boolean).join(" ");
        button.addEventListener("click", () => show(index));
        list.appendChild(button);
      }});
    }}

    function show(index) {{
      if (!items.length) return;
      current = Math.max(0, Math.min(index, items.length - 1));
      const item = items[current];
      const record = activeRecord(item);
      const gt = record.ground_truth || {{}};
      const imageUrl = showingOriginal && item.original_image_url ? item.original_image_url : item.image_url;

      $("image").src = imageUrl || "";
      $("image").alt = item.sample_id;
      $("path").textContent = showingOriginal && item.original_image_path ? item.original_image_path : item.image_path;
      $("counter").textContent = `${{current + 1}} / ${{items.length}}`;
      $("prev").disabled = current === 0;
      $("next").disabled = current === items.length - 1;
      $("toggleImage").disabled = !item.original_image_url;
      $("toggleImage").textContent = showingOriginal ? "리사이즈" : "원본";
      $("sampleId").textContent = item.sample_id || "-";
      $("carrier").textContent = item.carrier || "-";
      $("condition").textContent = item.condition || "-";
      $("source").textContent = item.annotation_source || "-";
      $("benchStatus").textContent = item.benchmark_status || "-";
      $("status").value = record.status || "verified";
      $("floor").value = gt.destination_floor || "";
      $("room").value = gt.destination_room || "";
      $("dong").value = gt.destination_dong || "";
      $("note").value = record.review_note || "";
      $("gtNow").textContent = formatDestination(gt);

      const edited = Boolean(edits[item.sample_id]);
      $("editState").textContent = edited ? "edited" : "saved";
      $("editState").className = edited ? "status edited" : "status";
      if (record.status !== "verified") $("editState").className = "status warn";
      updateList();
    }}

    function currentFormRecord() {{
      const item = items[current];
      const groundTruth = {{
        destination_floor: $("floor").value.trim(),
        destination_room: $("room").value.trim(),
        destination_dong: $("dong").value.trim()
      }};
      Object.keys(groundTruth).forEach((key) => {{
        if (!groundTruth[key]) delete groundTruth[key];
      }});
      return {{
        sample_id: item.sample_id,
        status: $("status").value,
        ground_truth: groundTruth,
        review_note: $("note").value.trim(),
        reviewed_at: new Date().toISOString()
      }};
    }}

    function applyRecordToItem(record) {{
      const item = items[current];
      item.ground_truth = record.ground_truth || {{}};
      item.benchmark_status = record.status || "verified";
      item.annotation_source = "manual_jsonl";
      item.has_ground_truth = Boolean(Object.keys(item.ground_truth).length);
      delete edits[item.sample_id];
      localStorage.setItem(storeKey, JSON.stringify(edits));
    }}

    async function saveCurrent() {{
      const item = items[current];
      const record = currentFormRecord();
      try {{
        const response = await fetch("/api/review/save", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify(record)
        }});
        if (!response.ok) {{
          const errorText = await response.text();
          throw new Error(errorText || `HTTP ${{response.status}}`);
        }}
        applyRecordToItem(record);
        show(current);
        $("editState").textContent = "file saved";
        $("editState").className = "status edited";
        lastSaveWasRemote = true;
        return;
      }} catch (error) {{
        lastSaveWasRemote = false;
      }}
      edits[item.sample_id] = record;
      localStorage.setItem(storeKey, JSON.stringify(edits));
      show(current);
      $("editState").textContent = "browser saved";
      $("editState").className = "status warn";
    }}

    function refreshCurrentPreview() {{
      const record = currentFormRecord();
      $("gtNow").textContent = formatDestination(record.ground_truth || {{}});
      if (!lastSaveWasRemote) {{
        $("editState").textContent = "editing";
        $("editState").className = "status warn";
      }}
    }}

    function resetCurrent() {{
      const item = items[current];
      delete edits[item.sample_id];
      localStorage.setItem(storeKey, JSON.stringify(edits));
      show(current);
    }}

    function exportEdits() {{
      const rows = Object.values(edits).map((row) => JSON.stringify(row)).join("\\n");
      const blob = new Blob([rows ? rows + "\\n" : ""], {{type: "application/x-jsonlines;charset=utf-8"}});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "review_corrections.jsonl";
      link.click();
      URL.revokeObjectURL(link.href);
    }}

    function move(delta) {{
      const indexes = filteredIndexes();
      if (!indexes.length) return;
      const position = indexes.indexOf(current);
      const nextPosition = position >= 0 ? position + delta : 0;
      const bounded = Math.max(0, Math.min(nextPosition, indexes.length - 1));
      show(indexes[bounded]);
    }}

    $("prev").addEventListener("click", () => move(-1));
    $("next").addEventListener("click", () => move(1));
    $("save").addEventListener("click", saveCurrent);
    $("reset").addEventListener("click", resetCurrent);
    $("export").addEventListener("click", exportEdits);
    ["status", "floor", "room", "dong", "note"].forEach((id) => {{
      $(id).addEventListener("input", () => {{
        lastSaveWasRemote = false;
        refreshCurrentPreview();
      }});
      $(id).addEventListener("change", () => {{
        lastSaveWasRemote = false;
        refreshCurrentPreview();
      }});
    }});
    $("toggleImage").addEventListener("click", () => {{
      showingOriginal = !showingOriginal;
      show(current);
    }});
    $("filter").addEventListener("change", (event) => {{
      filter = event.target.value;
      const indexes = filteredIndexes();
      show(indexes.length ? indexes[0] : 0);
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
      if (event.key === "s") saveCurrent();
    }});

    show(0);
  </script>
</body>
</html>
"""


def build_review(*, manifest_path: Path, out_dir: Path, title: str) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    samples = [sample for sample in manifest["samples"] if isinstance(sample, dict)]
    items = [review_item(sample, index=index + 1, out_dir=out_dir) for index, sample in enumerate(samples)]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(
        build_html(title=title, manifest_path=manifest_path, items=items),
        encoding="utf-8",
    )
    (out_dir / "review_items.json").write_text(json_dumps({"items": items}, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out_dir / "review_items.jsonl", items)
    write_csv(out_dir / "review_items.csv", items)

    summary = {
        "manifest": str(manifest_path),
        "out_dir": str(out_dir),
        "html": str(out_dir / "index.html"),
        "item_count": len(items),
        "ground_truth_count": sum(1 for item in items if item.get("has_ground_truth")),
        "manual_jsonl_count": sum(1 for item in items if item.get("annotation_source") == "manual_jsonl"),
        "source_manifest_count": sum(1 for item in items if item.get("annotation_source") == "source_manifest"),
    }
    (out_dir / "review_summary.json").write_text(json_dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML reviewer for waybill benchmark GT.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--title", default="Indory Waybill GT Review")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_review(
        manifest_path=args.manifest,
        out_dir=args.out_dir,
        title=args.title,
    )
    print(json_dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
