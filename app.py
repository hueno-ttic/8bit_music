"""ブラウザから mp4 をアップロードして 8bit 化した曲をダウンロードする Web アプリ.

起動:  ./run.sh   (または  .venv/bin/uvicorn app:app --port 35607)
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from chiptune import ConvertOptions, convert_file
from chiptune import analysis_ml, separation
from chiptune.youtube import download_audio, is_url

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
MAX_UPLOAD_MB = 300

app = FastAPI(title="8bit Music Converter")
# GitHub Pages など別オリジンで配信した UI から、手元のこのサーバーを呼べるようにする
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Verify", "Content-Disposition"])
if (ROOT / "samples").is_dir():
    app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")

# 変換ジョブの状態 (単一プロセス前提の簡易版)
JOBS: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/info")
def info() -> dict:
    return {"ml_available": analysis_ml.is_available(), "sep_available": separation.is_available(),
            "max_upload_mb": MAX_UPLOAD_MB}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in job.items() if k not in ("path", "dir")}


def _parse_opts(mode: str, duty: float, arpeggio: bool, arp_speed: int, drums: bool, bass: bool,
                crunch: bool, fmt: str, sensitivity: float = 0.5, tempo_mult: str = "auto",
                fill_gaps: bool = True, key_snap: bool = True, echo: bool = True,
                melody_octave: int = 0, quantize: bool = True, trim_intro: bool = True) -> ConvertOptions:
    if fmt not in ("wav", "mp3", "midi"):
        raise HTTPException(400, "fmt は wav / mp3 / midi")
    if tempo_mult not in ("auto", "1", "2"):
        raise HTTPException(400, "tempo_mult は auto / 1 / 2")
    if not 0.2 <= sensitivity <= 0.8 or melody_octave not in (-1, 0, 1):
        raise HTTPException(400, "sensitivity / melody_octave の範囲が不正です")
    if mode not in ("auto", "sep", "ml", "dsp"):
        raise HTTPException(400, "mode が不正です")
    if duty not in (0.125, 0.25, 0.5):
        raise HTTPException(400, "duty は 0.125 / 0.25 / 0.5")
    return ConvertOptions(mode=mode, melody_duty=duty, arpeggio=arpeggio, arp_speed=arp_speed,
                          drums=drums, bass=bass, crunch=crunch, fmt=fmt,
                          sensitivity=sensitivity, tempo_mult=tempo_mult if tempo_mult == "auto" else int(tempo_mult),
                          fill_gaps=fill_gaps, key_snap=key_snap, echo=echo, melody_octave=melody_octave, quantize=quantize, trim_intro=trim_intro)


async def _run_job(job_id: str, work: Path, src: Path, stem: str, opts: ConvertOptions,
                   background: BackgroundTasks):
    """変換を実行してダウンロード用レスポンスを返す (アップロード / URL 共通)."""
    def progress(msg: str):
        JOBS[job_id]["message"] = msg

    dst = work / f"{stem}_8bit.{opts.fmt}"
    try:
        out = await run_in_threadpool(convert_file, src, dst, opts, progress)
    except Exception as e:  # noqa: BLE001
        log.exception("conversion failed")
        JOBS[job_id] = {"status": "error", "message": str(e)}
        shutil.rmtree(work, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=500)

    JOBS[job_id]["status"] = "done"
    media = {"mp3": "audio/mpeg", "midi": "audio/midi"}.get(opts.fmt, "audio/wav")
    from chiptune import convert as _cv
    vr = _cv.LAST_VERIFY_REPORT
    headers = {}
    if vr:
        # 自己検証の結果をヘッダで UI に渡す (ASCII のみ)
        headers["X-Verify"] = json.dumps({k: vr[k] for k in ("iterations", "passed", "rate", "n_bad", "n_checked", "target", "max_iter", "history")},
                                         ensure_ascii=True)
        headers["Access-Control-Expose-Headers"] = "X-Verify"

    def cleanup():
        shutil.rmtree(work, ignore_errors=True)
        JOBS.pop(job_id, None)

    background.add_task(cleanup)
    return FileResponse(out, media_type=media, filename=out.name, background=background, headers=headers)


@app.post("/api/convert_url")
async def convert_url(
    background: BackgroundTasks,
    url: str = Form(...),
    mode: str = Form("auto"),
    duty: float = Form(0.5),
    arpeggio: bool = Form(True),
    arp_speed: int = Form(1),
    drums: bool = Form(True),
    bass: bool = Form(True),
    crunch: bool = Form(True),
    fmt: str = Form("wav"),
    job_id: str = Form(""),
    sensitivity: float = Form(0.5),
    tempo_mult: str = Form("auto"),
    fill_gaps: bool = Form(False),
    key_snap: bool = Form(True),
    echo: bool = Form(False),
    melody_octave: int = Form(0),
    quantize: bool = Form(True),
    trim_intro: bool = Form(True),
):
    opts = _parse_opts(mode, duty, arpeggio, arp_speed, drums, bass, crunch, fmt,
                       sensitivity, tempo_mult, fill_gaps, key_snap, echo, melody_octave, quantize, trim_intro)
    if not is_url(url):
        raise HTTPException(400, "URL の形式が正しくありません (https://... で始めてください)")
    job_id = job_id or uuid.uuid4().hex
    work = Path(tempfile.mkdtemp(prefix="8bit_"))
    JOBS[job_id] = {"status": "running", "message": "YouTube からダウンロード中…", "dir": str(work)}

    def progress(msg: str):
        JOBS[job_id]["message"] = msg

    try:
        src, title = await run_in_threadpool(download_audio, url, work, progress)
    except Exception as e:  # noqa: BLE001
        log.exception("download failed")
        JOBS[job_id] = {"status": "error", "message": str(e)}
        shutil.rmtree(work, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=500)
    opts.source_url = url.strip()
    return await _run_job(job_id, work, src, title, opts, background)


@app.post("/api/convert")
async def convert(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    duty: float = Form(0.5),
    arpeggio: bool = Form(True),
    arp_speed: int = Form(1),
    drums: bool = Form(True),
    bass: bool = Form(True),
    crunch: bool = Form(True),
    fmt: str = Form("wav"),
    job_id: str = Form(""),
    sensitivity: float = Form(0.5),
    tempo_mult: str = Form("auto"),
    fill_gaps: bool = Form(False),
    key_snap: bool = Form(True),
    echo: bool = Form(False),
    melody_octave: int = Form(0),
    quantize: bool = Form(True),
    trim_intro: bool = Form(True),
):
    opts = _parse_opts(mode, duty, arpeggio, arp_speed, drums, bass, crunch, fmt,
                       sensitivity, tempo_mult, fill_gaps, key_snap, echo, melody_octave, quantize, trim_intro)
    job_id = job_id or uuid.uuid4().hex
    work = Path(tempfile.mkdtemp(prefix="8bit_"))
    src_name = Path(file.filename or "input.mp4").name
    src = work / src_name
    size = 0
    with src.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                shutil.rmtree(work, ignore_errors=True)
                raise HTTPException(413, f"ファイルが大きすぎます (上限 {MAX_UPLOAD_MB}MB)")
            f.write(chunk)

    JOBS[job_id] = {"status": "running", "message": "開始", "dir": str(work)}
    return await _run_job(job_id, work, src, Path(src_name).stem, opts, background)
