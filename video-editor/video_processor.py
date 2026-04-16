"""
video_processor.py
Core video processing engine:
  - Silence detection + jump cuts via FFmpeg
  - Transcription via OpenAI Whisper
  - ASS subtitle generation (Saykin-style bold center captions)
  - Dynamic zoom-in effects on sentence starts
"""

import os
import re
import json
import subprocess
import tempfile
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _probe(path: str) -> Dict:
    """Return ffprobe JSON for a file."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _get_duration(path: str) -> float:
    info = _probe(path)
    return float(info["format"]["duration"])


def _get_video_info(path: str) -> Tuple[int, int, float]:
    """Returns (width, height, fps)."""
    info = _probe(path)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    num, den = vs["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    return w, h, fps


# ──────────────────────────────────────────────
# Step 1 – Silence Detection & Jump Cuts
# ──────────────────────────────────────────────

def detect_silences(
    input_path: str,
    noise_db: float = -35,
    min_silence_s: float = 0.4,
) -> List[Dict]:
    """
    Returns list of {'start': float, 'end': float} silent intervals.
    """
    cmd = [
        "ffmpeg", "-i", input_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr

    silences: List[Dict] = []
    start: Optional[float] = None

    for line in stderr.splitlines():
        if "silence_start" in line:
            m = re.search(r"silence_start:\s*([\d.]+)", line)
            if m:
                start = float(m.group(1))
        elif "silence_end" in line:
            m = re.search(r"silence_end:\s*([\d.]+)", line)
            if m and start is not None:
                silences.append({"start": start, "end": float(m.group(1))})
                start = None

    return silences


def remove_silences(
    input_path: str,
    output_path: str,
    silences: List[Dict],
    progress_cb=None,
) -> str:
    """
    Cuts silent gaps and concatenates the speech segments.
    Returns output_path.
    """
    if not silences:
        # Nothing to cut — just copy
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-c", "copy", "-y", output_path],
            check=True, capture_output=True,
        )
        return output_path

    duration = _get_duration(input_path)

    # Build list of "keep" intervals
    segments: List[Tuple[float, float]] = []
    prev_end = 0.0
    for s in silences:
        if s["start"] > prev_end + 0.05:
            segments.append((prev_end, s["start"]))
        prev_end = s["end"]
    if prev_end < duration - 0.05:
        segments.append((prev_end, duration))

    if not segments:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-c", "copy", "-y", output_path],
            check=True, capture_output=True,
        )
        return output_path

    # Build filter_complex
    n = len(segments)
    filter_parts = []
    for i, (s, e) in enumerate(segments):
        filter_parts.append(
            f"[0:v]trim=start={s:.4f}:end={e:.4f},setpts=PTS-STARTPTS[v{i}];"
        )
        filter_parts.append(
            f"[0:a]atrim=start={s:.4f}:end={e:.4f},asetpts=PTS-STARTPTS[a{i}];"
        )

    vconcat = "".join(f"[v{i}]" for i in range(n))
    aconcat = "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(
        f"{vconcat}{aconcat}concat=n={n}:v=1:a=1[vout][aout]"
    )

    filter_complex = "".join(filter_parts)

    if progress_cb:
        progress_cb(30, "Applying jump cuts...")

    cmd = [
        "ffmpeg", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-y", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg error: %s", result.stderr[-2000:])
        raise RuntimeError(f"Jump cut failed: {result.stderr[-500:]}")

    return output_path


# ──────────────────────────────────────────────
# Step 2 – Whisper Transcription
# ──────────────────────────────────────────────

def transcribe(input_path: str, api_key: str) -> Dict:
    """
    Transcribes audio using OpenAI Whisper API (cloud).
    Returns the full Whisper response dict.
    """
    import openai
    client = openai.OpenAI(api_key=api_key)

    with open(input_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

    # Convert to dict
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


# ──────────────────────────────────────────────
# Step 3 – ASS Subtitle Generation (Saykin Style)
# ──────────────────────────────────────────────

def _seconds_to_ass(t: float) -> str:
    """Convert float seconds → ASS timestamp h:mm:ss.cc"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int((t - int(t)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _chunk_words(words: List[Dict], chunk_size: int = 3) -> List[Dict]:
    """Group words into small on-screen chunks."""
    chunks = []
    for i in range(0, len(words), chunk_size):
        group = words[i : i + chunk_size]
        text = " ".join(w.get("word", "").strip() for w in group).upper()
        chunks.append(
            {
                "text": text,
                "start": group[0].get("start", 0),
                "end": group[-1].get("end", group[0].get("start", 0) + 1),
            }
        )
    return chunks


def build_ass_subtitles(transcript: Dict, output_path: str) -> str:
    """
    Generate a Saykin-style ASS subtitle file:
      - Bold, large white text with black outline
      - Centered horizontally AND vertically
      - Yellow highlights on key words (every chunk alternates)
    """
    header = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,88,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,5,2,5,80,80,80,1
Style: Highlight,Arial Black,88,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,5,2,5,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    words = []
    # Try word-level timestamps first
    if "words" in transcript:
        words = transcript["words"]
    else:
        for seg in transcript.get("segments", []):
            words.extend(seg.get("words", []))

    if words:
        chunks = _chunk_words(words, chunk_size=3)
    else:
        # Fallback to segments
        chunks = []
        for seg in transcript.get("segments", []):
            chunks.append(
                {
                    "text": seg.get("text", "").strip().upper(),
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )

    for i, chunk in enumerate(chunks):
        style = "Highlight" if i % 3 == 0 else "Default"
        start_ts = _seconds_to_ass(chunk["start"])
        end_ts = _seconds_to_ass(chunk["end"])
        text = chunk["text"].replace("\n", "\\N")
        events.append(
            f"Dialogue: 0,{start_ts},{end_ts},{style},,0,0,0,,{text}"
        )

    ass_content = header + "\n".join(events) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    return output_path


def burn_subtitles(
    input_path: str,
    ass_path: str,
    output_path: str,
    progress_cb=None,
) -> str:
    """Burn ASS subtitles into the video."""
    if progress_cb:
        progress_cb(65, "Burning captions onto video...")

    # Escape path for ffmpeg filter
    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", f"ass={escaped_ass}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        "-y", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Subtitle burn error: %s", result.stderr[-2000:])
        raise RuntimeError(f"Subtitle burn failed: {result.stderr[-500:]}")

    return output_path


# ──────────────────────────────────────────────
# Step 4 – Dynamic Zoom Effect
# ──────────────────────────────────────────────

def apply_dynamic_zooms(
    input_path: str,
    output_path: str,
    sentence_starts: List[float],
    progress_cb=None,
) -> str:
    """
    Apply a quick snap-zoom-in at each sentence start timestamp.
    Uses FFmpeg zoompan for smooth, GPU-friendly zoom.
    """
    if progress_cb:
        progress_cb(80, "Applying dynamic zooms...")

    w, h, fps = _get_video_info(input_path)
    fps_int = max(1, round(fps))

    # Build a zoom expression that punches in (1.0→1.08) over 12 frames
    # at each sentence start, then eases back.
    # We keep it simple: zoompan with a piecewise expression.

    # For robustness, limit to first 15 sentence starts
    starts = sorted(sentence_starts)[:15]

    if not starts:
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-c", "copy", "-y", output_path],
            check=True, capture_output=True,
        )
        return output_path

    # Build zoom expression using if() chains
    # Each event: zoom punches from 1 to 1.08 over punch_frames frames,
    # then holds at 1.02 for hold_frames, then snaps back.
    punch_frames = 8
    hold_frames = fps_int * 2  # 2 seconds zoomed in

    conditions = []
    for t in starts:
        frame_start = int(t * fps_int)
        frame_end = frame_start + punch_frames + hold_frames
        conditions.append(
            f"if(between(on,{frame_start},{frame_start+punch_frames}),"
            f"1+0.08*(on-{frame_start})/{punch_frames},"
            f"if(between(on,{frame_start+punch_frames},{frame_end}),1.08,0))"
        )

    zoom_expr = "+".join(f"({c})" for c in conditions)
    zoom_expr = f"max(1,{zoom_expr})"

    filter_str = (
        f"zoompan=z='{zoom_expr}':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={w}x{h}:fps={fps_int}"
    )

    cmd = [
        "ffmpeg", "-i", input_path,
        "-vf", filter_str,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        "-y", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Zoom effect failed (falling back to no zoom): %s", result.stderr[-500:])
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-c", "copy", "-y", output_path],
            check=True, capture_output=True,
        )

    return output_path


# ──────────────────────────────────────────────
# Master Pipeline
# ──────────────────────────────────────────────

def process_video(
    input_path: str,
    job_dir: str,
    api_key: str,
    progress_cb=None,
) -> Dict:
    """
    Full pipeline:
      1. Silence removal / jump cuts
      2. Whisper transcription
      3. ASS caption generation + burn
      4. Dynamic zoom overlay
    Returns {'output_path': str, 'transcript': str, 'segments': list}
    """
    os.makedirs(job_dir, exist_ok=True)

    if progress_cb:
        progress_cb(5, "Detecting silences...")

    # ── 1. Silence removal
    silences = detect_silences(input_path)
    cut_path = os.path.join(job_dir, "cut.mp4")
    remove_silences(input_path, cut_path, silences, progress_cb)

    if progress_cb:
        progress_cb(40, "Transcribing with Whisper...")

    # ── 2. Transcription
    transcript_data = transcribe(cut_path, api_key)
    full_text = transcript_data.get("text", "")

    # ── 3. Captions
    if progress_cb:
        progress_cb(55, "Generating Saykin-style captions...")

    ass_path = os.path.join(job_dir, "captions.ass")
    build_ass_subtitles(transcript_data, ass_path)

    captioned_path = os.path.join(job_dir, "captioned.mp4")
    burn_subtitles(cut_path, ass_path, captioned_path, progress_cb)

    # ── 4. Dynamic zooms
    sentence_starts = [
        seg["start"]
        for seg in transcript_data.get("segments", [])
    ]

    zoomed_path = os.path.join(job_dir, "final.mp4")
    apply_dynamic_zooms(captioned_path, zoomed_path, sentence_starts, progress_cb)

    if progress_cb:
        progress_cb(95, "Finalizing...")

    return {
        "output_path": zoomed_path,
        "transcript": full_text,
        "segments": transcript_data.get("segments", []),
    }
