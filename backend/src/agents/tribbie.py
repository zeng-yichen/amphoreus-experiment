"""
Tribbie — Live interview companion.

Captures system audio via BlackHole virtual audio device, transcribes in
near-real-time with faster-whisper, and suggests follow-up questions using
Claude Haiku, loaded with client context from memory files and Aglaea briefings.

One-time BlackHole setup (Mac):
  1. brew install blackhole-2ch
  2. Reboot
  3. Open Audio MIDI Setup → Create Multi-Output Device
     (check BlackHole 2ch + your speakers, then set as system output)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from typing import Callable

import numpy as np

from backend.src.db import vortex as P

logger = logging.getLogger("tribbie")

# Module-level stop signals keyed by job_id
_sessions: dict[str, threading.Event] = {}

# Lazy-loaded Whisper model — loaded once on first session start
_whisper_model = None
_whisper_lock = threading.Lock()

# Audio constants
SAMPLE_RATE = 16_000           # Hz — faster-whisper expects 16kHz mono
CHUNK_SECONDS = 0.1            # sounddevice callback interval (100ms for fast silence detection)
SILENCE_THRESHOLD = 0.005      # RMS below this counts as silence
SILENCE_GAP_SECONDS = 0.4      # consecutive silence → flush & transcribe buffer
MAX_BUFFER_SECONDS = 8.0       # force-flush after this many seconds regardless
MIN_SEGMENT_WORDS = 2          # min words in a segment to qualify for a Haiku suggestion
SUGGESTION_COOLDOWN_SECONDS = 15  # minimum gap between Haiku calls
MAX_CONTEXT_CHARS = 12_000     # max chars of client context sent to Haiku


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel
                logger.info("[Tribbie] Loading faster-whisper 'base' model...")
                _whisper_model = WhisperModel("base", device="auto", compute_type="auto")
                logger.info("[Tribbie] Whisper model ready.")
    return _whisper_model


def _find_blackhole_device() -> int | None:
    """Return the sounddevice index of the BlackHole input device, or None."""
    import sounddevice as sd
    for i, dev in enumerate(sd.query_devices()):
        if "blackhole" in str(dev["name"]).lower() and dev["max_input_channels"] > 0:
            return i
    return None


def _load_context(company: str) -> str:
    """
    Load client context for Haiku: Aglaea briefing + content strategy + sample posts.
    Returns at most MAX_CONTEXT_CHARS characters.
    """
    parts: list[str] = []
    total = 0

    # 1. Aglaea briefing (most valuable — generated specifically for this interview)
    brief_file = P.brief_dir(company) / f"{company}_briefing.md"
    if brief_file.exists():
        text = brief_file.read_text(encoding="utf-8", errors="ignore")
        parts.append(f"## Interview Briefing\n{text}")
        total += len(text)

    # 2. Content strategy
    strat_dir = P.content_strategy_dir(company)
    if strat_dir.exists() and total < MAX_CONTEXT_CHARS:
        for f in sorted(strat_dir.iterdir()):
            if f.suffix in (".md", ".txt"):
                text = f.read_text(encoding="utf-8", errors="ignore")[:3_000]
                parts.append(f"## Content Strategy\n{text}")
                total += len(text)
                break  # one file is enough

    # 3. Sample accepted posts (voice/style awareness)
    acc_dir = P.accepted_dir(company)
    if acc_dir.exists() and total < MAX_CONTEXT_CHARS:
        for f in sorted(acc_dir.iterdir())[:2]:
            if f.suffix in (".md", ".txt"):
                text = f.read_text(encoding="utf-8", errors="ignore")[:1_500]
                parts.append(f"## Sample Accepted Post\n{text}")
                total += len(text)

    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def _suggest_followup(
    segment: str,
    context: str,
    transcript_so_far: str,
    anthropic_client,
) -> str | None:
    """Call Claude Haiku to suggest the single best follow-up question."""
    try:
        snippet = transcript_so_far[-3_000:]
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": (
                    "You are assisting a LinkedIn ghostwriter conducting a content interview. "
                    "Your job: suggest the single best question to ask right now.\n\n"
                    "Priority order:\n"
                    "1. If the client just shared something specific and interesting, probe deeper — "
                    "ask for the exact moment, the emotion, what happened next.\n"
                    "2. Otherwise, navigate toward a briefing question that hasn't been covered yet. "
                    "Check the transcript and avoid topics already discussed.\n"
                    "3. Never ask generic industry or product questions. Every question should target "
                    "a *specific personal story* — a failure, surprise, decision, or realization the "
                    "client experienced firsthand that could anchor a standalone LinkedIn post.\n\n"
                    f"<client_context>\n{context}\n</client_context>\n\n"
                    f"<transcript_so_far>\n{snippet}\n</transcript_so_far>\n\n"
                    f"<latest_utterance>\n{segment}\n</latest_utterance>\n\n"
                    "Reply with ONLY the question. No preamble, no explanation, no label."
                ),
            }],
        )
        return response.content[0].text.strip() if response.content else None
    except Exception as e:
        logger.warning("[Tribbie] Haiku suggestion failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_audio_devices() -> list[dict]:
    """Return all available audio input devices. Used by the health-check endpoint."""
    import sounddevice as sd
    result = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            result.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "is_blackhole": "blackhole" in str(dev["name"]).lower(),
            })
    return result


def start_session(company: str, job_id: str, event_callback: Callable) -> None:
    """
    Run the live interview capture loop.

    Called from a background thread via job_manager.run_in_background.
    Blocks until stop_session(job_id) is called or a fatal error occurs.

    event_callback(event_type: str, data: dict) — emits structured events.
    """
    import sounddevice as sd
    from anthropic import Anthropic

    stop_event = threading.Event()
    _sessions[job_id] = stop_event
    start_time = time.time()

    # --- Load client context ---
    context = _load_context(company)
    if not context:
        event_callback("status", {
            "message": "No client context found — run Aglaea first for richer suggestions.",
        })
    else:
        event_callback("status", {"message": "Client context loaded."})

    # --- Verify BlackHole device ---
    device_idx = _find_blackhole_device()
    if device_idx is None:
        event_callback("error", {
            "message": (
                "BlackHole audio device not found. "
                "Install: brew install blackhole-2ch, reboot, then open Audio MIDI Setup "
                "and create a Multi-Output Device combining BlackHole 2ch + your speakers."
            ),
        })
        _sessions.pop(job_id, None)
        return

    # --- Load Whisper model ---
    model = _get_whisper_model()
    anthropic_client = Anthropic()

    # --- Audio capture state ---
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    transcribe_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
    chunk_frames = int(SAMPLE_RATE * CHUNK_SECONDS)
    silence_threshold_chunks = int(SILENCE_GAP_SECONDS / CHUNK_SECONDS)
    max_work_chunks = int(MAX_BUFFER_SECONDS / CHUNK_SECONDS)

    segment_count = 0
    transcript_lines: list[str] = []
    transcribe_error: list[str] = []
    last_suggestion_time: float = 0.0

    def _audio_cb(indata: np.ndarray, frames: int, _time_info, _status) -> None:
        audio_q.put(indata.copy())

    # --- Transcription worker (separate thread so capture never stalls) ---
    def _transcribe_worker() -> None:
        while True:
            try:
                audio_array = transcribe_q.get(timeout=1.0)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            if audio_array is None:  # poison pill
                break

            nonlocal segment_count
            segment_count += 1
            event_callback("status", {"message": f"Transcribing…"})

            try:
                segs, _ = model.transcribe(
                    audio_array,
                    language="en",
                    beam_size=1,
                    vad_filter=True,
                )
                text = " ".join(s.text for s in segs).strip()
            except Exception as te:
                logger.warning("[Tribbie] Whisper error: %s", te)
                transcribe_error.append(str(te))
                continue

            if not text:
                event_callback("status", {"message": "Listening…"})
                continue

            transcript_lines.append(text)
            event_callback("text_delta", {"text": text})
            event_callback("status", {"message": "Listening…"})

            nonlocal last_suggestion_time
            now = time.time()
            if (
                len(text.split()) >= MIN_SEGMENT_WORDS
                and now - last_suggestion_time >= SUGGESTION_COOLDOWN_SECONDS
            ):
                last_suggestion_time = now
                transcript_so_far = "\n".join(transcript_lines)
                suggestion = _suggest_followup(text, context, transcript_so_far, anthropic_client)
                if suggestion:
                    event_callback("tool_result", {
                        "name": "follow_up",
                        "result": suggestion,
                        "is_error": False,
                    })

    transcribe_thread = threading.Thread(target=_transcribe_worker, daemon=True)
    transcribe_thread.start()

    work_buf: list[np.ndarray] = []
    silence_streak = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames,
            device=device_idx,
            callback=_audio_cb,
        ):
            event_callback("status", {"message": "Recording started. Speak now…"})

            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                work_buf.append(chunk)

                if rms < SILENCE_THRESHOLD:
                    silence_streak += 1
                else:
                    silence_streak = 0

                should_flush = (
                    silence_streak >= silence_threshold_chunks
                    or len(work_buf) >= max_work_chunks
                )

                if not should_flush:
                    continue

                audio_array = np.concatenate(work_buf, axis=0).flatten()
                work_buf.clear()
                silence_streak = 0

                if len(audio_array) < SAMPLE_RATE * 0.3:
                    continue  # skip sub-300ms fragments (clicks, breath)

                try:
                    transcribe_q.put_nowait(audio_array)
                except queue.Full:
                    logger.warning("[Tribbie] Transcription queue full — dropping segment")

    except Exception as e:
        logger.exception("[Tribbie] Fatal error in capture loop")
        event_callback("error", {"message": str(e)})
    finally:
        # Drain remaining audio then shut down the transcription thread
        if work_buf:
            audio_array = np.concatenate(work_buf, axis=0).flatten()
            if len(audio_array) >= SAMPLE_RATE * 0.3:
                try:
                    transcribe_q.put_nowait(audio_array)
                except queue.Full:
                    pass
        transcribe_q.put(None)  # poison pill
        transcribe_thread.join(timeout=30)
    finally:
        _sessions.pop(job_id, None)

    # --- Save transcript ---
    if transcript_lines:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = P.transcripts_dir(company)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"live_interview_{timestamp}.txt"
        out_path.write_text("\n\n".join(transcript_lines), encoding="utf-8")
        duration_min = round((time.time() - start_time) / 60, 1)
        event_callback("done", {
            "output": str(out_path),
            "message": (
                f"Session complete — {segment_count} segments, {duration_min} min. "
                f"Transcript saved: {out_path.name}"
            ),
        })
    else:
        event_callback("done", {
            "output": None,
            "message": "Session ended — no speech captured.",
        })


def stop_session(job_id: str) -> bool:
    """Signal a running session to stop gracefully. Returns True if session was found."""
    ev = _sessions.get(job_id)
    if ev:
        ev.set()
        return True
    return False
