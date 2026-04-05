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
MAX_CONTEXT_CHARS = 30_000     # max chars of client context sent to Haiku (Haiku has 200k ctx)


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
    Load client context for Haiku:
      1. All existing interview transcripts (so the model knows what's already been captured)
      2. Aglaea briefing (prepared questions and goals for this interview)
      3. Content strategy snippet (ICP and post-type targets)
    Returns at most MAX_CONTEXT_CHARS characters.
    """
    parts: list[str] = []
    total = 0

    # 1. All existing interview transcripts — the primary source for knowing what's been covered.
    #    Skip any live_interview_* files (those are from the current or past Tribbie sessions
    #    and will already appear in transcript_so_far). Load named transcripts only.
    trans_dir = P.transcripts_dir(company)
    if trans_dir.exists():
        transcript_budget = MAX_CONTEXT_CHARS // 2  # reserve half the budget for transcripts
        transcript_parts: list[str] = []
        transcript_total = 0
        for f in sorted(trans_dir.iterdir()):
            if f.suffix not in (".txt", ".md"):
                continue
            if f.name.startswith("live_interview_"):
                continue  # already captured in transcript_so_far
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                chunk = text[:6_000]  # cap each file to keep variety
                transcript_parts.append(f"### {f.name}\n{chunk}")
                transcript_total += len(chunk)
                if transcript_total >= transcript_budget:
                    break
            except Exception:
                continue
        if transcript_parts:
            parts.append("## Existing Interview Transcripts\n" + "\n\n".join(transcript_parts))
            total += transcript_total

    # 2. Aglaea briefing (prepared questions and interview goals)
    brief_file = P.brief_dir(company) / f"{company}_briefing.md"
    if brief_file.exists() and total < MAX_CONTEXT_CHARS:
        text = brief_file.read_text(encoding="utf-8", errors="ignore")[:6_000]
        parts.append(f"## Interview Briefing\n{text}")
        total += len(text)

    # 3. Content strategy (ICP profile, post-type targets — informs what material is useful)
    strat_dir = P.content_strategy_dir(company)
    if strat_dir.exists() and total < MAX_CONTEXT_CHARS:
        for f in sorted(strat_dir.iterdir()):
            if f.suffix in (".md", ".txt"):
                text = f.read_text(encoding="utf-8", errors="ignore")[:3_000]
                parts.append(f"## Content Strategy\n{text}")
                total += len(text)
                break

    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def _suggest_followup(
    segment: str,
    context: str,
    transcript_so_far: str,
    anthropic_client,
) -> str | None:
    """Call Claude Haiku to suggest the single best follow-up question."""
    try:
        snippet = transcript_so_far[-4_000:]
        response = anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "You are assisting a LinkedIn ghostwriter conducting a content interview.\n\n"
                    "Read everything below: the existing transcripts (what has already been captured "
                    "across all sessions), the briefing (what the interviewer planned to cover), the "
                    "content strategy (what types of posts this client needs), and the current "
                    "conversation so far. Then decide the single most useful question to ask right now.\n\n"
                    "The goal is to collect raw material that can become LinkedIn posts. "
                    "A question is useful if it surfaces something not already in the transcripts "
                    "and would produce content the strategy calls for.\n\n"
                    "Quality bar: the question must be specific enough that the client's answer "
                    "would directly produce a LinkedIn post draft — a concrete claim, outcome, or "
                    "decision with enough detail to write from. If the answer would require further "
                    "follow-up to be usable, the question is too vague.\n\n"
                    f"<client_context>\n{context}\n</client_context>\n\n"
                    f"<conversation_so_far>\n{snippet}\n</conversation_so_far>\n\n"
                    f"<just_said>\n{segment}\n</just_said>\n\n"
                    "Reply in exactly two lines:\n"
                    "Line 1: one short italicised note explaining what gap or opportunity you're targeting "
                    "(e.g. *This topic hasn't come up yet* or *Good moment to get the concrete outcome*)\n"
                    "Line 2: the question itself — natural, conversational, specific."
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
        # Drain remaining audio, shut down transcription thread, clean up session
        if work_buf:
            audio_array = np.concatenate(work_buf, axis=0).flatten()
            if len(audio_array) >= SAMPLE_RATE * 0.3:
                try:
                    transcribe_q.put_nowait(audio_array)
                except queue.Full:
                    pass
        transcribe_q.put(None)  # poison pill
        transcribe_thread.join(timeout=30)
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
