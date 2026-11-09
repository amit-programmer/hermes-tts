try:
    import spaces
except ImportError:
    pass

import os
import sys
import asyncio
import time
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Ensure root directory is in sys.path (Fixes Windows folder space import bugs)
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI, HTTPException, Query, Request, Response, File, UploadFile, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.text_cleaner import (
    clean_text_for_tts,
    sanitize_text,
    chunk_large_text,
    detect_language_type,
    optimize_speech_prosody,
    is_pure_code_snippet,
    split_into_sentences,
)
from utils.audio_processor import (
    apply_persona_preset,
    normalize_audio_lufs,
    convert_audio_format,
    stitch_audio_chunks,
    strip_mp3_id3_headers,
    PERSONA_PRESETS,
)
from config import (
    PORT,
    KOKORO_VOICES,
    EDGE_VOICES,
    DEFAULT_VOICE,
    DEFAULT_RATE,
    DEFAULT_PITCH,
    DEFAULT_VOLUME,
)

app = FastAPI(
    title="Unified Hybrid Voice Engine",
    description="Multi-engine TTS pipeline integrating Kokoro-82M, ChatTTS, ElevenLabs & Edge-TTS with auto-fallback",
    version="2.0.0",
)

request_semaphore = asyncio.Semaphore(5)

import hashlib
from collections import OrderedDict

# In-memory LRU Audio Cache for instant (sub-10ms) repeated responses
AUDIO_CACHE = OrderedDict()
MAX_CACHE_SIZE = 100

def get_cache_key(engine: str, voice: str, text: str) -> str:
    raw = f"{engine}:{voice}:{text.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def cache_get(key: str) -> Optional[bytes]:
    if key in AUDIO_CACHE:
        AUDIO_CACHE.move_to_end(key)
        return AUDIO_CACHE[key]
    return None

def cache_put(key: str, data: bytes):
    AUDIO_CACHE[key] = data
    if len(AUDIO_CACHE) > MAX_CACHE_SIZE:
        AUDIO_CACHE.popitem(last=False)

# Enable CORS for frontend apps & web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy/Explicit engine holders
kokoro_engine = None
chattts_engine = None
elevenlabs_engine = None
edge_engine = None
whisper_engine = None

@app.on_event("startup")
async def startup_event():
    global kokoro_engine, chattts_engine, elevenlabs_engine, edge_engine, whisper_engine
    print("\n[INFO] Starting TTS & STT voice engine pre-warming...")

    # 1. Kokoro init
    try:
        from core.kokoro_engine import KokoroEngine
        kokoro_engine = KokoroEngine(default_voice=os.getenv("KOKORO_VOICE", "af_bella"))
        print("[SUCCESS] Kokoro-82M Engine loaded.")
    except Exception as e:
        print(f"[ERROR] Failed to load Kokoro: {e}")

    # 2. ChatTTS / HuggingFace disabled as requested
    chattts_engine = None
    print("[INFO] HuggingFace / ChatTTS integration disabled.")

    # 3. ElevenLabs init
    try:
        from core.elevenlabs_engine import ElevenLabsEngine
        elevenlabs_engine = ElevenLabsEngine()
        print("[SUCCESS] ElevenLabs Engine ready.")
    except Exception as e:
        print(f"[ERROR] Failed to load ElevenLabs: {e}")

    # 4. Edge-TTS init & background pre-warm
    try:
        from core.edge_engine import EdgeTTSEngine
        edge_engine = EdgeTTSEngine(default_voice=os.getenv("EDGE_DEFAULT_VOICE", "hi-IN-SwaraNeural"))
        print("[SUCCESS] Edge-TTS Engine ready.")

        # Background pre-warm DNS & socket so first user request has zero cold-start delay
        async def background_prewarm():
            try:
                await edge_engine.generate_async("Namaste")
                print("[PRE-WARM] Edge-TTS socket & DNS pre-warmed successfully.")
            except Exception as pe:
                print(f"[PRE-WARM] Background pre-warm notice: {pe}")
        asyncio.create_task(background_prewarm())
    except Exception as e:
        print(f"[ERROR] Failed to load Edge-TTS: {e}")

    # 5. Whisper STT init
    try:
        from core.whisper_engine import WhisperEngine
        whisper_engine = WhisperEngine()
        print("[SUCCESS] Whisper STT Engine ready.")
    except Exception as e:
        print(f"[ERROR] Failed to load Whisper STT: {e}")

    print("[INFO] Unified Hybrid Voice Engines (TTS + STT) startup complete.")
    AUDIO_CACHE.clear()


@app.on_event("shutdown")
async def shutdown_event():
    print("\n[INFO] Graceful shutdown initiated...")
    AUDIO_CACHE.clear()
    print("[SUCCESS] Audio cache cleared and resources freed.")


@app.get("/health")
def health_check():
    """Health check endpoint providing engine readiness and cache metrics."""
    return {
        "status": "ok",
        "engines": {
            "edge": edge_engine is not None,
            "kokoro": kokoro_engine is not None,
            "elevenlabs": elevenlabs_engine is not None,
            "whisper_stt": whisper_engine is not None,
        },
        "capabilities": ["TTS", "STT"],
        "audio_cache_entries": len(AUDIO_CACHE),
        "version": "2.1.0",
    }


# =====================================================================
# Speech-to-Text (STT) Endpoints
# =====================================================================

@app.post("/v1/stt/transcribe")
async def transcribe_audio_endpoint(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
):
    """
    Speech-to-Text (STT) endpoint:
    Transcribes uploaded audio (WAV, MP3, OGG, WEBM, M4A) using Whisper.
    """
    global whisper_engine
    if whisper_engine is None:
        try:
            from core.whisper_engine import WhisperEngine
            whisper_engine = WhisperEngine()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Whisper STT engine unavailable: {e}")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            whisper_engine.transcribe,
            audio_bytes,
            language,
            task,
            file.filename,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/v1/audio/transcriptions")
async def openai_compatible_transcription(
    file: UploadFile = File(...),
    model: Optional[str] = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    OpenAI-compatible Whisper Audio Transcription Endpoint.
    Drop-in replacement for api.openai.com/v1/audio/transcriptions.
    """
    global whisper_engine
    if whisper_engine is None:
        try:
            from core.whisper_engine import WhisperEngine
            whisper_engine = WhisperEngine()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Whisper STT engine unavailable: {e}")

    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            whisper_engine.transcribe,
            audio_bytes,
            language,
            "transcribe",
            file.filename,
        )
        if response_format == "text":
            return Response(content=result.get("text", ""), media_type="text/plain")
        return {"text": result.get("text", "")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


class TTSRequest(BaseModel):
    text: str
    engine: str = "auto"  # options: "kokoro", "chattts", "elevenlabs", "edge", "auto"
    voice_id: Optional[str] = None


@app.post("/synthesize")
async def synthesize_voice(req: TTSRequest):
    """
    Unified speech synthesis endpoint with intelligent auto-fallback:
    - kokoro: Zero-cost, ultra-fast, local CPU female voice (af_bella)
    - chattts: Conversational & emotional with locked seed vector
    - elevenlabs: Premium studio quality voice with auto-fallback to Kokoro
    - edge: Free cloud neural voices (Hindi, Hinglish, English)
    - auto: Kokoro (primary default) -> auto-fallback to Edge-TTS on failure
    """
    async with request_semaphore:
        # Apply prosody pacing (natural breath pauses at commas/ellipses) and text cleaning
        prosody_text = optimize_speech_prosody(req.text)
        cleaned_text = clean_text_for_tts(prosody_text)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Empty text supplied after cleaning.")

        detected_lang = detect_language_type(cleaned_text)
        selected_engine = (req.engine or "auto").lower()
        start_time = time.time()

        cache_hit_flag = [False]

        def make_headers(engine_name: str, elapsed_ms: int, fallback: bool = False, reason: str = "none"):
            return {
                "X-Engine-Used": engine_name,
                "X-Response-Time-Ms": str(elapsed_ms),
                "X-Cache-Hit": "true" if cache_hit_flag[0] else "false",
                "X-Fallback-Triggered": "true" if fallback else "false",
                "X-Fallback-Reason": str(reason) if fallback else "none",
                "Content-Disposition": f'inline; filename="{engine_name.replace(" ", "_")}.audio"',
            }

        # Voice ID validation: prevent cross-engine voice mismatches
        def resolve_voice_for_engine(engine_name: str, raw_voice_id: str = None) -> Optional[str]:
            """Return a valid voice_id for the given engine, preventing cross-engine voice ID mismatches."""
            vid = (raw_voice_id or "").strip()
            if engine_name == "elevenlabs":
                if not vid or vid == "21m00Tcm4TlvDq8ikWAM" or vid.startswith(("af_", "am_", "bf_", "bm_")) or "Neural" in vid:
                    return os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
                return vid
            elif engine_name == "kokoro":
                if vid.startswith(("af_", "am_", "bf_", "bm_")):
                    return vid
                return os.getenv("KOKORO_VOICE", "af_bella")
            elif engine_name == "chattts":
                return None
            elif engine_name == "edge":
                if vid and "Neural" in vid:
                    return vid
                # Discard ElevenLabs or Kokoro IDs passed to Edge, default to Swara for Hinglish / Indian accent
                return "hi-IN-SwaraNeural" if detected_lang == "hinglish" else os.getenv("EDGE_DEFAULT_VOICE", "hi-IN-SwaraNeural")
            return None

        async def generate_single_chunk(chunk_text: str, engine_name: str, resolved_voice: Optional[str]) -> bytes:
            if engine_name == "edge":
                if not edge_engine:
                    raise RuntimeError("Edge-TTS engine not initialized.")
                buf = await edge_engine.generate_async(chunk_text, voice=resolved_voice)
                return buf.getvalue()
            elif engine_name == "kokoro":
                if not kokoro_engine:
                    raise RuntimeError("Kokoro engine not initialized.")
                loop = asyncio.get_running_loop()
                buf = await loop.run_in_executor(None, kokoro_engine.generate, chunk_text, resolved_voice)
                return buf.getvalue()
            elif engine_name == "chattts":
                if kokoro_engine:
                    loop = asyncio.get_running_loop()
                    buf = await loop.run_in_executor(None, kokoro_engine.generate, chunk_text, resolved_voice)
                    return buf.getvalue()
                elif edge_engine:
                    buf = await edge_engine.generate_async(chunk_text)
                    return buf.getvalue()
                else:
                    raise RuntimeError("Kokoro/Edge engines not initialized.")
            elif engine_name == "elevenlabs":
                if not elevenlabs_engine:
                    raise RuntimeError("ElevenLabs engine not initialized.")
                loop = asyncio.get_running_loop()
                buf = await loop.run_in_executor(None, elevenlabs_engine.generate, chunk_text, resolved_voice)
                return buf.getvalue()
            else:
                raise RuntimeError(f"Unsupported engine: {engine_name}")

        async def generate_full_audio(text: str, engine_name: str, voice_id_override=None) -> bytes:
            """Pre-generate audio bytes combining chunks concurrently via asyncio.gather (Promise.all)."""
            candidate_voice = voice_id_override if voice_id_override is not None else req.voice_id
            resolved_voice = resolve_voice_for_engine(engine_name, candidate_voice)

            # Check in-memory LRU cache for instant sub-10ms response
            ckey = get_cache_key(engine_name, resolved_voice or "default", text)
            cached_audio = cache_get(ckey)
            if cached_audio is not None:
                cache_hit_flag[0] = True
                return cached_audio

            # Intelligent chunking: chunks up to 1000 characters stay in 1 fast roundtrip!
            text_chunks = chunk_large_text(text, max_chars=1000)
            if not text_chunks:
                raise RuntimeError(f"No audio chunks available for engine={engine_name}")

            if len(text_chunks) == 1:
                audio_res = await generate_single_chunk(text_chunks[0], engine_name, resolved_voice)
            else:
                # Concurrent Promise Execution: run all chunks in parallel using asyncio.gather!
                tasks = [generate_single_chunk(chunk, engine_name, resolved_voice) for chunk in text_chunks]
                chunk_buffers = await asyncio.gather(*tasks)
                audio_res = b"".join(chunk_buffers)

            cache_put(ckey, audio_res)
            return audio_res

        # --- Engine routing with real fallback ---
        audio_bytes = None
        used_engine = selected_engine
        media_type = "audio/wav"
        fallback = False
        fallback_reason = "none"

        if selected_engine == "elevenlabs":
            media_type = "audio/mpeg"
            try:
                audio_bytes = await generate_full_audio(cleaned_text, "elevenlabs")
                used_engine = "elevenlabs"
            except Exception as e:
                reason = str(e)
                print(f"[WARN] ElevenLabs failed: {reason}.")
                fallback = True
                fallback_reason = reason
                if detected_lang == "hinglish":
                    print("[AUTO-ROUTE] Hinglish text detected: Falling back to Edge-TTS (hi-IN-SwaraNeural).")
                    audio_bytes = await generate_full_audio(cleaned_text, "edge", voice_id_override="hi-IN-SwaraNeural")
                    used_engine = "edge (SwaraNeural - Native Indian Accent)"
                    media_type = "audio/mpeg"
                else:
                    try:
                        audio_bytes = await generate_full_audio(cleaned_text, "kokoro", voice_id_override="af_bella")
                        used_engine = "kokoro"
                        media_type = "audio/wav"
                    except Exception as e2:
                        fallback_reason = str(e2)
                        audio_bytes = await generate_full_audio(cleaned_text, "edge")
                        used_engine = "edge"
                        media_type = "audio/mpeg"

        elif selected_engine == "edge":
            # Free & native Indian female voice (hi-IN-SwaraNeural for Hinglish / Hindi)
            resolved_edge_voice = resolve_voice_for_engine("edge", req.voice_id)
            audio_bytes = await generate_full_audio(cleaned_text, "edge", voice_id_override=resolved_edge_voice)
            used_engine = f"edge ({resolved_edge_voice})"
            media_type = "audio/mpeg"

        elif selected_engine == "kokoro":
            # Kokoro is English-only. If Hinglish text is provided, Kokoro sounds robotic/foreign.
            if detected_lang == "hinglish":
                print("[NOTICE] Kokoro is English-only. Routing Hinglish text to Edge-TTS hi-IN-SwaraNeural for natural accent.")
                fallback = True
                fallback_reason = "Kokoro does not support Hinglish phonetics; routed to SwaraNeural"
                audio_bytes = await generate_full_audio(cleaned_text, "edge", voice_id_override="hi-IN-SwaraNeural")
                used_engine = "edge (SwaraNeural - Hinglish Protected)"
                media_type = "audio/mpeg"
            else:
                try:
                    audio_bytes = await generate_full_audio(cleaned_text, "kokoro", voice_id_override="af_bella")
                    used_engine = "kokoro (af_bella)"
                    media_type = "audio/wav"
                except Exception as e:
                    fallback = True
                    fallback_reason = str(e)
                    audio_bytes = await generate_full_audio(cleaned_text, "edge", voice_id_override="en-IN-NeerjaNeural")
                    used_engine = "edge (NeerjaNeural)"
                    media_type = "audio/mpeg"

        else:  # auto mode: Smart Context & Language Aware Selection
            if detected_lang == "hinglish":
                # Hinglish / Hindi: Edge-TTS hi-IN-SwaraNeural (Free Native Accent) -> ElevenLabs fallback
                try:
                    audio_bytes = await generate_full_audio(cleaned_text, "edge", voice_id_override="hi-IN-SwaraNeural")
                    used_engine = "edge (hi-IN-SwaraNeural - Native Girl Voice)"
                    media_type = "audio/mpeg"
                except Exception as e:
                    reason = str(e)
                    print(f"[WARN] Edge-TTS Swara failed: {reason}. Trying ElevenLabs Multilingual.")
                    fallback = True
                    fallback_reason = reason
                    audio_bytes = await generate_full_audio(cleaned_text, "elevenlabs")
                    used_engine = "elevenlabs (Multilingual V2)"
                    media_type = "audio/mpeg"
            else:
                # Pure English: Kokoro-82M (Fast local English) -> Edge-TTS fallback
                try:
                    audio_bytes = await generate_full_audio(cleaned_text, "kokoro", voice_id_override="af_bella")
                    used_engine = "kokoro (af_bella - Pure English)"
                    media_type = "audio/wav"
                except Exception as e:
                    reason = str(e)
                    print(f"[WARN] Kokoro English failed: {reason}. Falling back to Edge-TTS.")
                    fallback = True
                    fallback_reason = reason
                    audio_bytes = await generate_full_audio(cleaned_text, "edge", voice_id_override="en-IN-NeerjaNeural")
                    used_engine = "edge (en-IN-NeerjaNeural)"
                    media_type = "audio/mpeg"

        if not audio_bytes:
            raise HTTPException(status_code=500, detail="All TTS engines failed to generate audio.")

        import io
        elapsed_ms = int((time.time() - start_time) * 1000)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers=make_headers(used_engine, elapsed_ms, fallback=fallback, reason=fallback_reason)
        )


@app.get("/health")
def health():
    """
    Health check endpoint reporting available engines and configuration status.
    """
    has_eleven_key = bool(os.getenv("ELEVENLABS_API_KEY"))
    return {
        "status": "ok",
        "service": "tts-engine",
        "description": "Unified Hybrid Voice Engine for Hermes AI Agent",
        "available_engines": ["kokoro", "chattts", "elevenlabs", "edge"],
        "primary_engine": "kokoro",
        "default_kokoro_voice": os.getenv("KOKORO_VOICE", "af_bella"),
        "elevenlabs_configured": has_eleven_key,
        "chattts_seed": int(os.getenv("CHATTTS_SEED", 2222)),
    }


# =====================================================================
# Hermes Dedicated Voice Assistant Routes (v1 API Schema)
# =====================================================================

VOICE_MAP = {
    "hinglish_female": "hi-IN-SwaraNeural",
    "english_in_female": "en-IN-NeerjaNeural",
    "english_us_female": "en-US-JennyNeural",
    "kokoro_fast_female": "af_bella",
    "elevenlabs_studio_female": os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9"),
}


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2500, description="Input text to synthesize (1-2500 characters)")
    voice_profile: Optional[str] = None
    rate: Optional[str] = "+0%"
    pitch: Optional[str] = "+0Hz"
    clean_markdown: Optional[bool] = True
    format: Optional[str] = "mp3"
    engine: Optional[str] = "auto"
    persona: Optional[str] = None


class CleanTextRequest(BaseModel):
    text: str


def join_audio_chunks(chunks: list) -> bytes:
    """Concatenates raw audio chunks (RIFF WAV or MPEG frames) cleanly in memory."""
    if not chunks:
        return b""
    if len(chunks) == 1:
        return chunks[0]

    # If chunks are RIFF WAV, safely merge PCM bodies and update WAV header
    if chunks[0].startswith(b"RIFF"):
        import struct

        data_parts = []
        for c in chunks:
            if c.startswith(b"RIFF") and len(c) > 44:
                data_parts.append(c[44:])
            else:
                data_parts.append(c)
        combined_pcm = b"".join(data_parts)
        header = bytearray(chunks[0][:44])
        total_len = len(combined_pcm)
        header[4:8] = struct.pack("<I", total_len + 36)
        header[40:44] = struct.pack("<I", total_len)
        return bytes(header) + combined_pcm
    else:
        # MP3 frames concatenate natively
        return b"".join(chunks)


async def synthesize_sentence_chunk(
    sentence: str,
    target_engine: str,
    target_voice: Optional[str],
    rate_val: str = "+0%",
    pitch_val: str = "+0Hz",
) -> bytes:
    """
    Synthesizes an individual sentence chunk using the specified engine with automatic fallback.
    Returns in-memory audio bytes (WAV from Kokoro, MP3 from Edge/ElevenLabs).
    """
    engine_attempt = (target_engine or "edge").lower().strip()

    # 1. Kokoro-82M (Fast local CPU engine for English)
    if engine_attempt == "kokoro":
        try:
            if not kokoro_engine:
                raise RuntimeError("Kokoro-82M engine not loaded.")
            kokoro_voice = (
                target_voice
                if target_voice and target_voice.startswith(("af_", "am_", "bf_", "bm_"))
                else os.getenv("KOKORO_VOICE", "af_bella")
            )
            loop = asyncio.get_running_loop()
            buf = await asyncio.wait_for(
                loop.run_in_executor(None, kokoro_engine.generate, sentence, kokoro_voice),
                timeout=4.0,
            )
            return buf.getvalue()
        except Exception as e:
            print(f"[FALLBACK] Kokoro timed out / failed on chunk: {e}. Falling back to Edge-TTS.")
            engine_attempt = "edge"
            target_voice = "en-IN-NeerjaNeural"

    # 2. ElevenLabs (Hyper-realistic studio voice)
    if engine_attempt == "elevenlabs":
        try:
            if not elevenlabs_engine:
                raise RuntimeError("ElevenLabs engine not loaded.")
            eleven_voice = (
                target_voice
                if target_voice and not "Neural" in target_voice and not target_voice.startswith(("af_", "am_"))
                else os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
            )
            loop = asyncio.get_running_loop()
            buf = await asyncio.wait_for(
                loop.run_in_executor(None, elevenlabs_engine.generate, sentence, eleven_voice),
                timeout=5.0,
            )
            return buf.getvalue()
        except Exception as e:
            print(f"[FALLBACK] ElevenLabs timed out / failed on chunk: {e}. Falling back to Edge-TTS.")
            engine_attempt = "edge"
            target_voice = "hi-IN-SwaraNeural"

    # 3. Edge-TTS (Default resilient neural cloud engine)
    global edge_engine
    if not edge_engine:
        from core.edge_engine import EdgeTTSEngine

        edge_engine = EdgeTTSEngine()
    edge_voice = (
        target_voice
        if (target_voice and "Neural" in target_voice)
        else os.getenv("EDGE_DEFAULT_VOICE", "hi-IN-SwaraNeural")
    )
    buf = await edge_engine.generate_async(sentence, voice=edge_voice, rate=rate_val, pitch=pitch_val)
    return buf.getvalue()


@app.get("/v1/voices")
def list_v1_voices():
    """
    Returns supported voice profiles, personas, engines, and formats for Hermes agent.
    """
    return {
        "profiles": list(VOICE_MAP.keys()),
        "mapping": VOICE_MAP,
        "personas": list(PERSONA_PRESETS.keys()),
        "supported_engines": ["auto", "edge", "kokoro", "elevenlabs"],
        "supported_formats": ["mp3", "ogg", "opus", "wav", "pcm"],
    }


@app.post("/v1/tts/synthesize")
async def synthesize_audio_v1(req: SynthesizeRequest):
    """
    Direct in-memory binary audio buffer route for Telegram bot (bot.send_voice) or direct API download.
    Zero disk I/O. Supports Multi-Engine routing, Persona pacing, Multi-Format encoding, and LUFS balancing.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "EMPTY_PAYLOAD", "message": "Text payload cannot be empty."},
        )

    if len(req.text) > 2500:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "PAYLOAD_TOO_LARGE",
                "message": f"Text exceeds maximum permitted length of 2500 characters (received {len(req.text)} characters).",
            },
        )

    content = sanitize_text(req.text) if req.clean_markdown else req.text
    if not content or not content.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "EMPTY_AFTER_CLEANING", "message": "Text content contains only unpronounceable symbols or empty markup."},
        )

    content = optimize_speech_prosody(content)
    rate, pitch = apply_persona_preset(req.persona, req.rate, req.pitch)

    engine_req = (req.engine or "auto").lower().strip()
    detected_lang = detect_language_type(content)

    if engine_req == "auto":
        target_engine = "edge" if detected_lang == "hinglish" else "kokoro"
    else:
        target_engine = engine_req

    if target_engine == "kokoro":
        default_voice = os.getenv("KOKORO_VOICE", "af_bella")
    elif target_engine == "elevenlabs":
        default_voice = os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
    else:
        default_voice = "hi-IN-SwaraNeural" if detected_lang == "hinglish" else "en-IN-NeerjaNeural"

    if req.voice_profile and req.voice_profile in VOICE_MAP:
        voice = VOICE_MAP[req.voice_profile]
    elif req.voice_profile:
        voice = req.voice_profile
    else:
        voice = default_voice

    sentences = split_into_sentences(content)
    if not sentences:
        sentences = [content]

    try:
        if len(sentences) == 1:
            raw_audio = await synthesize_sentence_chunk(sentences[0], target_engine, voice, rate, pitch)
        else:
            tasks = [synthesize_sentence_chunk(s, target_engine, voice, rate, pitch) for s in sentences]
            chunks = await asyncio.gather(*tasks)
            raw_audio = stitch_audio_chunks(chunks)

        target_fmt = (req.format or "mp3").lower().strip()
        final_bytes, media_type = convert_audio_format(
            raw_audio, target_format=target_fmt, normalize_volume=True
        )

        return Response(
            content=final_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="voice.{target_fmt}"',
                "X-Engine-Used": target_engine,
                "X-Voice-Used": voice,
                "X-Audio-Format": target_fmt,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "ALL_ENGINES_FAILED",
                "engine": target_engine,
                "reason": str(e),
                "status": "failed",
            },
        )


@app.post("/v1/tts/stream")
async def stream_audio_v1(req: SynthesizeRequest, request: Request):
    """
    True Sentence-Level Stream Buffering for ultra-low latency (TTFA < 500ms).
    Synthesizes and delivers the first sentence immediately while prefetching subsequent sentences.
    Includes client abort check and bounded queue to eliminate memory leaks.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "EMPTY_PAYLOAD", "message": "Text payload cannot be empty."},
        )

    if len(req.text) > 2500:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "PAYLOAD_TOO_LARGE",
                "message": f"Text exceeds maximum permitted length of 2500 characters (received {len(req.text)} characters).",
            },
        )

    content = sanitize_text(req.text) if req.clean_markdown else req.text
    if not content or not content.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "EMPTY_AFTER_CLEANING", "message": "Text content contains only unpronounceable symbols or empty markup."},
        )

    content = optimize_speech_prosody(content)
    rate, pitch = apply_persona_preset(req.persona, req.rate, req.pitch)

    engine_req = (req.engine or "auto").lower().strip()
    detected_lang = detect_language_type(content)

    if engine_req == "auto":
        target_engine = "edge" if detected_lang == "hinglish" else "kokoro"
    else:
        target_engine = engine_req

    if target_engine == "kokoro":
        default_voice = os.getenv("KOKORO_VOICE", "af_bella")
    elif target_engine == "elevenlabs":
        default_voice = os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
    else:
        default_voice = "hi-IN-SwaraNeural" if detected_lang == "hinglish" else "en-IN-NeerjaNeural"

    if req.voice_profile and req.voice_profile in VOICE_MAP:
        voice = VOICE_MAP[req.voice_profile]
    elif req.voice_profile:
        voice = req.voice_profile
    else:
        voice = default_voice

    sentences = split_into_sentences(content)
    if not sentences:
        sentences = [content]

    target_fmt = (req.format or "mp3").lower().strip()
    # Bounded Queue (maxsize=2): Limits RAM to at most 2 prefetched sentences
    queue = asyncio.Queue(maxsize=2)

    async def producer():
        first_chunk = True
        try:
            for s in sentences:
                if not s.strip():
                    continue

                # 1. Abort check: if client disconnected, stop generating immediately
                if await request.is_disconnected():
                    print("[STREAM] Client disconnected before sentence generation. Halting producer.")
                    break

                raw_chunk = await synthesize_sentence_chunk(s, target_engine, voice, rate, pitch)
                chunk_bytes, _ = convert_audio_format(
                    raw_chunk, target_format=target_fmt, normalize_volume=True
                )
                if target_fmt == "mp3" and not first_chunk:
                    # Strip ID3 metadata from consecutive stream chunks so HTML5 audio players do not stall
                    chunk_bytes = strip_mp3_id3_headers(chunk_bytes)
                first_chunk = False

                # 2. Check disconnect again before enqueue
                if await request.is_disconnected():
                    print("[STREAM] Client disconnected before enqueue. Dropping chunk.")
                    break

                await queue.put(chunk_bytes)
        except asyncio.CancelledError:
            print("[STREAM] Producer task cancelled.")
        except Exception as pe:
            print(f"[STREAM-PRODUCER-ERR] {pe}")
        finally:
            await queue.put(None)  # Sentinel to end stream

    producer_task = asyncio.create_task(producer())

    async def audio_generator():
        try:
            while True:
                if await request.is_disconnected():
                    print("[STREAM] Consumer detected client disconnect. Exiting audio generator.")
                    break
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            if not producer_task.done():
                producer_task.cancel()

    media_types = {
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    out_media = media_types.get(target_fmt, "audio/mpeg")

    return StreamingResponse(
        audio_generator(),
        media_type=out_media,
        headers={
            "Content-Disposition": f'inline; filename="stream.{target_fmt}"',
            "X-Engine-Used": target_engine,
            "X-Voice-Used": voice,
            "X-Stream-Mode": "sentence-buffered",
            "X-Audio-Format": target_fmt,
        },
    )


@app.post("/v1/tts/clean-text")
async def preview_cleaned_text(request: Request):
    """
    Sanitize raw markdown and code blocks for speech tuning and debugging.
    Supports JSON body {"text": "..."}, Query param (?text=...), or raw text body.
    """
    text_content = ""
    try:
        data = await request.json()
        if isinstance(data, dict):
            text_content = data.get("text", "")
        elif isinstance(data, str):
            text_content = data
    except Exception:
        if "text" in request.query_params:
            text_content = request.query_params["text"]
        else:
            raw_bytes = await request.body()
            text_content = raw_bytes.decode("utf-8", errors="ignore")

    cleaned_speech = optimize_speech_prosody(sanitize_text(text_content))

    return {
        "original": text_content,
        "clean_for_speech": cleaned_speech,
    }


@app.get("/api/voices")
def get_curated_voices():
    """
    Returns curated voices across all integrated engines.
    """
    return {
        "kokoro": KOKORO_VOICES,
        "edge": EDGE_VOICES,
        "elevenlabs_recommended": {
            "21m00Tcm4TlvDq8ikWAM": "Rachel (Studio Female)",
            "EXAVITQu4vr4xnSDxMaL": "Bella (Warm Narration)",
            "2EiwWnXFnvU5JabPnv8n": "Clyde (Natural Male)",
            "ErXwobaYiN019PkySvjV": "Antoni (Deep Male)",
        },
        "chattts": {
            "2222": "Sweet young female voice (locked seed)",
            "3333": "Warm narrator female voice",
            "5555": "Soft friendly tone",
        },
    }


@app.get("/api/tts")
async def legacy_stream_tts(
    text: str = Query(..., description="Text to synthesize"),
    voice: str = Query(DEFAULT_VOICE, description="Edge voice identifier"),
    rate: str = Query(DEFAULT_RATE, description="Speed modifier e.g. +10%"),
    pitch: str = Query(DEFAULT_PITCH, description="Pitch modifier e.g. +2Hz"),
    volume: str = Query(DEFAULT_VOLUME, description="Volume modifier e.g. +5%"),
    clean: bool = Query(True, description="Clean markdown & code"),
):
    """
    Backward-compatible Edge-TTS streaming endpoint.
    """
    target_text = clean_text_for_tts(text) if clean else text.strip()
    if not target_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    buffer = await edge_engine.generate_async(target_text, voice=voice, rate=rate, pitch=pitch, volume=volume)
    return StreamingResponse(
        buffer,
        media_type="audio/mpeg",
        headers={
            "X-Engine-Used": "edge",
            "Content-Disposition": 'inline; filename="speech.mp3"',
        },
    )


@app.get("/synthesize_stream")
async def synthesize_stream(
    text: str = Query(..., description="Text to synthesize"),
    engine: str = Query("auto", description="TTS Engine"),
    voice_id: Optional[str] = Query(None, description="Voice ID"),
):
    """
    Ultra-low latency real-time streaming endpoint.
    Streams audio packets as they arrive for sub-second Time-to-First-Audio (TTFA).
    """
    cleaned_text = optimize_speech_prosody(clean_text_for_tts(text))
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    detected_lang = detect_language_type(cleaned_text)
    selected_engine = (engine or "auto").lower()

    if selected_engine == "auto":
        selected_engine = "edge" if detected_lang == "hinglish" else "kokoro"

    # 1. Edge-TTS Real-Time Streaming
    if selected_engine == "edge":
        resolved_voice = "hi-IN-SwaraNeural" if detected_lang == "hinglish" else os.getenv("EDGE_DEFAULT_VOICE", "hi-IN-SwaraNeural")
        if voice_id and "Neural" in voice_id:
            resolved_voice = voice_id.strip()

        async def edge_stream_generator():
            try:
                async for chunk in edge_engine.stream_audio_async(cleaned_text, voice=resolved_voice):
                    yield chunk
            except Exception as e:
                print(f"[STREAM-WARN] Edge streaming error: {e}")

        return StreamingResponse(
            edge_stream_generator(),
            media_type="audio/mpeg",
            headers={"X-Engine-Used": f"edge ({resolved_voice})", "Content-Disposition": 'inline; filename="stream.mp3"'}
        )

    # 2. ElevenLabs Real-Time Streaming
    elif selected_engine == "elevenlabs":
        resolved_voice = voice_id if (voice_id and not "Neural" in voice_id and not voice_id.startswith(("af_", "am_"))) else os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")

        def eleven_stream_generator():
            try:
                for chunk in elevenlabs_engine.stream_audio(cleaned_text, voice_id=resolved_voice):
                    yield chunk
            except Exception as e:
                print(f"[STREAM-WARN] ElevenLabs streaming error: {e}")

        return StreamingResponse(
            eleven_stream_generator(),
            media_type="audio/mpeg",
            headers={"X-Engine-Used": "elevenlabs", "Content-Disposition": 'inline; filename="stream.mp3"'}
        )

    # 3. Kokoro (Fast local generation)
    else:
        resolved_voice = voice_id if (voice_id and voice_id.startswith(("af_", "am_"))) else os.getenv("KOKORO_VOICE", "af_bella")
        loop = asyncio.get_running_loop()
        buf = await loop.run_in_executor(None, kokoro_engine.generate, cleaned_text, resolved_voice)
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"X-Engine-Used": f"kokoro ({resolved_voice})", "Content-Disposition": 'inline; filename="stream.wav"'}
        )


@app.get("/", response_class=HTMLResponse)
async def playground_ui():
    """
    Modern glassmorphic Web Playground for interactive testing of all TTS engines.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Unified Hybrid Voice Engine • Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090812;
      --card-bg: rgba(22, 18, 38, 0.7);
      --card-border: rgba(168, 85, 247, 0.22);
      --accent-purple: #a855f7;
      --accent-cyan: #06b6d4;
      --accent-pink: #ec4899;
      --accent-gradient: linear-gradient(135deg, #a855f7, #6366f1 50%, #06b6d4);
      --text: #f8fafc;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body {
      min-height: 100vh;
      background: radial-gradient(circle at 15% 15%, #1e1338 0%, #090812 70%);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2.5rem 1rem;
    }
    .container { width: 100%; max-width: 900px; }
    .header { text-align: center; margin-bottom: 2rem; }
    .badge {
      display: inline-flex; align-items: center; gap: 0.5rem;
      background: rgba(168, 85, 247, 0.12);
      border: 1px solid rgba(168, 85, 247, 0.35);
      color: #c084fc;
      padding: 0.35rem 0.9rem;
      border-radius: 9999px;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-bottom: 0.8rem;
    }
    .badge .dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; box-shadow: 0 0 8px #22c55e; }
    h1 {
      font-size: 2.4rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }
    .subtitle { color: var(--muted); font-size: 0.98rem; }
    
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 1.25rem;
      padding: 2rem;
      box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.6);
      margin-bottom: 1.5rem;
    }

    .section-title { font-size: 0.88rem; font-weight: 700; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.75rem; }

    /* Engine Selector Pills */
    .engine-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }
    .engine-btn {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 0.85rem;
      padding: 0.85rem 0.75rem;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s ease;
      color: var(--text);
    }
    .engine-btn:hover { border-color: rgba(168, 85, 247, 0.5); background: rgba(168, 85, 247, 0.06); }
    .engine-btn.active {
      border-color: #a855f7;
      background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(99, 102, 241, 0.15));
      box-shadow: 0 0 16px rgba(168, 85, 247, 0.3);
    }
    .engine-title { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.2rem; display: flex; align-items: center; justify-content: space-between; }
    .engine-desc { font-size: 0.75rem; color: var(--muted); }

    /* Text Input Area */
    .textarea-wrapper { margin-bottom: 1.25rem; }
    textarea {
      width: 100%;
      min-height: 120px;
      background: rgba(10, 8, 20, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 0.75rem;
      padding: 1rem;
      color: #fff;
      font-size: 0.95rem;
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
    }
    textarea:focus { border-color: var(--accent-purple); box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2); }

    /* Quick Prompt Chips */
    .chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.25rem; }
    .chip {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 9999px;
      padding: 0.35rem 0.75rem;
      font-size: 0.78rem;
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.2s;
    }
    .chip:hover { border-color: var(--accent-cyan); color: #fff; }

    /* Voice Options Row */
    .options-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    select, input[type="text"] {
      width: 100%;
      background: rgba(10, 8, 20, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 0.65rem;
      padding: 0.65rem 0.85rem;
      color: #fff;
      font-size: 0.88rem;
      outline: none;
    }

    /* Primary Generate Button */
    .btn-generate {
      width: 100%;
      background: var(--accent-gradient);
      color: #fff;
      font-weight: 700;
      font-size: 1rem;
      padding: 0.95rem;
      border: none;
      border-radius: 0.85rem;
      cursor: pointer;
      transition: transform 0.15s ease, box-shadow 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }
    .btn-generate:hover { transform: translateY(-2px); box-shadow: 0 10px 25px -5px rgba(168, 85, 247, 0.5); }
    .btn-generate:active { transform: translateY(0); }
    .btn-generate:disabled { opacity: 0.6; cursor: not-allowed; }

    /* Player Container */
    .player-card {
      display: none;
      background: rgba(10, 8, 20, 0.9);
      border: 1px solid rgba(168, 85, 247, 0.3);
      border-radius: 1rem;
      padding: 1.25rem;
      margin-top: 1.5rem;
    }
    audio { width: 100%; margin-top: 0.75rem; }
    .meta-bar {
      display: flex; justify-content: space-between; align-items: center;
      font-size: 0.82rem; color: var(--muted);
    }
    .badge-engine {
      background: rgba(34, 197, 94, 0.15);
      color: #4ade80;
      border: 1px solid rgba(34, 197, 94, 0.3);
      padding: 0.25rem 0.65rem;
      border-radius: 0.5rem;
      font-weight: 600;
    }

    /* Diagnostics & API Snippet */
    .snippet-box {
      background: rgba(5, 4, 10, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 0.75rem;
      padding: 1rem;
      font-family: monospace;
      font-size: 0.82rem;
      color: #38bdf8;
      overflow-x: auto;
      white-space: pre-wrap;
      position: relative;
    }
    .copy-btn {
      position: absolute; right: 0.75rem; top: 0.75rem;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #fff;
      padding: 0.2rem 0.5rem;
      font-size: 0.75rem;
      border-radius: 0.35rem;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="badge"><div class="dot"></div> Production Hybrid TTS Active</div>
      <h1>Unified Hybrid Voice Engine</h1>
      <p class="subtitle">Kokoro-82M (Local Fast) • ElevenLabs (Studio Quality) • Edge-TTS (Neural Free) • Auto-Fallback</p>
    </div>

    <div class="card">
      <div class="section-title">Select TTS Engine</div>
      <div class="engine-grid">
        <div class="engine-btn active" data-engine="edge">
          <div class="engine-title">🇮🇳 Edge-TTS Swara</div>
          <div class="engine-desc">Free & Native Hinglish Girl Voice</div>
        </div>
        <div class="engine-btn" data-engine="elevenlabs">
          <div class="engine-title">💎 ElevenLabs Studio</div>
          <div class="engine-desc">Hyper-Realistic Multilingual V2</div>
        </div>
        <div class="engine-btn" data-engine="kokoro">
          <div class="engine-title">⚡ Kokoro-82M</div>
          <div class="engine-desc">Pure English Only (Local CPU)</div>
        </div>
        <div class="engine-btn" data-engine="auto">
          <div class="engine-title">🪄 Smart Auto</div>
          <div class="engine-desc">Hinglish ➔ Swara, English ➔ Kokoro</div>
        </div>
      </div>

      <div class="section-title">Text to Synthesize</div>
      <div class="textarea-wrapper">
        <textarea id="textInput" placeholder="Apna text yahan likhein ya neeche diye gaye samples click karein...">Namaste! Unified voice server, seamlessly integrate ho chuka hai... Koh-koh-ro ultra-fast audio provide karega, aur ElevenLabs, ultra realistic voice!</textarea>
      </div>

      <div class="chips">
        <div class="chip" onclick="setPrompt('Namaste! Unified voice server, seamlessly integrate ho chuka hai... Koh-koh-ro ultra-fast audio provide karega, aur ElevenLabs, ultra realistic voice!')">🇮🇳 Natural Hinglish (Prosody)</div>
        <div class="chip" onclick="setPrompt('Namaste! Hermes voice engine bilkul ready hai, suniye aur bataiye aawaz kitni natural lag rahi hai.')">🌸 Native Indian Girl Demo</div>
        <div class="chip" onclick="setPrompt('Good morning! Testing our studio voice pipeline with natural breathing and human prosody.')">💎 Studio English Demo</div>
        <div class="chip" onclick="setPrompt('System notice: API response returned status code 200 OK. JSON payload delivered successfully.')">💻 Tech Acronyms Demo</div>
      </div>

      <div class="options-row">
        <div>
          <label class="section-title" style="display:block; margin-bottom:0.4rem;">Voice Profile / ID</label>
          <select id="voiceSelect">
            <option value="" selected>✨ Auto / Engine Recommended Voice</option>
            <optgroup label="🇮🇳 Edge-TTS (Free Neural Voices)">
              <option value="hi-IN-SwaraNeural">Edge: Swara (🇮🇳 Native Hindi/Hinglish Female)</option>
              <option value="hi-IN-MadhurNeural">Edge: Madhur (🇮🇳 Native Hindi Male)</option>
              <option value="en-IN-NeerjaNeural">Edge: Neerja (🇮🇳 Indian English Female)</option>
            </optgroup>
            <optgroup label="💎 ElevenLabs (Hyper-Realistic Studio)">
              <option value="cgSgspJ2msm6clMCkdW9">ElevenLabs: Jessica (🔥 100% Hyper-Realistic Female)</option>
              <option value="EXAVITQu4vr4xnSDxMaL">ElevenLabs: Bella (Warm Female)</option>
              <option value="pNInz6obpgDQGcFmaJgB">ElevenLabs: Adam (Deep Male)</option>
              <option value="JBFqnCBsd6RMkjVDRZzb">ElevenLabs: George (Studio Broadcast Male)</option>
            </optgroup>
            <optgroup label="⚡ Kokoro-82M (Local CPU English)">
              <option value="af_bella">Kokoro: af_bella (Fast Local Female)</option>
              <option value="af_sarah">Kokoro: af_sarah (Fast Calm Female)</option>
            </optgroup>
          </select>
        </div>
        <div>
          <label class="section-title" style="display:block; margin-bottom:0.4rem;">Custom Voice ID / Seed</label>
          <input type="text" id="customVoiceInput" placeholder="Optional: custom voice ID or seed" />
        </div>
      </div>

      <button id="btnGenerate" class="btn-generate" onclick="generateSpeech()">
        <span id="btnIcon">▶</span> <span id="btnText">Synthesize Speech</span>
      </button>

      <div id="playerCard" class="player-card">
        <div class="meta-bar">
          <div>Engine: <span id="usedEngineBadge" class="badge-engine">Kokoro</span></div>
          <div>Latency: <span id="latencySpan" style="color:#38bdf8; font-weight:600;">-- ms</span></div>
        </div>
        <audio id="audioPlayer" controls></audio>
      </div>
    </div>

    <div class="card">
      <div class="section-title">Quick cURL API Example</div>
      <div class="snippet-box">
        <button class="copy-btn" onclick="copyCurl()">Copy</button>
<span id="curlCode">curl -X POST "http://localhost:8005/synthesize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world from Unified Voice Server", "engine": "auto"}' \
     --output speech.mp3</span>
      </div>
    </div>
  </div>

  <script>
    let activeEngine = 'edge';

    // Engine Selection
    document.querySelectorAll('.engine-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.engine-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activeEngine = btn.getAttribute('data-engine');
        updateCurl();
      });
    });

    function setPrompt(text) {
      document.getElementById('textInput').value = text;
      updateCurl();
    }

    function updateCurl() {
      const text = document.getElementById('textInput').value.replace(/'/g, "");
      const port = window.location.port || '8005';
      const format = (activeEngine === 'elevenlabs' || activeEngine === 'edge') ? 'speech.mp3' : 'speech.wav';
      const snippet = `curl -X POST "http://localhost:${port}/synthesize" \\\n     -H "Content-Type: application/json" \\\n     -d '{"text": "${text.substring(0, 50)}...", "engine": "${activeEngine}"}' \\\n     --output ${format}`;
      document.getElementById('curlCode').innerText = snippet;
    }

    function copyCurl() {
      const text = document.getElementById('curlCode').innerText;
      navigator.clipboard.writeText(text);
      alert('cURL command copied to clipboard!');
    }

    async function generateSpeech() {
      const text = document.getElementById('textInput').value.trim();
      if (!text) {
        alert("Please enter text to synthesize.");
        return;
      }

      const customVoice = document.getElementById('customVoiceInput').value.trim();
      const selectedVoice = customVoice || document.getElementById('voiceSelect').value;

      const btn = document.getElementById('btnGenerate');
      const btnText = document.getElementById('btnText');
      const playerCard = document.getElementById('playerCard');
      const audioPlayer = document.getElementById('audioPlayer');
      const usedBadge = document.getElementById('usedEngineBadge');
      const latencySpan = document.getElementById('latencySpan');

      btn.disabled = true;
      btnText.innerText = "Synthesizing...";

      const t0 = performance.now();

      try {
        const streamUrl = `/synthesize_stream?text=${encodeURIComponent(text)}&engine=${encodeURIComponent(activeEngine)}&voice_id=${encodeURIComponent(selectedVoice || '')}&_t=${Date.now()}`;
        playerCard.style.display = 'block';
        usedBadge.innerText = activeEngine.toUpperCase();
        latencySpan.innerHTML = `<span style="color:#38bdf8;">Connecting stream...</span>`;
        
        let playbackTriggered = false;
        
        audioPlayer.onplaying = () => {
          if (!playbackTriggered) {
            playbackTriggered = true;
            const ttfb = Math.round(performance.now() - t0);
            latencySpan.innerHTML = `${ttfb} ms <span style="color:#4ade80; font-weight:700;">(⚡ Ultra-Fast Streaming)</span>`;
            btn.disabled = false;
            btnText.innerText = "Synthesize Speech";
          }
        };

        audioPlayer.src = streamUrl;
        const playPromise = audioPlayer.play();
        if (playPromise !== undefined) {
          playPromise.catch(async () => {
            // If browser blocks direct stream autoplay or fails, fall back to standard POST
            const resp = await fetch('/synthesize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text, engine: activeEngine, voice_id: selectedVoice || null })
            });
            const blob = await resp.blob();
            const serverMs = resp.headers.get('X-Response-Time-Ms');
            const hit = resp.headers.get('X-Cache-Hit') === 'true';
            audioPlayer.src = URL.createObjectURL(blob);
            audioPlayer.play();
            latencySpan.innerHTML = `${serverMs || Math.round(performance.now() - t0)} ms ${hit ? '<span style="color:#4ade80; font-weight:700;">(⚡ Cache Hit)</span>' : ''}`;
            btn.disabled = false;
            btnText.innerText = "Synthesize Speech";
          });
        }
      } catch (err) {
        alert(`Error: ${err.message}`);
        btn.disabled = false;
        btnText.innerText = "Synthesize Speech";
      }
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860 if os.getenv("SPACE_ID") else 8006))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
