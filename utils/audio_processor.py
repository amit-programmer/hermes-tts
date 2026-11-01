import io
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

PERSONA_PRESETS = {
    "code_review": {"rate": "+12%", "pitch": "+0Hz"},
    "serious": {"rate": "+10%", "pitch": "-1Hz"},
    "alert": {"rate": "+15%", "pitch": "+4Hz"},
    "empathetic": {"rate": "-4%", "pitch": "-1Hz"},
    "explain": {"rate": "-6%", "pitch": "+0Hz"},
    "casual_chat": {"rate": "+2%", "pitch": "+3Hz"},
    "conversational": {"rate": "+2%", "pitch": "+3Hz"},
    "story": {"rate": "-5%", "pitch": "-2Hz"},
    "fast": {"rate": "+25%", "pitch": "+0Hz"},
    "slow": {"rate": "-15%", "pitch": "+0Hz"},
}


def apply_persona_preset(
    persona: Optional[str] = None,
    rate: Optional[str] = "+0%",
    pitch: Optional[str] = "+0Hz",
) -> Tuple[str, str]:
    """
    Applies task-based persona pace and pitch modifications if custom values aren't explicitly passed.
    """
    if not persona:
        return rate or "+0%", pitch or "+0Hz"

    preset = PERSONA_PRESETS.get(persona.lower().strip())
    if not preset:
        return rate or "+0%", pitch or "+0Hz"

    final_rate = preset["rate"] if (not rate or rate == "+0%") else rate
    final_pitch = preset["pitch"] if (not pitch or pitch == "+0Hz") else pitch
    return final_rate, final_pitch


def normalize_audio_lufs(audio_data: np.ndarray, target_db: float = -16.0) -> np.ndarray:
    """
    Normalizes audio array to a standard RMS / LUFS level (-16 dBFS).
    Prevents loud/soft volume jumps between sentences and different TTS engines.
    Includes crash guard for very short or zero-length audio.
    """
    if audio_data.size == 0 or len(audio_data) < 240:
        return audio_data

    # Calculate Root Mean Square (RMS)
    rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
    if rms <= 1e-7 or np.isnan(rms):
        return audio_data

    current_db = 20.0 * np.log10(rms)
    gain_db = target_db - current_db
    # Limit gain to reasonable bounds to prevent boosting noise floor
    gain_db = max(-12.0, min(12.0, gain_db))
    factor = 10.0 ** (gain_db / 20.0)

    normalized = audio_data * factor
    # Clip to avoid digital clipping / distortion
    peak = np.max(np.abs(normalized))
    if peak > 0.99 and peak > 0:
        normalized = (normalized / peak) * 0.99

    return normalized


def apply_micro_fade(audio_data: np.ndarray, sample_rate: int = 24000, fade_ms: float = 4.0) -> np.ndarray:
    """
    Applies a smooth micro-fade-in and fade-out (2-5ms) to prevent audio clicks/pops 
    at sentence boundaries and cut-points.
    Includes crash guard for single-word / ultra-short sentences (< 100ms, e.g. 'Sure.', 'Done!').
    """
    if audio_data.size == 0:
        return audio_data

    # Crash Guard: For short audio (<100ms), avoid micro-fading to prevent array errors or truncation
    if len(audio_data) < int(sample_rate * 0.10):
        return audio_data

    fade_samples = int(sample_rate * (fade_ms / 1000.0))
    if fade_samples <= 0 or len(audio_data) < fade_samples * 2:
        return audio_data

    # Smooth half-hanning cosine window for clickless transition
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples, dtype=np.float32)))
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples, dtype=np.float32)))

    audio_data[:fade_samples] *= fade_in
    audio_data[-fade_samples:] *= fade_out
    return audio_data


def strip_mp3_id3_headers(mp3_bytes: bytes) -> bytes:
    """
    Strips ID3v2 metadata headers and seeks directly to the first MPEG sync frame.
    Prevents concatenated frame header decode errors in browser HTML5 <audio> / Web Audio API players.
    """
    if not mp3_bytes or len(mp3_bytes) < 10:
        return mp3_bytes

    # ID3v2 header starts with 'ID3' (0x49 0x44 0x33)
    if mp3_bytes.startswith(b"ID3"):
        # Header length: 10 bytes + 4 synchsafe integer size bytes at offset 6..9
        size_bytes = mp3_bytes[6:10]
        tag_size = (
            ((size_bytes[0] & 0x7F) << 21)
            | ((size_bytes[1] & 0x7F) << 14)
            | ((size_bytes[2] & 0x7F) << 7)
            | (size_bytes[3] & 0x7F)
        )
        total_header_size = 10 + tag_size
        # Extended footer flag (bit 4 of byte 5)
        if mp3_bytes[5] & 0x10:
            total_header_size += 10

        if len(mp3_bytes) > total_header_size:
            mp3_bytes = mp3_bytes[total_header_size:]

    # Seek forward to the first MPEG sync frame (0xFF with top 3 bits of next byte set)
    sync_idx = 0
    while sync_idx < len(mp3_bytes) - 1:
        if mp3_bytes[sync_idx] == 0xFF and (mp3_bytes[sync_idx + 1] & 0xE0) == 0xE0:
            return mp3_bytes[sync_idx:]
        sync_idx += 1

    return mp3_bytes


def stitch_audio_chunks(chunks: list, sample_rate: int = 24000) -> bytes:
    """
    Seamlessly stitches multiple sentence audio chunks into a single continuous stream
    with micro-fading between sentences to completely eliminate pops/clicks.
    """
    if not chunks:
        return b""
    if len(chunks) == 1:
        return chunks[0]

    decoded_arrays = []
    sr = sample_rate

    for c in chunks:
        if not c:
            continue
        try:
            in_buf = io.BytesIO(c)
            d, s = sf.read(in_buf, dtype="float32")
            sr = s
            # Apply micro-fade to sentence boundary if duration permits
            if len(d) >= int(sr * 0.10):
                d = apply_micro_fade(d, sample_rate=sr, fade_ms=4.0)
            decoded_arrays.append(d)
        except Exception:
            # Fallback to direct pass-through
            pass

    if not decoded_arrays:
        return b"".join(chunks)

    # Concatenate smoothly
    combined = np.concatenate(decoded_arrays, axis=0)
    out_buf = io.BytesIO()
    sf.write(out_buf, combined, sr, format="WAV", subtype="PCM_16")
    out_buf.seek(0)
    return out_buf.getvalue()


def convert_audio_format(
    raw_audio_bytes: bytes,
    target_format: str = "mp3",
    source_format: str = "auto",
    sample_rate: int = 24000,
    normalize_volume: bool = True,
    apply_fade: bool = True,
) -> Tuple[bytes, str]:
    """
    Converts audio bytes in-memory across formats (MP3, OGG/Opus, WAV, PCM) using soundfile.
    Applies micro-fading and LUFS volume leveling.
    Returns (converted_bytes, media_type).
    """
    fmt = (target_format or "mp3").lower().strip()
    media_types = {
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    media_type = media_types.get(fmt, "audio/mpeg")

    if not raw_audio_bytes:
        return b"", media_type

    try:
        in_buf = io.BytesIO(raw_audio_bytes)
        data, sr = sf.read(in_buf, dtype="float32")

        # Crash Guard: Skip fade and LUFS for single-word / short bursts (< 100ms)
        is_short = len(data) < int(sr * 0.10)

        if apply_fade and not is_short:
            data = apply_micro_fade(data, sample_rate=sr, fade_ms=4.0)

        if normalize_volume and not is_short:
            data = normalize_audio_lufs(data, target_db=-16.0)

        out_buf = io.BytesIO()
        if fmt in ["ogg", "opus"]:
            try:
                sf.write(out_buf, data, sr, format="OGG", subtype="OPUS")
            except Exception:
                sf.write(out_buf, data, sr, format="OGG", subtype="VORBIS")
        elif fmt == "wav":
            sf.write(out_buf, data, sr, format="WAV", subtype="PCM_16")
        elif fmt == "pcm":
            sf.write(out_buf, data, sr, format="RAW", subtype="PCM_16")
        else:  # mp3
            sf.write(out_buf, data, sr, format="MP3")

        out_buf.seek(0)
        return out_buf.getvalue(), media_type

    except Exception as e:
        return raw_audio_bytes, media_type
