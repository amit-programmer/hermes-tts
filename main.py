import argparse
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from config import VOICES, DEFAULT_VOICE, OUTPUT_DIR, KOKORO_VOICES
from utils.text_cleaner import clean_text_for_tts
from core.kokoro_engine import KokoroEngine
from core.chattts_engine import ChatTTSEngine
from core.elevenlabs_engine import ElevenLabsEngine
from core.edge_engine import EdgeTTSEngine
from tts_manager import EdgeTTSManager


async def run_cli():
    parser = argparse.ArgumentParser(
        description="Unified Hybrid TTS CLI (Kokoro-82M, ChatTTS, ElevenLabs, Edge-TTS)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Namaste! Hybrid TTS setup successfully complete ho gaya hai. Kokoro aur ElevenLabs dono ready hain.",
        help="Text to convert into speech",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=["auto", "kokoro", "chattts", "elevenlabs", "edge"],
        help="TTS engine: kokoro, chattts, elevenlabs, edge, or auto (default: auto)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice identifier or ChatTTS seed (Default depends on engine)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path where output audio will be saved",
    )
    parser.add_argument(
        "--rate",
        type=str,
        default="+0%",
        help="Speech speed modifier (Edge-TTS only, e.g. +10%%, -15%%)",
    )
    parser.add_argument(
        "--pitch",
        type=str,
        default="+0Hz",
        help="Pitch adjustment (Edge-TTS only, e.g. +5Hz, -5Hz)",
    )
    parser.add_argument(
        "--volume",
        type=str,
        default="+0%",
        help="Volume modifier (Edge-TTS only, e.g. +10%%, -10%%)",
    )
    parser.add_argument(
        "--list-voices",
        type=str,
        nargs="?",
        const="all",
        help="List available voices. Pass locale like 'hi-IN' or 'en-IN' to filter.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Demonstrate in-memory streaming instead of direct file save (Edge-TTS)",
    )

    args = parser.parse_args()
    manager = EdgeTTSManager()

    # Handle voice listing
    if args.list_voices:
        print("\n--- Curated Kokoro Voices ---")
        for k, v in KOKORO_VOICES.items():
            print(f"  {k:<12} : {v}")

        locale = None if args.list_voices == "all" else args.list_voices
        print(f"\n--- Edge-TTS Voices (Filter: {locale or 'ALL'}) ---")
        voices = await manager.get_available_voices(locale_filter=locale)
        print(f"{'Short Name':<28} | {'Gender':<8} | {'Locale':<10} | {'Friendly Name'}")
        print("-" * 80)
        for v in voices[:20]:
            print(
                f"{v.get('ShortName', ''):<28} | "
                f"{v.get('Gender', ''):<8} | "
                f"{v.get('Locale', ''):<10} | "
                f"{v.get('FriendlyName', '')}"
            )
        if len(voices) > 20:
            print(f"... and {len(voices) - 20} more voices.")
        return

    cleaned_text = clean_text_for_tts(args.text)
    selected_engine = args.engine.lower()

    # Determine output filename if not provided
    ext = ".mp3" if selected_engine in ["elevenlabs", "edge"] else ".wav"
    out_file = Path(args.output) if args.output else OUTPUT_DIR / f"cli_output{ext}"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Unified Hybrid TTS Speech Generation")
    print("=" * 60)
    print(f"Engine: {selected_engine.upper()}")
    print(f"Voice:  {args.voice or 'default'}")
    print(f"Output: {out_file}")
    print(f"Text:   {cleaned_text}")
    print("-" * 60)

    # 1. Kokoro
    if selected_engine == "kokoro":
        try:
            engine = KokoroEngine(default_voice=args.voice or os.getenv("KOKORO_VOICE", "af_bella"))
            buf = engine.generate(cleaned_text, voice=args.voice)
            with open(out_file, "wb") as f:
                f.write(buf.getvalue())
            print(f"[DONE] Kokoro audio saved to: {out_file}")
            return
        except Exception as e:
            print(f"[WARN] Kokoro failed: {e}. Falling back to Edge-TTS.")
            selected_engine = "edge"

    # 2. ChatTTS
    elif selected_engine == "chattts":
        try:
            seed = int(args.voice) if args.voice and args.voice.isdigit() else int(os.getenv("CHATTTS_SEED", 2222))
            engine = ChatTTSEngine(seed=seed)
            buf = engine.generate(cleaned_text)
            with open(out_file, "wb") as f:
                f.write(buf.getvalue())
            print(f"[DONE] ChatTTS audio saved to: {out_file}")
            return
        except Exception as e:
            print(f"[WARN] ChatTTS failed: {e}. Falling back to Edge-TTS.")
            selected_engine = "edge"

    # 3. ElevenLabs
    elif selected_engine == "elevenlabs":
        try:
            engine = ElevenLabsEngine()
            buf = engine.generate(cleaned_text, voice_id=args.voice)
            with open(out_file, "wb") as f:
                f.write(buf.getvalue())
            print(f"[DONE] ElevenLabs audio saved to: {out_file}")
            return
        except Exception as e:
            print(f"[WARN] ElevenLabs failed: {e}. Falling back to Edge-TTS.")
            selected_engine = "edge"

    # 4. Auto Mode
    elif selected_engine == "auto":
        try:
            engine = KokoroEngine(default_voice=args.voice or os.getenv("KOKORO_VOICE", "af_bella"))
            buf = engine.generate(cleaned_text, voice=args.voice)
            with open(out_file, "wb") as f:
                f.write(buf.getvalue())
            print(f"[DONE] Auto (Kokoro-82M) audio saved to: {out_file}")
            return
        except Exception as e:
            print(f"[WARN] Auto Kokoro failed: {e}. Falling back to Edge-TTS.")
            selected_engine = "edge"

    # Edge-TTS (Direct or Fallback)
    edge_voice = args.voice if args.voice and "Neural" in args.voice else DEFAULT_VOICE
    edge_engine = EdgeTTSEngine(default_voice=edge_voice)
    buf = await edge_engine.generate_async(cleaned_text, voice=edge_voice, rate=args.rate, pitch=args.pitch, volume=args.volume)
    out_file = out_file.with_suffix(".mp3")
    with open(out_file, "wb") as f:
        f.write(buf.getvalue())
    print(f"[DONE] Edge-TTS audio saved to: {out_file}")


if __name__ == "__main__":
    asyncio.run(run_cli())
