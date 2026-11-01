import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Server Configuration (Default 7860 for Hugging Face Spaces, 8006 for local dev)
PORT = int(os.getenv("PORT", 7860 if os.getenv("SPACE_ID") else 8006))
HOST = os.getenv("HOST", "0.0.0.0")

# Engine Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_KEY_2 = os.getenv("ELEVENLABS_API_KEY_2")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_bella")
CHATTTS_SEED = int(os.getenv("CHATTTS_SEED", 2222))
EDGE_DEFAULT_VOICE = os.getenv("EDGE_DEFAULT_VOICE", "hi-IN-SwaraNeural")

# Curated Voice IDs for Kokoro
KOKORO_VOICES = {
    "af_bella": "American Female - Bella (Clean young friendly)",
    "af_sarah": "American Female - Sarah (Professional calm)",
    "af_nicole": "American Female - Nicole (Warm storytelling)",
    "af_sky": "American Female - Sky (Energetic bright)",
    "am_adam": "American Male - Adam (Tech narrator)",
    "am_michael": "American Male - Michael (Deep authoritative)",
    "bf_emma": "British Female - Emma (Posh sophisticated)",
    "bm_george": "British Male - George (Classic British)",
}

# Curated Voice IDs for Edge-TTS
EDGE_VOICES = {
    "hindi_female": "hi-IN-SwaraNeural",
    "hindi_male": "hi-IN-MadhurNeural",
    "indian_english_female": "en-IN-NeerjaNeural",
    "indian_english_male": "en-IN-PrabhatNeural",
    "us_english_jenny": "en-US-JennyNeural",
    "us_english_aria": "en-US-AriaNeural",
    "multilingual_ava": "en-US-AvaMultilingualNeural",
    "multilingual_andrew": "en-US-AndrewMultilingualNeural",
    "us_english_guy": "en-US-GuyNeural",
}

# Legacy alias for backward compatibility
VOICES = EDGE_VOICES
DEFAULT_VOICE = EDGE_DEFAULT_VOICE
DEFAULT_RATE = "+0%"
DEFAULT_PITCH = "+0Hz"
DEFAULT_VOLUME = "+0%"
