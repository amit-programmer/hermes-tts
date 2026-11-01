"""
Voice Profiles & Acoustic Configurations for Edge-TTS Engine
"""

VOICE_PROFILES = {
    # Natural Female Hinglish/Hindi (Swara - Recommended Default)
    "hinglish_female": {
        "voice": "hi-IN-SwaraNeural",
        "rate": "+2%",      # Natural fast pace suited for tech conversation
        "pitch": "+0Hz",
        "volume": "+0%",
        "description": "Best for Devanagari Hindi & Roman Hinglish (Female)",
    },
    # Natural Male Hindi/Hinglish (Madhur)
    "hinglish_male": {
        "voice": "hi-IN-MadhurNeural",
        "rate": "+2%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "description": "Natural Indian Hindi/Hinglish (Male)",
    },
    # Professional Neutral Indian English Female (Neerja)
    "english_in_female": {
        "voice": "en-IN-NeerjaNeural",
        "rate": "+5%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "description": "Professional Indian English (Female)",
    },
    # Professional Neutral Indian English Male (Prabhat)
    "english_in_male": {
        "voice": "en-IN-PrabhatNeural",
        "rate": "+5%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "description": "Professional Indian English (Male)",
    },
    # Western Tech Persona Female (Jenny)
    "english_us_female": {
        "voice": "en-US-JennyNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "description": "Standard US English Female",
    },
    # Western Tech Persona Female (Aria)
    "english_us_aria": {
        "voice": "en-US-AriaNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "description": "Expressive US English Female",
    },
    # Western Tech Persona Male (Guy)
    "english_us_male": {
        "voice": "en-US-GuyNeural",
        "rate": "+0%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "description": "Standard US English Male",
    },
}

DEFAULT_PROFILE = "hinglish_female"
