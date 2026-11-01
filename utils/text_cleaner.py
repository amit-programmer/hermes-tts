import re
from typing import List

HINGLISH_MAP = {
    r"\bbhai\b": "bhaay",
    r"\bkya\b": "kyaa",
    r"\bkaro\b": "karro",
    r"\bkuch\b": "kooch",
    r"\btheek\b": "theek",
    r"\baaj\b": "aaj",
    r"\bkaise\b": "kaisey",
    r"\blog\b": "log",
    r"\bhi\b": "hee",
}


def normalize_hinglish_text(text: str) -> str:
    """Lightweight phonetic normalization for Romanized Hinglish input."""
    if not text:
        return ""
    normalized = text
    for pattern, replacement in HINGLISH_MAP.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}


def ordinal_day(d: int) -> str:
    if 11 <= (d % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    return f"{d}{suffix}"


def expand_numbers_and_currency(text: str) -> str:
    """
    Intelligently expands currency, percentages, dates, status codes, hex values, software versions, and standalone numbers/OTPs.
    - Dates: 2026-11-15 -> 15th November 2026, 05/12/2026 -> 5th December 2026
    - Status codes: HTTP 404 -> four oh four, 500 -> five hundred
    - Hex codes: 0x7FFE -> Hex 0 x 7 F F E
    - Negative values: -5.4% -> minus 5.4 percent
    - Currency: $500 -> 500 dollars, ₹1200 -> 1200 rupees
    - Percentage: 99.9% -> 99.9 percent
    - Version: v2.4.0 -> version 2 point 4 point 0
    - Large codes/OTPs (5+ continuous digits): 98102 -> 9 8 1 0 2 (digit-by-digit)
    """
    if not text:
        return ""
    expanded = text

    # 1. Dates (ISO 2026-11-15 or Slash 15/11/2026)
    def replace_date_iso(match):
        y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{ordinal_day(d)} {MONTH_NAMES[m]} {y}"
        return match.group(0)

    def replace_date_slash(match):
        d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{ordinal_day(d)} {MONTH_NAMES[m]} {y}"
        if 1 <= d <= 12 and 1 <= m <= 31:
            return f"{ordinal_day(m)} {MONTH_NAMES[d]} {y}"
        return match.group(0)

    expanded = re.sub(r"\b(\d{4})-(0[1-9]|1[0-2])-([0-2]\d|3[01])\b", replace_date_iso, expanded)
    expanded = re.sub(r"\b([0-2]?\d|3[01])/(0[1-9]|1[0-2])/(\d{4})\b", replace_date_slash, expanded)

    # 2. Version numbers: v1.2.3, ver 2.4.0, version 3.0.1 (or 3-part numbers like 1.2.3)
    def replace_version(match):
        v_num = match.group(1)
        digits_with_point = " point ".join(v_num.split("."))
        return f"version {digits_with_point}"

    # Match explicit version prefixes (e.g. v2.4, version 1.0) or semantic 3-part versioning (1.2.3)
    expanded = re.sub(r"\b(?:v|ver|version\s+)(\d+\.\d+(?:\.\d+)*)\b", replace_version, expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\b(\d{1,2}\.\d{1,2}\.\d{1,3})\b", replace_version, expanded)

    # 3. Currency expansion (Rupees, Dollars, Euros, Pounds)
    expanded = re.sub(r"₹\s*(\d+(?:\.\d+)?)", r"\1 rupees", expanded)
    expanded = re.sub(r"\$\s*(\d+(?:\.\d+)?)", r"\1 dollars", expanded)
    expanded = re.sub(r"€\s*(\d+(?:\.\d+)?)", r"\1 euros", expanded)
    expanded = re.sub(r"£\s*(\d+(?:\.\d+)?)", r"\1 pounds", expanded)

    # 4. Negative numbers: -5.4%, -10
    expanded = re.sub(r"(^|\s)-(\d+(?:\.\d+)?)\b", r"\1minus \2", expanded)

    # 5. Time formats: 15:30 -> 3:30 PM, 14:00 -> 2:00 PM, 09:15 -> 9:15 AM
    def replace_time_24h(match):
        h, m = int(match.group(1)), match.group(2)
        period = "AM" if h < 12 else "PM"
        h12 = h if (h == 12 or h == 0) else (h % 12)
        if h == 0:
            h12 = 12
        return f"{h12}:{m} {period}"

    expanded = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*(IST|UTC|GMT|PST|EST))?\b", replace_time_24h, expanded)

    # 6. Tech durations & memory units (ms, sec, KB, MB, GB)
    expanded = re.sub(r"\b(\d+(?:\.\d+)?)\s*ms\b", r"\1 milliseconds", expanded)
    expanded = re.sub(r"\b(\d+(?:\.\d+)?)\s*secs?\b", r"\1 seconds", expanded)
    expanded = re.sub(r"\b(\d+(?:\.\d+)?)\s*mins?\b", r"\1 minutes", expanded)
    expanded = re.sub(r"\b(\d+(?:\.\d+)?)\s*hrs?\b", r"\1 hours", expanded)
    expanded = re.sub(r"\b(\d+(?:\.\d+)?)\s*KB\b", r"\1 kilobytes", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\b(\d+(?:\.\d+)?)\s*MB\b", r"\1 megabytes", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\b(\d+(?:\.\d+)?)\s*GB\b", r"\1 gigabytes", expanded, flags=re.IGNORECASE)

    # 7. Percentages: 50% -> 50 percent
    expanded = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", expanded)

    # 8. HTTP & Error status codes: 404 -> four oh four, 500 error -> five hundred error
    def replace_status_codes(match):
        prefix = match.group(1) or ""
        code = match.group(2)
        code_spoken_map = {
            "404": "four oh four",
            "500": "five hundred",
            "502": "five oh two",
            "503": "five oh three",
            "401": "four oh one",
            "403": "four oh three",
            "200": "two hundred",
            "201": "two oh one",
        }
        spoken = code_spoken_map.get(code, code)
        return f"{prefix}{spoken}"

    expanded = re.sub(
        r"\b(HTTP\s+|status\s+|error\s+|code\s+)(404|500|502|503|401|403)\b",
        replace_status_codes,
        expanded,
        flags=re.IGNORECASE,
    )
    expanded = re.sub(
        r"\b(500|502|503|401|403)\s+(error|server error|bad gateway|service unavailable)\b",
        lambda m: f"{'five hundred' if m.group(1) == '500' else ('five oh two' if m.group(1) == '502' else 'five oh three')} {m.group(2)}",
        expanded,
        flags=re.IGNORECASE,
    )
    expanded = re.sub(r"\b(404)\b", "four oh four", expanded)

    # 9. Hexadecimal representations: 0x7FFE, 0x004F3A -> Hex 0 x 7 F F E
    def replace_hex(match):
        hex_digits = match.group(1)
        spaced = " ".join(hex_digits.upper())
        return f"Hex 0 x {spaced}"

    expanded = re.sub(r"\b0x([0-9a-fA-F]{2,8})\b", replace_hex, expanded)

    # 10. Long numbers / OTPs / Phone codes (5 or more digits -> digit-by-digit)
    def replace_otp(match):
        digits = match.group(0)
        return " ".join(digits)

    expanded = re.sub(r"\b\d{5,}\b", replace_otp, expanded)

    return expanded


DEV_TECHNICAL_TERMS = {
    r"\bAPI\b": "A P I",
    r"\bAPIs\b": "A P Is",
    r"\bJSON\b": "Jason",
    r"\bSQL\b": "Sequel",
    r"\bNoSQL\b": "No Sequel",
    r"\bPostgreSQL\b": "Post-gress Sequel",
    r"\bMySQL\b": "My-Sequel",
    r"\bPR\b": "P R",
    r"\bPRs\b": "P Rs",
    r"\bDB\b": "Data base",
    r"\bJWT\b": "J W T",
    r"\bJWTs\b": "J W Ts",
    r"\bUUID\b": "U-U-I-D",
    r"\bUUIDs\b": "U-U-I-Ds",
    r"\bvCPU\b": "v-C-P-U",
    r"\bvCPUs\b": "v-C-P-Us",
    r"\bAWS\b": "A W S",
    r"\bGCP\b": "G C P",
    r"\bCLI\b": "C L I",
    r"\bHF\b": "Hugging Face",
    r"\bCI/CD\b": "C I C D",
    r"\bCI\b": "C I",
    r"\bCD\b": "C D",
    r"\bUI\b": "U I",
    r"\bUX\b": "U X",
    r"\bGUI\b": "Gooey",
    r"\bOAuth\b": "O-Auth",
    r"\bOAuth2\b": "O-Auth 2",
    r"\bURL\b": "U R L",
    r"\bURLs\b": "U R Ls",
    r"\bTTS\b": "T T S",
    r"\bLLM\b": "L L M",
    r"\bLLMs\b": "L L Ms",
    r"\bHTML\b": "H T M L",
    r"\bCSS\b": "C S S",
    r"\bJS\b": "J S",
    r"\bTS\b": "T S",
    r"\bOS\b": "O S",
    r"\bK8s\b": "Kubernetes",
    r"\bk8s\b": "Kubernetes",
    r"\bRAM\b": "Ram",
    r"\bGPU\b": "G P U",
    r"\bGPUs\b": "G P Us",
    r"\bCPU\b": "C P U",
    r"\bHTTP\b": "H T T P",
    r"\bHTTPS\b": "H T T P S",
    r"\bSSH\b": "S S H",
    r"\bDNS\b": "D N S",
    r"\bNPM\b": "N P M",
    r"\bPIP\b": "Pip",
    r"\bTCP\b": "T C P",
    r"\bUDP\b": "U D P",
    r"\bIP\b": "I P",
    r"\bIPs\b": "I Ps",
    r"\bIDE\b": "I D E",
    r"\bIDEs\b": "I D Es",
    r"\bDOM\b": "D O M",
    r"\bCORS\b": "Cors",
    r"\bWASM\b": "Wasm",
    r"\bSDK\b": "S D K",
    r"\brepo\b": "repository",
    r"\brepos\b": "repositories",
    r"\bregex\b": "regular expression",
    r"\bRegex\b": "Regular expression",
    r"\basync/await\b": "async await",
    r"\bi/o\b": "I O",
    r"\bI/O\b": "I O",
    r"\bstdout\b": "standard out",
    r"\bstderr\b": "standard error",
    r"\bstdin\b": "standard in",
    r"O\(n log n\)": "Order of n log n",
    r"O\(n\^2\)": "Order of n squared",
    r"O\(n\)": "Order of n",
    r"O\(1\)": "Order of 1",
    # Phonetic Pronunciation Fixes
    r"\bKokoro\b": "Koh-koh-ro",
    r"\bkokoro\b": "Koh-koh-ro",
    r"\bElevenLabs\b": "Eleven-Labs",
    r"\belevenlabs\b": "Eleven-Labs",
    r"\bHermes\b": "Her-meez",
}


def is_pure_code_snippet(text: str) -> bool:
    """Detects if the entire text is essentially just a raw code block or snippet without commentary."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = re.sub(r"^```[\w]*\n?", "", stripped)
        inner = re.sub(r"```$", "", inner).strip()
        # If it doesn't contain complete sentences
        return not bool(re.search(r"[.!?]\s+[A-Z]", inner))
    return False


def clean_text_for_tts(text: str) -> str:
    """
    Cleans raw text / developer prompt outputs for TTS engines:
    - Replaces pure code snippets with smart spoken summaries
    - Strips robotic AI fillers and conversational verbose preamble
    - Maps emojis into expressive prosody cues and strips extraneous glyphs
    - Converts URLs, localhost, ports, and IP addresses to spoken conversational forms
    - Normalizes multi-line status / key-value blocks to prevent run-on speech
    - Wraps parentheses / side-notes in natural breathing pauses
    - Strips code blocks, tables, markdown formatting, HTML tags
    - Normalizes technical software engineering lexicon (Dev Lexicon Engine)
    - Normalizes punctuation and preserves natural pauses
    """
    if not text:
        return ""

    # Check if the entire response is a solitary code block
    if is_pure_code_snippet(text):
        lang = detect_language_type(text)
        if lang == "hinglish":
            return "Maine updated implementation ka code attach kar diya hai, screen par review kar lo."
        return "I have attached the implementation code on screen for your review."

    cleaned = text

    # 1. Strip verbose conversational AI fillers at start
    ai_fillers = [
        r"^(?:Sure(?: thing)?|Certainly|Of course|Alright)!(?:\s+I(?:'d|\s+would)?\s+be\s+happy\s+to\s+help(?:\s+you)?(?:\s+with\s+that)?\.?)?\s*",
        r"^(?:Sure|Certainly|Of course|Alright),\s+(?:I(?:'d|\s+would)?\s+be\s+happy\s+to\s+help(?:\s+you)?(?:\s+with\s+that)?\.?)\s*",
        r"^As an AI(?: language model)?,?\s*",
        r"^Here is (?:the|your) (?:updated|requested)?\s*(?:code|implementation|solution|explanation):\s*",
    ]
    for pattern in ai_fillers:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # 2. Emojis and emotion markers to natural spoken cues
    emoji_transforms = {
        "🚀": " ",
        "⚠️": "Caution, ",
        "✅": "Success, ",
        "❌": "Error, ",
        "🎉": "Congratulations! ",
        "💡": "Note, ",
        "🔥": "Awesome! ",
        "⚡": "fast ",
        "👍": "Done, ",
    }
    for em, repl in emoji_transforms.items():
        cleaned = cleaned.replace(em, repl)
    # Strip remaining emojis/symbols that could cause TTS stutters
    cleaned = re.sub(r"[\U00010000-\U0010ffff]", "", cleaned)

    # 3. URLs, Ports, Localhost & IP Addresses
    # Localhost with port: localhost:3000 -> localhost on port 3000
    cleaned = re.sub(r"\b(?:https?://)?localhost:(\d+)\b", r"localhost on port \1", cleaned, flags=re.IGNORECASE)

    # IP address with port: http://192.168.1.1:8080 -> IP 192 dot 168 dot 1 dot 1, on port 8080
    def replace_ip_port(match):
        ip = match.group(1)
        port = match.group(2)
        spoken_ip = ip.replace(".", " dot ")
        return f"IP {spoken_ip}, on port {port}"

    cleaned = re.sub(r"\b(?:https?://)?((?:\d{1,3}\.){3}\d{1,3}):(\d+)\b", replace_ip_port, cleaned)

    # Standalone IP address: 192.168.1.1 -> IP 192 dot 168 dot 1 dot 1
    def replace_standalone_ip(match):
        ip = match.group(0)
        spoken_ip = ip.replace(".", " dot ")
        return f"IP {spoken_ip}"

    cleaned = re.sub(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", replace_standalone_ip, cleaned)

    # Domain with port: https://domain.com:8443 -> domain.com on port 8443
    cleaned = re.sub(r"\bhttps?://([a-zA-Z0-9.-]+):(\d+)\b", r"\1 on port \2", cleaned)

    cleaned = normalize_hinglish_text(cleaned)
    cleaned = expand_numbers_and_currency(cleaned)

    # 4. Code blocks & tables to voice-friendly spoken notes
    cleaned = re.sub(
        r"```[\w]*\n[\s\S]*?\n```",
        " [Code snippet attached on screen.] ",
        cleaned,
    )
    cleaned = re.sub(r"```[\s\S]*?```", " [Code snippet attached on screen.] ", cleaned)
    cleaned = re.sub(r"\|[^\n]+\|\n\|[-:\s|]+\|\n(\|[^\n]+\|\n?)*", " [Table data displayed on screen.] ", cleaned)

    # 5. Multi-line Key-Value pairs / Status blocks (e.g. CPU: 45%\nMemory: 3.2 GB) -> CPU, 45 percent. Memory, 3.2 megabytes.
    cleaned = re.sub(r"^([A-Za-z0-9_\s]{2,20}):\s+([^\n]+)$", r"\1, \2.", cleaned, flags=re.MULTILINE)

    # 6. Parentheses & Side-notes: Add conversational pauses around bracketed notes
    cleaned = re.sub(r"\s*\(([^\)]+)\)", r", \1, ", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)

    # 7. Numbered lists & bullet points: "1. Pull the repo. 2. Run npm install" -> "Step 1, Pull the repo. Step 2, Run npm install"
    cleaned = re.sub(r"^\s*(\d+)\.\s+", r"Step \1, ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"(?<=[.!?]\s)(\d+)\.\s+", r"Step \1, ", cleaned)
    cleaned = re.sub(r"^\s*[-*•]\s+", r"Next, ", cleaned, flags=re.MULTILINE)

    # 8. File paths: e.g. "src/auth/jwt_handler.py" -> "src auth jwt_handler file"
    def normalize_filepath(match):
        path = match.group(0)
        parts = [p for p in re.split(r"[/\\_]", path) if p]
        base_name = parts[-1].split(".")[0] if parts else "file"
        folder = " ".join(parts[:-1]) if len(parts) > 1 else ""
        if folder:
            return f"{folder} {base_name} file"
        return f"{base_name} file"

    cleaned = re.sub(r"\b(?:[a-zA-Z0-9_-]+[/\\UI])+[a-zA-Z0-9_.-]+\.[a-zA-Z0-9]{2,4}\b", normalize_filepath, cleaned)
    cleaned = re.sub(r"\b([a-zA-Z0-9_]+)\.(py|ts|js|json|html|css|cpp|rs|go|yaml|yml|md)\b", r"\1 file", cleaned)

    # 9. Git commands & terminal runs: "git commit -m" -> "git commit"
    cleaned = re.sub(r"\bgit\s+commit\s+-m\s+[\"']([^\"']+)[\"']", r"git commit with message \1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bgit\s+(checkout|push|pull|merge|status|add|rebase)\b", r"git \1", cleaned, flags=re.IGNORECASE)

    # 10. Math & Logic operators to natural spoken speech:
    cleaned = re.sub(r"&&", " and ", cleaned)
    cleaned = re.sub(r"\|\|", " or ", cleaned)
    cleaned = re.sub(r"!=", " is not equal to ", cleaned)
    cleaned = re.sub(r"===", " equals exactly ", cleaned)
    cleaned = re.sub(r"==", " equals ", cleaned)
    cleaned = re.sub(r">=", " is greater than or equal to ", cleaned)
    cleaned = re.sub(r"<=", " is less than or equal to ", cleaned)
    cleaned = re.sub(r"(?<!\w)>(?!\w)", " is greater than ", cleaned)
    cleaned = re.sub(r"(?<!\w)<(?!\w)", " is less than ", cleaned)

    # 11. Markdown, HTML tags & formatting stripping
    cleaned = re.sub(r"<[^>]+>", "", cleaned)  # HTML tags
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)  # inline code
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)  # markdown links
    cleaned = re.sub(r"https?://\S+", "link", cleaned)  # raw URLs
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)  # headings
    cleaned = re.sub(r"^\s*>\s*", "", cleaned, flags=re.MULTILINE)  # blockquotes
    cleaned = re.sub(r"[*_~]{1,3}", "", cleaned)  # bold, italics, strikethrough

    # 12. Technical Dev Lexicon Normalization
    for pattern, replacement in DEV_TECHNICAL_TERMS.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    # Preserve ellipses (...) for natural human breathing pauses; collapse 4+ dots
    cleaned = re.sub(r"\.{4,}", "...", cleaned)
    cleaned = re.sub(r"[-_]+", " ", cleaned)
    cleaned = re.sub(r"[ ]+", " ", cleaned).strip()

    if cleaned and cleaned[-1] not in [".", "!", "?", "।"]:
        cleaned += "."

    return cleaned


def sanitize_text(text: str) -> str:
    """Alias for clean_text_for_tts for Hermes API route standard compatibility."""
    return clean_text_for_tts(text)


HINGLISH_KEYWORDS = {
    # Greetings & Politeness
    "namaste", "namaskar", "shukriya", "dhanyavaad", "alvida",
    # Pronouns
    "aap", "aapka", "aapki", "aapke", "aapko", "hum", "humne", "humara", "humari", "humare",
    "maine", "mujhe", "mera", "meri", "mere", "mujhse",
    "tum", "tumhe", "tumhara", "tumhari", "tumhare", "tu", "tujhe", "tera", "teri", "tere",
    "yeh", "ye", "woh", "wo", "inka", "inki", "inke", "unka", "unki", "unke", "isse", "usse",
    # Verbs (Doing / Action)
    "kar", "karo", "kare", "karein", "karna", "karta", "karti", "karte", "kiya", "kiye", "ki",
    "diya", "diye", "di", "dena", "dijiye", "dete", "deta", "deti",
    "liya", "liye", "le", "lo", "lena", "lijiye", "lete", "leta", "leti",
    # Auxiliary & State Verbs
    "hai", "hain", "hoon", "ho", "hoga", "hogi", "hoge", "honge", "hota", "hoti", "hote", "hona",
    "tha", "thi",
    "raha", "rahi", "rahe", "chuka", "chuki", "chuke",
    # Negation
    "nahi", "nahin",
    # Questions
    "kya", "kyaa", "kyu", "kyun", "kaise", "kaisey", "kahan", "kidhar", "kaun", "kitna", "kitne", "kitni", "kab",
    # Conjunctions & Prepositions
    "aur", "ya", "lekin", "magar", "toh", "bhi", "hee", "abhi", "tabhi", "jab", "saath", "liye", "baad", "pehle",
    # Conversational & Slang
    "bhai", "bhaay", "yaar", "dost", "theek", "achha", "accha", "sahi", "galat", "suno", "suniye", "sunte",
    "dekh", "dekho", "dekhiye", "batao", "bataiye", "bol", "bolo", "boliye", "samajh", "samjho",
    "thoda", "thodi", "thode", "zyada", "bohot", "bahut", "bilkul", "zaroor", "kuch", "kooch"
}


def detect_language_type(text: str) -> str:
    """
    Detects whether text is Hinglish/Hindi or pure English.
    - If Devanagari script is present -> 'hinglish'
    - If any Romanized Hindi stop-words are present -> 'hinglish'
    - Otherwise -> 'english'
    """
    if not text:
        return "english"

    # 1. Direct Devanagari Unicode check
    if re.search(r"[\u0900-\u097F]", text):
        return "hinglish"

    # 2. Comprehensive Romanized Hindi token scanner
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return "english"

    hinglish_matches = sum(1 for w in words if w in HINGLISH_KEYWORDS)
    if hinglish_matches >= 1:
        return "hinglish"

    return "english"


def optimize_speech_prosody(text: str) -> str:
    """
    Adds natural breath and punctuation pacing to text for realistic human prosody.
    - Adds natural breathing pause after greetings, acknowledgments, and transitions
    - Softens colons and semicolons with commas for realistic pacing
    - Adds natural pauses before coordinating conjunctions
    """
    if not text:
        return ""
    prosody_text = text

    # 1. Natural pause after greeting words
    prosody_text = re.sub(
        r"\b(Namaste|Namaskar|Hello|Hey|Hi|Good morning|Good evening|Good afternoon)\s*([!.,]?)",
        r"\1! ",
        prosody_text,
        flags=re.IGNORECASE,
    )

    # 2. Natural pause after conversational acknowledgments (Sure, Done, Wait, Got it, Perfect, Alright)
    prosody_text = re.sub(
        r"^(Sure|Done|Wait|Got it|Understood|Perfect|Alright|Okay|OK|Haan|Theek hai)\b\s*([,.]?)",
        r"\1, ",
        prosody_text,
        flags=re.IGNORECASE,
    )

    # 3. Conversational transitions and clause openers (Actually, However, Furthermore, Basically, In fact, Sach bataun toh)
    prosody_text = re.sub(
        r"\b(Actually|However|Furthermore|Basically|In fact|Fortunately|Unfortunately)\b\s*(?![,])",
        r"\1, ",
        prosody_text,
        flags=re.IGNORECASE,
    )

    # 4. Ensure natural breath pause before conjunctions when not already punctuated
    prosody_text = re.sub(
        r"(?<![,.!?;:])\s+\b(aur|and|lekin|but|however|magar|isliye|therefore)\b",
        r", \1",
        prosody_text,
        flags=re.IGNORECASE,
    )

    # 5. Normalize colons followed by clauses into natural pauses (e.g. "Note: this is" -> "Note, this is")
    prosody_text = re.sub(r":\s+", ", ", prosody_text)

    # 6. Question Intonation & Pitch Inflection: Ensure interrogatives have a noticeable pause before answer
    prosody_text = re.sub(r"\?\s*", "? ... ", prosody_text)

    # 7. Paragraph Transitions: Deep breath pause between distinct paragraphs to avoid listening fatigue
    prosody_text = re.sub(r"\n{2,}", " ... \n\n", prosody_text)

    return prosody_text


def split_into_sentences(text: str) -> List[str]:
    """Split text on sentence endings and paragraph breaks, while keeping a word-based fallback for plain text."""
    if not text:
        return []
    raw_sentences = [s.strip() for s in re.split(r"(?:(?<=[.!?])\s+|\n{2,})", text) if s.strip()]
    valid_sentences = []
    for s in raw_sentences:
        # Ignore solitary punctuation or fragments like '...' or '.'
        if re.search(r"[a-zA-Z0-9]", s):
            valid_sentences.append(s)
    if valid_sentences:
        return valid_sentences
    return [text.strip()] if text.strip() else []


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Splits long text across sentence boundaries to keep synthesis latency low.
    """
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return []

    if len(clean_text) <= max_chars:
        return [clean_text]

    sentences = split_into_sentences(clean_text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        if not sentence:
            continue
        if current_len + len(sentence) + 1 <= max_chars:
            current_chunk.append(sentence)
            current_len += len(sentence) + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            if len(sentence) > max_chars:
                words = sentence.split()
                sub_chunk: List[str] = []
                sub_len = 0
                for word in words:
                    if sub_len + len(word) + 1 <= max_chars:
                        sub_chunk.append(word)
                        sub_len += len(word) + 1
                    else:
                        if sub_chunk:
                            chunks.append(" ".join(sub_chunk))
                        sub_chunk = [word]
                        sub_len = len(word)
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
            else:
                current_chunk = [sentence]
                current_len = len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_large_text(text: str, max_chars: int = 1200) -> List[str]:
    """Sentence-safe batching for TTS workloads (avoids unnecessary multi-chunk splits)."""
    return chunk_text(text, max_chars=max_chars)
