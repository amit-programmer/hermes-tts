import re
from typing import List


def clean_text_for_tts(raw_text: str) -> str:
    """
    Hermes AI ke technical output (code snippets, markdown, acronyms, URLs)
    ko audio-friendly natural Hinglish/English speech me convert karta hai.
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Multi-line code blocks ko speech-friendly announcement me badalna
    text = re.sub(
        r"```[\w]*\n[\s\S]*?\n```",
        " [Code snippet generated, check screen.] ",
        text,
    )
    # Inline triple backticks without newlines
    text = re.sub(r"```[\s\S]*?```", " [Code snippet generated.] ", text)

    # 2. Markdown tables (| col1 | col2 |) ko simplify karna
    text = re.sub(r"\|[^\n]+\|\n\|[-:\s|]+\|\n(\|[^\n]+\|\n?)*", " [Table data displayed on screen.] ", text)

    # 3. Inline code (`variable_name`) ke backticks hatana
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # 4. Markdown URLs [text](url) me se sirf display text rakhna
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # 5. Raw URLs (https://example.com/xyz) ko readable banana
    text = re.sub(r"https?://\S+", "link", text)

    # 6. Markdown headers (# Header) aur blockquotes (> quote) ko clean karna
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s*", "", text, flags=re.MULTILINE)

    # 7. Bold / Italic / Strikethrough markers (*, _, ~) hatana
    text = re.sub(r"[*_~]{1,3}", "", text)

    # 8. Technical common acronyms ko pronunciation ke liye stretch / phonetic banana
    replacements = {
        "API": "A P I",
        "APIs": "A P Is",
        "JSON": "Jason",
        "SQL": "Sequel",
        "NoSQL": "No Sequel",
        "PR": "P R",
        "PRs": "P Rs",
        "CI/CD": "C I C D",
        "HF": "Hugging Face",
        "DB": "Data base",
        "UI": "U I",
        "UX": "U X",
        "URL": "U R L",
        "CLI": "C L I",
        "TTS": "T T S",
        "LLM": "L L M",
        "LLMs": "L L Ms",
        "RAM": "Ram",
        "GPU": "G P U",
        "GPUs": "G P Us",
        "CPU": "C P U",
        "HTTP": "H T T P",
        "HTTPS": "H T T P S",
        "SDK": "S D K",
        "repo": "repository",
        "regex": "regular expression",
    }
    for word, replacement in replacements.items():
        text = re.sub(rf"\b{re.escape(word)}\b", replacement, text)

    # 9. Extra spaces aur blank lines trim karna
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """
    Agar Hermes bada output deta hai, toh sentences ke basis par
    text ko safe chunks me divide karta hai taaki latency kam rahe.
    """
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        return []

    if len(clean_text) <= max_chars:
        return [clean_text]

    # Split on sentence boundaries (. ! ?)
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) + 1 <= max_chars:
            current_chunk.append(sentence)
            current_len += len(sentence) + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = len(sentence)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
