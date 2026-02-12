"""
Project Gutenberg Corpus Downloader and Processor

Downloads a curated collection of classic English literature from Project Gutenberg
for pre-training. The corpus focuses on:
- Early Modern English (Shakespeare-era plays, poetry)
- Classic English fiction (Dickens, Austen, Melville)
- Epic poetry and religious texts (King James Bible, Homer, Virgil translations)

This gives the model broad English language understanding before fine-tuning on Shakespeare.
"""

import urllib.request
import re
from pathlib import Path
from typing import Optional

import config


# ============================================================================
# GUTENBERG CORPUS — curated for literary/archaic English pre-training
# ============================================================================
GUTENBERG_TEXTS = {
    # ---- Religious / archaic ----
    "King James Bible": "https://www.gutenberg.org/cache/epub/10/pg10.txt",

    # ---- Epic poetry (translations into English verse) ----
    "Paradise Lost - Milton": "https://www.gutenberg.org/cache/epub/26/pg26.txt",
    "Canterbury Tales - Chaucer": "https://www.gutenberg.org/cache/epub/2383/pg2383.txt",
    "Faerie Queene - Spenser": "https://www.gutenberg.org/cache/epub/15272/pg15272.txt",
    "Divine Comedy - Dante": "https://www.gutenberg.org/cache/epub/8800/pg8800.txt",
    "Iliad - Homer": "https://www.gutenberg.org/cache/epub/6130/pg6130.txt",
    "Odyssey - Homer": "https://www.gutenberg.org/cache/epub/3160/pg3160.txt",
    "Beowulf": "https://www.gutenberg.org/cache/epub/16328/pg16328.txt",
    "Aeneid - Virgil": "https://www.gutenberg.org/cache/epub/228/pg228.txt",

    # ---- Elizabethan / Jacobean drama (Shakespeare-era contemporaries) ----
    "Marlowe Plays": "https://www.gutenberg.org/cache/epub/1094/pg1094.txt",
    "Ben Jonson Plays": "https://www.gutenberg.org/cache/epub/5333/pg5333.txt",
    "Donne Poems": "https://www.gutenberg.org/cache/epub/1141/pg1141.txt",

    # ---- Classic English fiction ----
    "Don Quixote - Cervantes": "https://www.gutenberg.org/cache/epub/996/pg996.txt",
    "Moby Dick - Melville": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "Pride and Prejudice - Austen": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "Great Expectations - Dickens": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    "A Tale of Two Cities - Dickens": "https://www.gutenberg.org/cache/epub/98/pg98.txt",
    "War and Peace - Tolstoy": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
    "Les Miserables - Hugo": "https://www.gutenberg.org/cache/epub/135/pg135.txt",
}


def _strip_gutenberg_header_footer(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate header and footer.
    
    Gutenberg texts have standard markers:
    - Header ends with: '*** START OF THE PROJECT GUTENBERG EBOOK ...'
    - Footer starts with: '*** END OF THE PROJECT GUTENBERG EBOOK ...'
    """
    # Strip header
    start_markers = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK",
        r"\*\*\*START OF THE PROJECT GUTENBERG EBOOK",
    ]
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            # Find the end of the line after the marker
            newline_pos = text.find('\n', match.end())
            if newline_pos != -1:
                text = text[newline_pos + 1:]
            break
    
    # Strip footer
    end_markers = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK",
        r"\*\*\*END OF THE PROJECT GUTENBERG EBOOK",
        r"End of the Project Gutenberg EBook",
        r"End of Project Gutenberg",
    ]
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    return text.strip()


def download_gutenberg_corpus(
    cache_path: Optional[Path] = None,
    force_download: bool = False
) -> str:
    """
    Download and concatenate all Gutenberg texts into a single pre-training corpus.
    
    Args:
        cache_path: Path to cache the combined corpus (default: data/gutenberg_corpus.txt)
        force_download: Re-download even if cache exists
    
    Returns:
        Combined corpus text (all Gutenberg texts concatenated)
    """
    if cache_path is None:
        cache_path = config.DATA_DIR / "gutenberg_corpus.txt"
    
    # Return cached version if available
    if not force_download and cache_path.exists():
        print(f"Loading cached Gutenberg corpus from {cache_path}")
        with open(cache_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  Corpus size: {len(text):,} characters ({len(text)/1024/1024:.1f} MB)")
        return text
    
    print("=" * 70)
    print("DOWNLOADING GUTENBERG PRE-TRAINING CORPUS")
    print("=" * 70)
    
    all_texts = []
    total_chars = 0
    failed = []
    
    for name, url in GUTENBERG_TEXTS.items():
        try:
            print(f"  Downloading {name}...", end=" ", flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req, timeout=30)
            raw = response.read().decode('utf-8', errors='replace')
            
            # Strip Gutenberg header/footer
            clean = _strip_gutenberg_header_footer(raw)
            
            chars = len(clean)
            total_chars += chars
            print(f"{chars:,} chars ({chars/1024:.0f} KB)")
            
            all_texts.append(clean)
            
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n  ⚠ Failed to download: {', '.join(failed)}")
    
    # Concatenate all texts with separator
    corpus = "\n\n".join(all_texts)
    
    # Basic cleaning: normalise whitespace, remove excessive blank lines
    corpus = re.sub(r'\n{4,}', '\n\n\n', corpus)  # Max 3 newlines in a row
    corpus = re.sub(r'[ \t]+', ' ', corpus)         # Collapse spaces/tabs
    
    print(f"\n  Total corpus: {len(corpus):,} characters ({len(corpus)/1024/1024:.1f} MB)")
    print(f"  Texts included: {len(all_texts)}/{len(GUTENBERG_TEXTS)}")
    
    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(corpus)
    print(f"  Saved to {cache_path}")
    
    return corpus


def download_shakespeare() -> str:
    """
    Download the Complete Works of Shakespeare (same as data_loader.py).
    Used separately for fine-tuning after pre-training on Gutenberg.
    """
    from data_loader import download_shakespeare as _download
    return _download()


# ============================================================================
# For quick testing
# ============================================================================
if __name__ == "__main__":
    corpus = download_gutenberg_corpus()
    print(f"\nCorpus stats:")
    print(f"  Characters: {len(corpus):,}")
    print(f"  Estimated words: {len(corpus.split()):,}")
    print(f"  Estimated BPE tokens (≈4 chars/token): {len(corpus)//4:,}")
    
    # Show first 500 chars
    print(f"\nFirst 500 characters:")
    print(corpus[:500])
