"""
Configuration settings for Word-Level Shakespeare Text Generation
"""

import torch
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA SETTINGS
# =============================================================================
# Choose tokenizer: "word" (word-level with FastText) or "bpe" (byte-pair encoding)
TOKENIZER_TYPE = "bpe"  # Options: "word", "bpe"
BPE_VOCAB_SIZE = 5000   # BPE vocabulary size (smaller = lower PPL, more subwords)

# Choose dataset size: "tiny" (~1MB), "full" (~5.5MB complete works)
DATASET_SIZE = "full"  # Options: "tiny", "full"

# Dataset URLs
SHAKESPEARE_URLS = {
    # Tiny Shakespeare (~1MB, ~200K tokens) - good for quick testing
    "tiny": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    # Complete Works of Shakespeare from Project Gutenberg (~5.5MB, ~900K tokens)
    "full": "https://www.gutenberg.org/cache/epub/100/pg100.txt",
}

SHAKESPEARE_URL = SHAKESPEARE_URLS[DATASET_SIZE]
MIN_WORD_FREQUENCY = 3  # Increased since larger vocab (was 2)
MAX_VOCAB_SIZE = 15000  # Increased for larger corpus (was 10000)
MAX_SEQ_LENGTH = 64     # Maximum sequence length in words

# =============================================================================
# EMBEDDING SETTINGS
# =============================================================================
# Using gensim pre-trained vectors (fasttext-wiki-news-subwords-300 or glove-wiki-gigaword-300)
EMBEDDING_DIM = 300               # Embedding dimension (300 for both word & BPE for fair comparison)
FREEZE_EMBEDDINGS = False         # Whether to freeze during training
# BPE uses same LR for all params; word-level uses lower LR for pre-trained embeddings
EMBEDDING_LR = 5e-5               # Lower learning rate for pre-trained embeddings (word-level only)

# Semantic anchors: map archaic Shakespeare words to modern equivalents
SEMANTIC_ANCHORS = {
    # Pronouns
    'thou': 'you',
    'thee': 'you',
    'thy': 'your',
    'thine': 'yours',
    'ye': 'you',
    
    # Verbs (archaic conjugations)
    'hath': 'has',
    'doth': 'does',
    'art': 'are',
    'wilt': 'will',
    'shalt': 'shall',
    'wouldst': 'would',
    'shouldst': 'should',
    'canst': 'can',
    'didst': 'did',
    'hast': 'have',
    'dost': 'do',
    'wert': 'were',
    'hadst': 'had',
    'mayst': 'may',
    'couldst': 'could',
    
    # Contractions
    "'tis": 'it_is',
    "tis": 'it_is',
    "'twas": 'it_was',
    "twas": 'it_was',
    "'twill": 'it_will',
    "t'": 'it',
    "o'er": 'over',
    "e'er": 'ever',
    "ne'er": 'never',
    "e'en": 'even',
    
    # Adverbs/Prepositions
    'wherefore': 'why',
    'hither': 'here',
    'thither': 'there',
    'whither': 'where',
    'hence': 'from_here',
    'thence': 'from_there',
    'whence': 'from_where',
    'ere': 'before',
    'oft': 'often',
    'anon': 'soon',
    'betwixt': 'between',
    'methinks': 'i_think',
    'perchance': 'perhaps',
    'prithee': 'please',
    'forsooth': 'indeed',
    'verily': 'truly',
    'nay': 'no',
    'aye': 'yes',
    'yea': 'yes',
    
    # Nouns
    'morrow': 'morning',
    'fortnight': 'two_weeks',
    'sennight': 'week',
    'hark': 'listen',
    
    # Adjectives
    'oft': 'often',
}

# =============================================================================
# TRANSFORMER SETTINGS
# =============================================================================
NUM_LAYERS = 5          # Number of transformer decoder layers (reduced from 6, 4 was too weak)
NUM_HEADS = 6           # Number of attention heads (must divide EMBEDDING_DIM)
FFN_HIDDEN_DIM = 1024   # Feed-forward network hidden dimension (reduced from 1200)
DROPOUT = 0.3           # Dropout probability (increased from 0.25 to reduce overfitting)
ATTENTION_DROPOUT = 0.25 # Attention-specific dropout (increased from 0.2)

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
BATCH_SIZE = 64           # Increased from 32 for faster training and smoother gradients
LEARNING_RATE = 7e-4       # Slightly higher than 5e-4 to compensate for larger batch
WEIGHT_DECAY = 0.05        # Increased from 0.01 for stronger regularization
NUM_EPOCHS = 50
WARMUP_STEPS = 500         # Reduced — fewer batches per epoch now
GRAD_CLIP_NORM = 1.0

# Early stopping
PATIENCE = 10
MIN_DELTA = 0.001

# Learning rate schedule
LR_SCHEDULER = "cosine"  # Options: "cosine", "linear", "constant"

# =============================================================================
# LOSS SETTINGS
# =============================================================================
LABEL_SMOOTHING = 0.15    # Increased from 0.1 for softer targets
SEMANTIC_PRESERVATION_WEIGHT = 0.05  # Weight for anchor preservation loss

# =============================================================================
# GENERATION SETTINGS
# =============================================================================
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9
MAX_GENERATE_LENGTH = 100
REPETITION_PENALTY = 1.2

# =============================================================================
# DEVICE
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# RANDOM SEED
# =============================================================================
SEED = 42

# =============================================================================
# SPECIAL TOKENS
# =============================================================================
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"  # Beginning of sequence
EOS_TOKEN = "<EOS>"  # End of sequence

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
