"""
Text preprocessing module for NLP tasks.
Handles tokenization, stopword removal, lemmatization, POS tagging, and N-gram generation.
"""

import re
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.util import ngrams
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def _has_nltk_resource(path):
    """Check whether an NLTK resource is available locally."""
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False


HAS_PUNKT = _has_nltk_resource('tokenizers/punkt')
HAS_STOPWORDS = _has_nltk_resource('corpora/stopwords')
HAS_WORDNET = _has_nltk_resource('corpora/wordnet')
HAS_POS_TAGGER = _has_nltk_resource('taggers/averaged_perceptron_tagger') or _has_nltk_resource('taggers/averaged_perceptron_tagger_eng')


POS_TAG_LABELS = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition/subordinating conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'LS': 'List item marker',
    'MD': 'Modal',
    'NN': 'Noun, singular or mass',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'RP': 'Particle',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund/present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb',
    'UNK': 'Unknown (tagger resource unavailable)',
}


def tokenize_text(text):
    """
    Tokenize text into words.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of word tokens
    """
    if text is None:
        return []

    text = str(text)
    if not text.strip():
        return []

    if HAS_PUNKT:
        tokens = word_tokenize(text.lower())
    else:
        # Regex fallback when punkt is unavailable in offline environments.
        tokens = re.findall(r"\b\w+\b", text.lower())
    return tokens


def normalize_tokens(tokens):
    """
    Keep only alphanumeric tokens for stable downstream analysis.

    Args:
        tokens (list): Token list

    Returns:
        list: Cleaned alphanumeric tokens
    """
    return [token for token in tokens if token.isalnum()]


def remove_stopwords(tokens):
    """
    Remove common English stopwords from tokens.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: List of tokens without stopwords
    """
    if HAS_STOPWORDS:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set(ENGLISH_STOP_WORDS)
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return filtered_tokens


def lemmatize_tokens(tokens):
    """
    Lemmatize tokens to their base form.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: List of lemmatized tokens
    """
    if HAS_WORDNET:
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        lemmatized = tokens
    return lemmatized


def pos_tag_tokens(tokens):
    """
    Perform POS (Part-of-Speech) tagging on tokens.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: List of tuples (token, pos_tag)
    """
    if HAS_POS_TAGGER:
        try:
            pos_tags = pos_tag(tokens)
        except LookupError:
            pos_tags = [(token, 'UNK') for token in tokens]
    else:
        pos_tags = [(token, 'UNK') for token in tokens]
    return pos_tags


def pos_tag_label(tag):
    """
    Convert POS tag code into a human-readable label.

    Args:
        tag (str): POS tag code

    Returns:
        str: Human readable meaning of the POS tag
    """
    return POS_TAG_LABELS.get(tag, 'Other/Unmapped tag')


def generate_ngrams(tokens, n):
    """
    Generate n-grams from tokens.
    
    Args:
        tokens (list): List of word tokens
        n (int): Size of n-gram (2 for bigrams, 1 for unigrams, etc.)
        
    Returns:
        list: List of n-gram tuples
    """
    n_grams = list(ngrams(tokens, n))
    return n_grams


def preprocess_sentence(sentence):
    """
    Preprocess a single sentence: tokenize, remove stopwords, and lemmatize.
    
    Args:
        sentence (str): Input sentence
        
    Returns:
        list: List of preprocessed tokens
    """
    tokens = tokenize_text(sentence)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens


def split_into_sentences(text):
    """
    Split text into sentences.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of sentences
    """
    if text is None:
        return []

    text = str(text)
    if not text.strip():
        return []

    normalized_text = " ".join(text.split())

    if HAS_PUNKT:
        sentences = sent_tokenize(normalized_text)
    else:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', normalized_text) if s.strip()]
    return sentences


def analyze_text(text):
    """
    Perform comprehensive text analysis.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Tokenization and normalization
    raw_tokens = tokenize_text(text)
    tokens = normalize_tokens(raw_tokens)

    # Stopword removal
    filtered_tokens = remove_stopwords(tokens)
    
    # Lemmatization
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)
    
    # POS tagging
    pos_tags = pos_tag_tokens(tokens)
    
    # N-grams
    unigrams = generate_ngrams(lemmatized_tokens, 1)
    bigrams = generate_ngrams(lemmatized_tokens, 2)

    # Additional quality diagnostics for explainability
    token_count = len(tokens)
    lexical_density = len(filtered_tokens) / token_count if token_count else 0.0
    avg_token_length = (
        sum(len(token) for token in tokens) / token_count if token_count else 0.0
    )
    pos_distribution = Counter(tag for _, tag in pos_tags).most_common()
    keyword_frequency = Counter(lemmatized_tokens).most_common(10)
    unknown_pos_count = sum(1 for _, tag in pos_tags if tag == 'UNK')
    
    return {
        'raw_tokens': raw_tokens,
        'tokens': tokens,
        'filtered_tokens': filtered_tokens,
        'lemmatized_tokens': lemmatized_tokens,
        'pos_tags': pos_tags,
        'pos_distribution': pos_distribution,
        'unigrams': unigrams,
        'bigrams': bigrams,
        'keyword_frequency': keyword_frequency,
        'lexical_density': lexical_density,
        'average_token_length': avg_token_length,
        'unknown_pos_count': unknown_pos_count,
    }
