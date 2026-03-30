"""
Utility functions for the notes assistant chatbot.
Handles file operations and helper functions.
"""

import os
import re
from preprocessing import split_into_sentences


LIST_PREFIX_NUMERIC_RE = re.compile(r"^\d+\s*[.)-]\s*")
LIST_PREFIX_ALPHA_RE = re.compile(r"^[A-Za-z]\s*[.)-]\s*")
NON_ALNUM_PREFIX_RE = re.compile(r"^[^A-Za-z0-9(]+")
WORD_RE = re.compile(r"[A-Za-z]+")
PURE_ENUM_RE = re.compile(r"^\d+[.)]?$")
LINKING_VERB_RE = re.compile(
    r"\b(is|are|was|were|be|being|been|has|have|had|can|could|will|would|should|may|might|do|does|did|"
    r"deal|deals|focus|focuses|analyzes|determine|determines|understand|understands|"
    r"predict|predicts|represent|represents|assign|assigns|identify|identifies|"
    r"allow|allows|handle|handles|improve|improves|combine|combines|operate|operates|"
    r"generate|generates|bridge|bridges|gather|gathers|remove|removes|split|splits|"
    r"reduce|reduces|detect|detects|translate|translates|simulate|simulates|"
    r"measure|measures|include|includes|use|uses)\b"
)


def _clean_note_line(raw_line):
    """Normalize and clean a single note line before sentence splitting."""
    line = raw_line.strip().replace("\t", " ")
    if not line:
        return ""

    if not any(char.isalnum() for char in line):
        return ""

    # Remove bullets, decorative markers, and numbering prefixes repeatedly.
    previous = None
    while line and line != previous:
        previous = line
        line = NON_ALNUM_PREFIX_RE.sub("", line).strip()
        line = LIST_PREFIX_NUMERIC_RE.sub("", line).strip()
        line = LIST_PREFIX_ALPHA_RE.sub("", line).strip()

    line = " ".join(line.split())
    if not line:
        return ""
    if PURE_ENUM_RE.fullmatch(line):
        return ""
    return line


def _is_heading_like(text):
    """Heuristic to detect short heading-style lines."""
    if '->' in text or '→' in text:
        return False

    words = WORD_RE.findall(text)
    if not words:
        return True

    has_linking_verb = LINKING_VERB_RE.search(text.lower())
    return len(words) <= 8 and not text.endswith((".", "!", "?")) and not has_linking_verb


def load_notes(file_path):
    """
    Load notes from a text file.
    
    Args:
        file_path (str): Path to the notes file
        
    Returns:
        str: Content of the notes file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not file_path or not str(file_path).strip():
        raise ValueError("Notes file path is empty. Provide a valid text dataset path.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Notes file not found at {file_path}")
    if not os.path.isfile(file_path):
        raise ValueError(f"Provided notes path is not a file: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Notes file is not readable: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        if not content or not content.strip():
            raise ValueError(
                f"Notes file is empty: {file_path}. Add valid note content before running the chatbot."
            )
        return content
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Could not decode notes file as UTF-8: {file_path}. Ensure it is a plain UTF-8 text file."
        ) from e
    except Exception as e:
        raise Exception(f"Error reading notes file: {str(e)}")


def split_notes_into_sentences(notes_content):
    """
    Split loaded notes into individual sentences.
    
    Args:
        notes_content (str): Content of the notes
        
    Returns:
        list: List of sentences
    """
    # Normalize noisy note formatting (bullets/headings) before sentence splitting.
    cleaned_lines = []
    for raw_line in notes_content.splitlines():
        line = _clean_note_line(raw_line)
        if line:
            cleaned_lines.append(line)

    sentences = []
    seen = set()
    index = 0

    while index < len(cleaned_lines):
        line = cleaned_lines[index]

        # Merge short heading lines with the immediate detail line.
        if _is_heading_like(line) and index + 1 < len(cleaned_lines):
            next_line = cleaned_lines[index + 1]
            if not _is_heading_like(next_line):
                separator = " " if line.endswith(":") else ": "
                line = f"{line}{separator}{next_line}"
                index += 1

        for sentence in split_into_sentences(line):
            normalized = " ".join(sentence.split()).strip()
            if not normalized:
                continue
            if PURE_ENUM_RE.fullmatch(normalized):
                continue

            words = WORD_RE.findall(normalized)
            has_linking_verb = LINKING_VERB_RE.search(normalized.lower())
            is_heading_like = (
                len(words) <= 6
                and not normalized.endswith((".", "!", "?"))
                and not has_linking_verb
            )
            if is_heading_like:
                continue

            # Drop low-information lead-in fragments ending with a colon.
            if normalized.endswith(":") and len(words) <= 14:
                continue

            dedupe_key = normalized.lower()
            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            sentences.append(normalized)

        index += 1

    return sentences


def validate_input(user_input):
    """
    Validate user input to ensure it's not empty.
    
    Args:
        user_input (str): User input string
        
    Returns:
        bool: True if input is valid, False otherwise
    """
    if user_input is None or user_input.strip() == "":
        return False
    return True


def format_output(text, width=80):
    """
    Format text output for better readability.
    
    Args:
        text (str): Text to format
        width (int): Maximum line width
        
    Returns:
        str: Formatted text
    """
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)


def print_separator(char='-', length=80):
    """
    Print a separator line.
    
    Args:
        char (str): Character to use for separator
        length (int): Length of separator
    """
    print(char * length)


def print_header(title):
    """
    Print a formatted header.
    
    Args:
        title (str): Title to display
    """
    print_separator()
    print(f"\n{title.upper().center(80)}\n")
    print_separator()


def print_success(message):
    """
    Print a success message.
    
    Args:
        message (str): Success message
    """
    print(f"\n✓ {message}\n")


def print_error(message):
    """
    Print an error message.
    
    Args:
        message (str): Error message
    """
    print(f"\n✗ {message}\n")


def get_similarity_percentage(similarity_score):
    """
    Convert similarity score to percentage.
    
    Args:
        similarity_score (float): Similarity score between 0 and 1
        
    Returns:
        float: Similarity as percentage
    """
    return round(similarity_score * 100, 2)
