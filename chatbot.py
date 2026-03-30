"""
Chatbot module with semantic search functionality.
Uses SentenceTransformer embeddings for retrieval and TF-IDF for summary analytics.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import logging
import os
import pickle
import re
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import (
    preprocess_sentence,
    tokenize_text,
    remove_stopwords,
    lemmatize_tokens,
)
from utils import load_notes, split_notes_into_sentences, print_error


class NotesAssistantChatbot:
    """
    Intelligent chatbot that answers questions using semantic similarity.
    """
    
    def __init__(
        self,
        notes_file_path='data/notes.txt',
        similarity_threshold=0.2,
        embedding_model_name='all-MiniLM-L6-v2',
        cache_dir='.cache/embeddings',
        log_file='chatbot_queries.log',
    ):
        """
        Initialize the chatbot with notes from a file.
        
        Args:
            notes_file_path (str): Path to the notes file
            similarity_threshold (float): Minimum similarity score to return an answer
            embedding_model_name (str): SentenceTransformer model name
            cache_dir (str): Directory used for storing embedding cache files
            log_file (str): Path to a file where query logs are written
        """
        self.notes_file_path = notes_file_path
        self.similarity_threshold = similarity_threshold
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir

        self.notes_content = None
        self.sentences = []
        self.processed_sentences = []

        self.embedding_model = None
        self.sentence_embeddings = None
        self.sentence_embedding_norms = None
        self.query_embedding_cache = OrderedDict()
        self.max_query_cache_size = 128

        self.vectorizer = None
        self.tfidf_matrix = None

        self.logger = self._setup_logger(log_file)
        
        # Load and process notes
        self._load_and_process_notes()

    @staticmethod
    def _setup_logger(log_file):
        """
        Configure a file logger for chatbot queries and scores.

        Args:
            log_file (str): Log file path

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('notes_assistant_chatbot')
        if logger.handlers:
            return logger

        logger.setLevel(logging.INFO)
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger
    
    def _load_and_process_notes(self):
        """
        Load notes from file and process them.
        """
        try:
            self.notes_content = load_notes(self.notes_file_path)
            self.sentences = split_notes_into_sentences(self.notes_content)
            if not self.sentences:
                raise ValueError('No valid sentences found in notes. Please use a richer dataset.')

            self._load_embedding_model()
            self._build_embeddings_with_cache()
            self._build_tfidf_model()
        except FileNotFoundError as e:
            print_error(str(e))
            raise
        except Exception as e:
            print_error(f"Error processing notes: {str(e)}")
            raise

    def _cache_file_path(self):
        """
        Build a deterministic cache path for a dataset/model pair.

        Returns:
            str: Absolute cache file path
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_key = hashlib.sha256(
            f"{os.path.abspath(self.notes_file_path)}|{self.embedding_model_name}".encode('utf-8')
        ).hexdigest()[:20]
        return os.path.join(self.cache_dir, f'{cache_key}.pkl')

    def _cache_metadata(self):
        """
        Build cache metadata to detect stale embeddings.

        Returns:
            dict: Cache metadata
        """
        stat = os.stat(self.notes_file_path)
        return {
            'file_path': os.path.abspath(self.notes_file_path),
            'file_size': stat.st_size,
            'file_mtime': stat.st_mtime,
            'notes_hash': hashlib.sha256(self.notes_content.encode('utf-8')).hexdigest(),
            'sentence_count': len(self.sentences),
            'model_name': self.embedding_model_name,
        }

    def _is_valid_cache_payload(self, payload):
        """
        Check whether a loaded cache payload matches current dataset state.

        Args:
            payload (dict): Cache payload

        Returns:
            bool: True if cache payload can be reused
        """
        if not isinstance(payload, dict):
            return False

        metadata = payload.get('metadata')
        embeddings = payload.get('embeddings')
        cached_sentences = payload.get('sentences')

        if metadata is None or embeddings is None or cached_sentences is None:
            return False
        if not isinstance(cached_sentences, list):
            return False
        if cached_sentences != self.sentences:
            return False

        current = self._cache_metadata()
        metadata_keys = (
            'file_path',
            'file_size',
            'file_mtime',
            'notes_hash',
            'sentence_count',
            'model_name',
        )
        for key in metadata_keys:
            if metadata.get(key) != current.get(key):
                return False

        return isinstance(embeddings, np.ndarray) and embeddings.shape[0] == len(self.sentences)

    def _load_embedding_model(self):
        """
        Load the sentence embedding model once.
        """
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def _load_embeddings_from_cache(self):
        """
        Try loading precomputed sentence embeddings from local cache.

        Returns:
            np.ndarray | None: Cached embeddings if available and valid
        """
        cache_path = self._cache_file_path()
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as file_handle:
                payload = pickle.load(file_handle)
            if not self._is_valid_cache_payload(payload):
                return None

            embeddings = payload['embeddings'].astype(np.float32)
            self.logger.info('Loaded sentence embeddings from cache: %s', cache_path)
            return embeddings
        except Exception as exc:
            self.logger.warning('Failed to load embedding cache (%s): %s', cache_path, exc)
            return None

    def _save_embeddings_to_cache(self):
        """
        Save sentence embeddings to local cache for fast future startup.
        """
        cache_path = self._cache_file_path()
        payload = {
            'metadata': self._cache_metadata(),
            'sentences': self.sentences,
            'embeddings': self.sentence_embeddings,
        }

        temp_path = f'{cache_path}.tmp'
        try:
            with open(temp_path, 'wb') as file_handle:
                pickle.dump(payload, file_handle)
            os.replace(temp_path, cache_path)
            self.logger.info('Saved sentence embeddings cache: %s', cache_path)
        except Exception as exc:
            self.logger.warning('Failed to save embedding cache (%s): %s', cache_path, exc)
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _build_embeddings_with_cache(self):
        """
        Build note sentence embeddings using cache whenever possible.
        """
        cached_embeddings = self._load_embeddings_from_cache()
        if cached_embeddings is not None:
            self.sentence_embeddings = cached_embeddings
        else:
            self.sentence_embeddings = self.embedding_model.encode(
                self.sentences,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ).astype(np.float32)
            self._save_embeddings_to_cache()

        self.sentence_embedding_norms = np.linalg.norm(self.sentence_embeddings, axis=1)
    
    def _build_tfidf_model(self):
        """
        Build a TF-IDF model for summary statistics and keyword extraction.
        """
        if len(self.sentences) == 0:
            raise ValueError("No sentences found in notes to build TF-IDF model")
        
        # Preprocess sentences before vectorization to improve semantic matching.
        self.processed_sentences = []
        for sentence in self.sentences:
            processed_tokens = preprocess_sentence(sentence)
            processed_text = " ".join(processed_tokens).strip()
            self.processed_sentences.append(processed_text if processed_text else sentence.lower())

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        
        # Fit and transform processed sentences
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_sentences)

    def _cosine_similarity_scores(self, query_embedding):
        """
        Compute cosine similarity between query embedding and cached sentence embeddings.

        Args:
            query_embedding (np.ndarray): Query vector

        Returns:
            np.ndarray: Similarity score for each sentence
        """
        query_norm = float(np.linalg.norm(query_embedding))
        if query_norm == 0.0:
            return np.zeros(len(self.sentences), dtype=np.float32)

        denominator = self.sentence_embedding_norms * query_norm
        numerator = np.dot(self.sentence_embeddings, query_embedding)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(self.sentence_embedding_norms, dtype=np.float32),
            where=denominator > 0,
        )

    def _encode_query_embedding(self, question):
        """
        Encode a query into embedding space with a tiny in-memory cache.

        Args:
            question (str): User question

        Returns:
            np.ndarray: Query embedding vector
        """
        cache_key = " ".join(question.lower().split())
        cached = self.query_embedding_cache.get(cache_key)
        if cached is not None:
            self.query_embedding_cache.move_to_end(cache_key)
            return cached

        embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )[0].astype(np.float32)

        self.query_embedding_cache[cache_key] = embedding
        self.query_embedding_cache.move_to_end(cache_key)
        if len(self.query_embedding_cache) > self.max_query_cache_size:
            self.query_embedding_cache.popitem(last=False)
        return embedding

    @staticmethod
    def _token_set(text):
        """Build a normalized token set used for lexical overlap checks."""
        return set(re.findall(r'[a-z0-9]+', text.lower()))

    def _answer_quality_adjustment(self, question, sentence):
        """
        Adjust ranking scores to demote low-information fragments.

        Args:
            question (str): User question
            sentence (str): Candidate note sentence

        Returns:
            float: Additive score adjustment
        """
        score = 0.0
        sentence_clean = sentence.strip()
        sentence_lower = sentence_clean.lower()
        sentence_tokens = self._token_set(sentence_clean)
        question_tokens = self._token_set(question)

        if sentence_clean.endswith(':'):
            score -= 0.20
        if len(sentence_tokens) <= 2:
            score -= 0.30
        if sentence_lower.startswith(('q:', 'a:')):
            score -= 0.25
        if sentence_lower.startswith(('example:', 'types:', 'tasks:', 'popular models:')):
            score -= 0.08

        if question_tokens and sentence_tokens:
            overlap_ratio = len(question_tokens.intersection(sentence_tokens)) / len(question_tokens)
            if overlap_ratio >= 0.35:
                score += 0.06

        return score

    def _confidence_label(self, score):
        """
        Convert similarity score to a readable confidence label.

        Args:
            score (float): Similarity score

        Returns:
            str: Confidence label
        """
        if score >= 0.6:
            return 'High'
        if score >= 0.35:
            return 'Medium'
        if score >= self.similarity_threshold:
            return 'Low'
        return 'Very Low'

    def _extract_definition_target(self, question):
        """
        Extract likely target term for definition-style questions.

        Args:
            question (str): User question

        Returns:
            str: Target term or empty string
        """
        cleaned = question.strip().lower()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.split(r"[?.!]", cleaned, maxsplit=1)[0].strip()
        patterns = [
            r"^(what is|what's|define|meaning of|full form of)\s+(.+)$",
            r"^what does\s+(.+)\s+mean$",
        ]
        for pattern in patterns:
            match = re.match(pattern, cleaned)
            if match:
                target = match.group(match.lastindex).strip()
                target = re.sub(r"^(the|a|an)\s+", "", target)
                target = re.split(
                    r"\b(explain|describe|elaborate|mention|list|give|with|including|and)\b",
                    target,
                    maxsplit=1,
                )[0].strip()
                target = target.strip("'\" ")
                return target
        return ""

    def _definition_bonus(self, question, sentence):
        """
        Add a small bonus for sentences that look like direct definitions.

        Args:
            question (str): User question
            sentence (str): Candidate sentence

        Returns:
            float: Bonus score to improve ranking quality
        """
        question_lower = question.strip().lower()
        definition_question = question_lower.startswith(
            ('what is', "what's", 'define', 'meaning of', 'full form of', 'what does')
        )
        if not definition_question:
            return 0.0

        sentence_lower = sentence.lower()
        target = self._extract_definition_target(question)
        if not target:
            return 0.0

        bonus = 0.0
        target_tokens = target.split()
        single_token_target = len(target_tokens) == 1

        if target in sentence_lower:
            bonus += 0.06

        alt_target = re.sub(r"\s*\([^)]*\)", "", target).strip()
        if alt_target and alt_target != target and alt_target in sentence_lower:
            bonus += 0.05

        if (
            target
            and (
                sentence_lower.startswith(f"{target} ")
                or sentence_lower.startswith(f"{target}(")
            )
            and (' is ' in sentence_lower or ' are ' in sentence_lower)
        ):
            bonus += 0.25

        # Strongly prefer expansion/definition patterns such as "..., or MRR, ..."
        if single_token_target and f", or {target}," in sentence_lower:
            bonus += 0.30

        if single_token_target and (
            f"{target} is " in sentence_lower
            or sentence_lower.startswith(f"{target} ")
            or f"{target} stands for" in sentence_lower
            or f"{target} refers to" in sentence_lower
        ):
            bonus += 0.20

        if any(cue in sentence_lower for cue in (' stands for ', ' refers to ', ' is ', ' means ')):
            bonus += 0.05

        return bonus

    def _rank_sentences(self, question, top_k=3):
        """
        Rank note sentences against the input question.

        Args:
            question (str): User question
            top_k (int): Number of top matches to return

        Returns:
            list: Ranked sentence matches
        """
        query_embedding = self._encode_query_embedding(question)

        similarities = self._cosine_similarity_scores(query_embedding)

        adjusted_scores = []
        for index, base_score in enumerate(similarities):
            sentence = self.sentences[index]
            bonus = self._definition_bonus(question, sentence)
            quality_adjustment = self._answer_quality_adjustment(question, sentence)
            adjusted_scores.append(float(base_score + bonus + quality_adjustment))

        adjusted_scores = np.clip(np.array(adjusted_scores, dtype=np.float32), -1.0, 1.0)
        top_indices = np.argsort(adjusted_scores)[::-1][:top_k]

        ranked = []
        for index in top_indices:
            ranked.append(
                {
                    'sentence': self.sentences[index],
                    'score': float(adjusted_scores[index]),
                    'base_score': float(similarities[index]),
                    'index': int(index),
                }
            )
        return ranked

    def _log_query(self, question, score, confidence, latency_ms, top_matches=None):
        """
        Write query-level telemetry to log file.

        Args:
            question (str): User question
            score (float): Best match score
            confidence (str): Confidence label
            latency_ms (float): Retrieval latency in milliseconds
            top_matches (list | None): Top ranked sentence matches
        """
        top_scores = ''
        if top_matches:
            top_scores = ','.join(f"{match['score']:.4f}" for match in top_matches[:3])

        self.logger.info(
            'query="%s" | score=%.4f | top_scores=[%s] | confidence=%s | latency_ms=%.2f',
            question,
            score,
            top_scores,
            confidence,
            latency_ms,
        )
    
    def get_answer(self, question):
        """
        Get answer to a question using semantic similarity.
        
        Args:
            question (str): User's question
            
        Returns:
            dict: Dictionary containing answer, score, and matched sentence
        """
        if not question or question.strip() == "":
            return {
                'answer': "Sorry, I couldn't find a relevant answer in the notes.",
                'score': 0.0,
                'matched_sentence': None,
                'confidence': 'Very Low',
                'top_matches': [],
            }
        
        try:
            start_time = time.perf_counter()
            ranked_matches = self._rank_sentences(question, top_k=3)
            if not ranked_matches:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                self._log_query(question, 0.0, 'Very Low', latency_ms, top_matches=[])
                return {
                    'answer': "Sorry, I couldn't find a relevant answer in the notes.",
                    'score': 0.0,
                    'matched_sentence': None,
                    'confidence': 'Very Low',
                    'top_matches': [],
                }

            best_match = ranked_matches[0]
            best_match_score = best_match['score']
            confidence = self._confidence_label(float(best_match_score))
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Check if similarity is above threshold
            if best_match_score < self.similarity_threshold:
                self._log_query(
                    question,
                    float(best_match_score),
                    confidence,
                    latency_ms,
                    top_matches=ranked_matches,
                )
                return {
                    'answer': "Sorry, I couldn't find a relevant answer in the notes.",
                    'score': float(best_match_score),
                    'matched_sentence': None,
                    'confidence': confidence,
                    'top_matches': ranked_matches,
                }
            
            # Return the best matching sentence
            self._log_query(
                question,
                float(best_match_score),
                confidence,
                latency_ms,
                top_matches=ranked_matches,
            )
            return {
                'answer': best_match['sentence'],
                'score': float(best_match_score),
                'matched_sentence': best_match['sentence'],
                'confidence': confidence,
                'top_matches': ranked_matches,
            }
        
        except Exception as e:
            self.logger.exception('Failed to answer query: %s', question)
            return {
                'answer': f"Error processing question: {str(e)}",
                'score': 0.0,
                'matched_sentence': None,
                'confidence': 'Very Low',
                'top_matches': [],
            }
    
    def get_summary(self):
        """
        Get summary statistics about the notes.
        
        Returns:
            dict: Summary information including sentence count and top keywords
        """
        tokens = [token for token in tokenize_text(self.notes_content) if token.isalnum()]
        filtered_tokens = remove_stopwords(tokens)
        lemmatized_tokens = lemmatize_tokens(filtered_tokens)
        sentence_count = len(self.sentences)

        summary = {
            'total_sentences': len(self.sentences),
            'total_words': len(tokens),
            'total_characters': len(self.notes_content),
            'vocabulary_size': len(set(lemmatized_tokens)),
            'average_sentence_length': round((len(tokens) / sentence_count), 2) if sentence_count else 0.0,
            'vocabulary_diversity': round((len(set(lemmatized_tokens)) / len(lemmatized_tokens)), 3) if lemmatized_tokens else 0.0,
            'top_keywords': self._get_top_keywords(),
            'top_keyphrases': self._get_top_keyphrases(),
            'extractive_highlights': self._get_extractive_highlights(),
            'most_frequent_terms': Counter(lemmatized_tokens).most_common(8),
        }
        return summary
    
    def _get_top_keywords(self, n=10):
        """
        Get top keywords using TF-IDF scores.
        
        Args:
            n (int): Number of top keywords to return
            
        Returns:
            list: List of top keywords with their scores
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        tfidf_scores = self.tfidf_matrix.mean(axis=0).A1
        unigram_indices = [index for index, term in enumerate(feature_names) if ' ' not in term]
        sorted_indices = sorted(
            unigram_indices,
            key=lambda index: tfidf_scores[index],
            reverse=True,
        )[:n]

        top_keywords = [(feature_names[i], float(tfidf_scores[i])) for i in sorted_indices]
        return top_keywords

    def _get_top_keyphrases(self, n=8):
        """
        Get top bigram keyphrases using TF-IDF scores.

        Args:
            n (int): Number of keyphrases to return

        Returns:
            list: List of top keyphrases with scores
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        tfidf_scores = self.tfidf_matrix.mean(axis=0).A1
        bigram_indices = [index for index, term in enumerate(feature_names) if ' ' in term]
        sorted_indices = sorted(
            bigram_indices,
            key=lambda index: tfidf_scores[index],
            reverse=True,
        )[:n]
        return [(feature_names[i], float(tfidf_scores[i])) for i in sorted_indices]

    def _get_extractive_highlights(self, n=3):
        """
        Return high-information sentences from notes.

        Args:
            n (int): Number of highlight sentences

        Returns:
            list: Extractive summary sentences
        """
        if not self.sentences or self.tfidf_matrix is None:
            return []
        sentence_scores = np.asarray(self.tfidf_matrix.sum(axis=1)).ravel()
        ranked_indices = np.argsort(sentence_scores)[::-1][:n]
        return [self.sentences[index] for index in ranked_indices]
    
    def analyze_question(self, question):
        """
        Analyze a question using NLP techniques.
        
        Args:
            question (str): Question to analyze
            
        Returns:
            dict: Analysis results
        """
        from preprocessing import analyze_text
        
        analysis = analyze_text(question)
        return analysis
    
    def reload_notes(self, notes_file_path=None):
        """
        Reload notes from file (useful if notes are updated).
        
        Args:
            notes_file_path (str): Optional new path to notes file
        """
        if notes_file_path:
            self.notes_file_path = notes_file_path
        
        self._load_and_process_notes()
    
    def get_all_sentences(self):
        """
        Get all sentences from notes.
        
        Returns:
            list: List of all sentences
        """
        return self.sentences
