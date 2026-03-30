"""
Evaluation module for the Notes Assistant Chatbot.
Runs a fixed benchmark of question-answer pairs and reports quality metrics.
"""

from __future__ import annotations

import argparse
from statistics import mean
import re

from chatbot import NotesAssistantChatbot


EVALUATION_CASES = [
    {
        'question': 'What is Natural Language Processing (NLP)?',
        'expected_answer': 'Natural Language Processing is a subfield of AI focused on understanding, interpreting, and generating human language.',
    },
    {
        'question': 'What are the goals of NLP?',
        'expected_answer': 'The goals include understanding language, generating language, and bridging human communication with machine understanding.',
    },
    {
        'question': 'What does the morphological level of NLP deal with?',
        'expected_answer': 'The morphological level deals with the structure of words, such as prefixes, roots, and suffixes.',
    },
    {
        'question': 'What is the lexical level in NLP?',
        'expected_answer': 'The lexical level focuses on word meaning and vocabulary.',
    },
    {
        'question': 'What does the syntactic level analyze?',
        'expected_answer': 'The syntactic level analyzes sentence structure using grammar rules.',
    },
    {
        'question': 'What is semantic level processing?',
        'expected_answer': 'Semantic level processing determines sentence meaning.',
    },
    {
        'question': 'What is the pragmatic level in NLP?',
        'expected_answer': 'The pragmatic level understands context and real-world meaning.',
    },
    {
        'question': 'What is tokenization?',
        'expected_answer': 'Tokenization splits text into smaller units like words or sentences.',
    },
    {
        'question': 'What is stemming?',
        'expected_answer': 'Stemming reduces words to their root forms.',
    },
    {
        'question': 'How is lemmatization different from stemming?',
        'expected_answer': 'Lemmatization uses dictionary knowledge and returns linguistically valid base forms.',
    },
    {
        'question': 'What is part-of-speech tagging?',
        'expected_answer': 'Part-of-speech tagging assigns grammatical tags like noun, verb, and adjective.',
    },
    {
        'question': 'What is Named Entity Recognition (NER)?',
        'expected_answer': 'Named Entity Recognition identifies entities such as person, location, and organization names.',
    },
    {
        'question': 'What is Bag of Words?',
        'expected_answer': 'Bag of Words represents text as word frequency and ignores word order.',
    },
    {
        'question': 'What does TF-IDF measure?',
        'expected_answer': 'TF-IDF measures word importance by balancing term frequency against document frequency.',
    },
    {
        'question': 'What are word embeddings?',
        'expected_answer': 'Word embeddings are dense vector representations that capture semantic meaning.',
    },
    {
        'question': 'How does Word2Vec CBOW and Skip-gram differ?',
        'expected_answer': 'CBOW predicts a word from context, while Skip-gram predicts context from a word.',
    },
    {
        'question': 'What is the famous word vector analogy example?',
        'expected_answer': 'A common analogy is king minus man plus woman approximately equals queen.',
    },
    {
        'question': 'What are contextual embeddings?',
        'expected_answer': 'Contextual embeddings from models like BERT and GPT change word representation based on context.',
    },
    {
        'question': 'What is a language model?',
        'expected_answer': 'A language model predicts the probability of word sequences.',
    },
    {
        'question': 'What are major types of language models?',
        'expected_answer': 'Major types include statistical n-gram models, neural models, and transformer models.',
    },
    {
        'question': 'What is a bigram?',
        'expected_answer': 'A bigram is a sequence of two words.',
    },
    {
        'question': 'Which machine learning algorithms are common in NLP?',
        'expected_answer': 'Common algorithms include Naive Bayes, Logistic Regression, SVM, and Decision Trees.',
    },
    {
        'question': 'What is a key limitation of RNNs?',
        'expected_answer': 'RNNs can suffer from the vanishing gradient problem.',
    },
    {
        'question': 'Why were transformers important for NLP?',
        'expected_answer': 'Transformers revolutionized NLP by modeling dependencies effectively with attention.',
    },
    {
        'question': 'What is the key concept in transformers?',
        'expected_answer': 'The key concept is the attention mechanism.',
    },
    {
        'question': 'What is text classification?',
        'expected_answer': 'Text classification assigns labels to text, such as spam or not spam.',
    },
    {
        'question': 'What does sentiment analysis detect?',
        'expected_answer': 'Sentiment analysis detects positive, negative, or neutral emotion in text.',
    },
    {
        'question': 'What metrics are used for classification evaluation?',
        'expected_answer': 'Accuracy, precision, recall, and F1-score are used for classification evaluation.',
    },
    {
        'question': 'What are common NLP challenges?',
        'expected_answer': 'Common challenges include ambiguity, sarcasm, context understanding, and multilingual complexity.',
    },
    {
        'question': 'Which libraries are used in NLP projects?',
        'expected_answer': 'Typical libraries include NLTK, spaCy, scikit-learn, TensorFlow or PyTorch, and Hugging Face Transformers.',
    },
]


def _token_set(text):
    """Convert text into a normalized token set for rough lexical scoring."""
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def _precision_recall(predicted, expected):
    """Compute simple token-overlap precision and recall."""
    predicted_tokens = _token_set(predicted)
    expected_tokens = _token_set(expected)
    if not predicted_tokens or not expected_tokens:
        return 0.0, 0.0

    overlap = predicted_tokens.intersection(expected_tokens)
    precision = len(overlap) / len(predicted_tokens) if predicted_tokens else 0.0
    recall = len(overlap) / len(expected_tokens) if expected_tokens else 0.0
    return precision, recall


def _is_correct(predicted, expected, recall, similarity_score):
    """
    Approximate answer correctness for retrieval QA.
    This intentionally uses practical heuristics for BYOP reporting.
    """
    fallback = "sorry, i couldn't find a relevant answer in the notes"
    predicted_lower = predicted.lower().strip()
    expected_lower = expected.lower().strip()

    if fallback in predicted_lower:
        return False
    if expected_lower and expected_lower in predicted_lower:
        return True
    if recall >= 0.50:
        return True
    if recall >= 0.35 and similarity_score >= 0.55:
        return True
    return False


def run_evaluation(
    notes_file_path='data/notes.txt',
    report_file='evaluation_report.txt',
    similarity_threshold=0.2,
):
    """
    Run benchmark evaluation and save a report.

    Args:
        notes_file_path (str): Dataset path used by chatbot
        report_file (str): Output text report path
        similarity_threshold (float): Similarity threshold for answer acceptance

    Returns:
        dict: Evaluation metrics and failed-case details
    """
    chatbot = NotesAssistantChatbot(
        notes_file_path=notes_file_path,
        similarity_threshold=similarity_threshold,
    )

    total_cases = len(EVALUATION_CASES)
    correct_count = 0
    precisions = []
    recalls = []
    failed_cases = []

    for index, case in enumerate(EVALUATION_CASES, start=1):
        question = case['question']
        expected_answer = case['expected_answer']

        result = chatbot.get_answer(question)
        predicted_answer = result.get('answer', '')
        similarity_score = float(result.get('score', 0.0))

        precision, recall = _precision_recall(predicted_answer, expected_answer)
        is_correct = _is_correct(predicted_answer, expected_answer, recall, similarity_score)

        precisions.append(precision)
        recalls.append(recall)

        if is_correct:
            correct_count += 1
        else:
            failed_cases.append(
                {
                    'index': index,
                    'question': question,
                    'expected': expected_answer,
                    'predicted': predicted_answer,
                    'score': similarity_score,
                    'precision': precision,
                    'recall': recall,
                }
            )

    accuracy = correct_count / total_cases if total_cases else 0.0
    avg_precision = mean(precisions) if precisions else 0.0
    avg_recall = mean(recalls) if recalls else 0.0
    f1_approx = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0.0
    )

    lines = []
    lines.append('=' * 88)
    lines.append('NOTES ASSISTANT EVALUATION REPORT')
    lines.append('=' * 88)
    lines.append('')
    lines.append(f'Dataset: {notes_file_path}')
    lines.append(f'Total Test Questions: {total_cases}')
    lines.append(f'Correct Answers: {correct_count}')
    lines.append(f'Failed Answers: {total_cases - correct_count}')
    lines.append('')
    lines.append(f'Accuracy: {accuracy * 100:.2f}%')
    lines.append(f'Average Precision (approx): {avg_precision * 100:.2f}%')
    lines.append(f'Average Recall (approx): {avg_recall * 100:.2f}%')
    lines.append(f'F1 (approx): {f1_approx * 100:.2f}%')
    lines.append('')

    if failed_cases:
        lines.append('-' * 88)
        lines.append('FAILED CASES')
        lines.append('-' * 88)
        for failure in failed_cases:
            lines.append(f"[{failure['index']}] Q: {failure['question']}")
            lines.append(f"Expected: {failure['expected']}")
            lines.append(f"Predicted: {failure['predicted']}")
            lines.append(
                f"Score: {failure['score']:.4f} | Precision: {failure['precision']:.2f} | Recall: {failure['recall']:.2f}"
            )
            lines.append('')
    else:
        lines.append('No failed cases. All benchmark questions were answered successfully.')

    report_text = '\n'.join(lines)

    with open(report_file, 'w', encoding='utf-8') as file_handle:
        file_handle.write(report_text)

    print(report_text)
    print('')
    print(f'Report saved to: {report_file}')

    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1_approx,
        'total': total_cases,
        'correct': correct_count,
        'failed': failed_cases,
        'report_file': report_file,
    }


def main():
    """CLI entry point for standalone evaluation runs."""
    parser = argparse.ArgumentParser(
        description='Run benchmark evaluation for Intelligent Notes Assistant Chatbot'
    )
    parser.add_argument(
        '--notes-file',
        default='data/notes.txt',
        help='Path to notes dataset file (default: data/notes.txt)',
    )
    parser.add_argument(
        '--report-file',
        default='evaluation_report.txt',
        help='Path to output evaluation report file (default: evaluation_report.txt)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.2,
        help='Similarity threshold used by chatbot (default: 0.2)',
    )
    args = parser.parse_args()

    run_evaluation(
        notes_file_path=args.notes_file,
        report_file=args.report_file,
        similarity_threshold=args.threshold,
    )


if __name__ == '__main__':
    main()