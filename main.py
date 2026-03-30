import argparse

from chatbot import NotesAssistantChatbot
from utils import (
    validate_input, print_header, print_error, print_success,
    print_separator, get_similarity_percentage
)
from preprocessing import analyze_text, pos_tag_label


class ChatbotInterface:
    """
    Command-line interface for the Notes Assistant Chatbot.
    """
    
    def __init__(self, notes_file_path='data/notes.txt', similarity_threshold=0.2):
        """Initialize the chatbot interface."""
        print_header("Intelligent Notes Assistant Chatbot")
        
        try:
            self.chatbot = NotesAssistantChatbot(
                notes_file_path=notes_file_path,
                similarity_threshold=similarity_threshold,
            )
            print_success("Notes loaded successfully!")
            print(f"Dataset file: {self.chatbot.notes_file_path}")
            print(f"Total note sentences: {len(self.chatbot.get_all_sentences())}")
            print(f"Similarity threshold: {similarity_threshold}")
            print("Tip: Use --notes-file <path> to run on your own text dataset.\n")
        except Exception as e:
            print_error(f"Failed to initialize chatbot: {str(e)}")
            raise
    
    def display_menu(self):
        """Display the main menu."""
        print("\n" + "="*80)
        print("MAIN MENU".center(80))
        print("="*80)
        print("\n1. Ask Question")
        print("2. Analyze Text")
        print("3. Summary")
        print("4. Exit\n")
        print("="*80 + "\n")
    
    def ask_question(self):
        """Handle question answering."""
        print_header("Ask a Question")
        
        question = input("Enter your question: ").strip()
        
        if not validate_input(question):
            print_error("Please enter a valid question!")
            return
        
        print("\nSearching for answer...\n")
        
        result = self.chatbot.get_answer(question)
        
        # Display result
        print_separator()
        print(f"\nQuestion: {question}\n")
        print(f"Answer: {result['answer']}\n")
        print(f"Similarity Score: {get_similarity_percentage(result['score'])}%\n")
        print(f"Confidence: {result.get('confidence', 'N/A')}\n")

        top_matches = result.get('top_matches', [])
        if top_matches:
            print("Top Candidate Matches:")
            for i, match in enumerate(top_matches, 1):
                score_pct = get_similarity_percentage(match['score'])
                print(f"  {i}. ({score_pct}%) {match['sentence']}")
            print()
        print_separator()
    
    def analyze_text_feature(self):
        """Handle text analysis feature."""
        print_header("Analyze Text")
        
        text = input("Enter text to analyze: ").strip()
        
        if not validate_input(text):
            print_error("Please enter valid text!")
            return
        
        print("\nAnalyzing text...\n")
        
        analysis = analyze_text(text)
        
        # Display analysis results
        print_separator()
        print(f"\nInput Text: {text}\n")
        
        print("TOKENIZATION:")
        print(f"Raw Tokens: {analysis['raw_tokens']}")
        print(f"Tokens: {analysis['tokens']}")
        print(f"Token Count: {len(analysis['tokens'])}\n")
        
        print("STOPWORD REMOVAL:")
        print(f"Filtered Tokens (without stopwords): {analysis['filtered_tokens']}")
        print(f"Tokens Removed: {len(analysis['tokens']) - len(analysis['filtered_tokens'])}\n")
        
        print("LEMMATIZATION:")
        print(f"Lemmatized Tokens: {analysis['lemmatized_tokens']}\n")
        
        print("POS TAGGING:")
        print("Token -> POS Tag (Meaning)")
        for token, pos in analysis['pos_tags'][:15]:
            print(f"  {token} -> {pos} ({pos_tag_label(pos)})")
        if len(analysis['pos_tags']) > 15:
            print(f"  ... and {len(analysis['pos_tags']) - 15} more")
        print(f"Unknown POS Tags: {analysis['unknown_pos_count']}")
        print()

        print("POS DISTRIBUTION:")
        for tag, count in analysis['pos_distribution'][:8]:
            print(f"  {tag}: {count} ({pos_tag_label(tag)})")
        print()
        
        print("N-GRAMS:")
        print(f"Unigrams (Top 10): {[' '.join(gram) for gram in analysis['unigrams'][:10]]}")
        print(f"Total Unigrams: {len(analysis['unigrams'])}\n")
        
        print(f"Bigrams (Top 10): {[' '.join(gram) for gram in analysis['bigrams'][:10]]}")
        print(f"Total Bigrams: {len(analysis['bigrams'])}\n")

        print("TEXT QUALITY METRICS:")
        print(f"Lexical Density: {analysis['lexical_density']:.2f}")
        print(f"Average Token Length: {analysis['average_token_length']:.2f}")
        print(f"Top Keywords: {analysis['keyword_frequency']}\n")
        
        print_separator()
    
    def display_summary(self):
        """Display summary of notes."""
        print_header("Notes Summary")
        
        summary = self.chatbot.get_summary()
        
        # Display summary
        print_separator()
        print(f"\nStatistics:")
        print(f"  Total Sentences: {summary['total_sentences']}")
        print(f"  Total Words: {summary['total_words']}")
        print(f"  Total Characters: {summary['total_characters']}\n")
        print(f"  Vocabulary Size: {summary['vocabulary_size']}")
        print(f"  Average Sentence Length: {summary['average_sentence_length']}")
        print(f"  Vocabulary Diversity: {summary['vocabulary_diversity']}\n")
        
        print("Top Keywords (by TF-IDF Score):")
        for i, (keyword, score) in enumerate(summary['top_keywords'], 1):
            print(f"  {i}. {keyword} (Score: {score:.4f})")

        print("\nTop Keyphrases (Bigrams):")
        for i, (phrase, score) in enumerate(summary['top_keyphrases'], 1):
            print(f"  {i}. {phrase} (Score: {score:.4f})")

        print("\nMost Frequent Terms:")
        for i, (term, count) in enumerate(summary['most_frequent_terms'], 1):
            print(f"  {i}. {term} (Count: {count})")

        print("\nExtractive Highlights:")
        for i, sentence in enumerate(summary['extractive_highlights'], 1):
            print(f"  {i}. {sentence}")
        
        print()
        print_separator()
    
    def run(self):
        """Run the chatbot interface."""
        while True:
            self.display_menu()
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                self.ask_question()
            elif choice == '2':
                self.analyze_text_feature()
            elif choice == '3':
                self.display_summary()
            elif choice == '4':
                print_header("Thank You")
                print("Thank you for using the Intelligent Notes Assistant Chatbot!")
                print("Goodbye!\n")
                break
            else:
                print_error("Invalid choice! Please enter 1, 2, 3, or 4.")


def main():
    """Main function to start the chatbot."""
    try:
        parser = argparse.ArgumentParser(
            description="Intelligent Notes Assistant Chatbot with Semantic Search"
        )
        parser.add_argument(
            "--notes-file",
            default="data/notes.txt",
            help="Path to the text dataset file (default: data/notes.txt)",
        )
        parser.add_argument(
            "--evaluate",
            action="store_true",
            help="Run benchmark evaluation instead of interactive chat mode",
        )
        parser.add_argument(
            "--report-file",
            default="evaluation_report.txt",
            help="Evaluation report output file (default: evaluation_report.txt)",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.2,
            help="Similarity threshold for answer acceptance (default: 0.2)",
        )
        args = parser.parse_args()

        if args.evaluate:
            from evaluation import run_evaluation

            print_header("Evaluation Mode")
            run_evaluation(
                notes_file_path=args.notes_file,
                report_file=args.report_file,
                similarity_threshold=args.threshold,
            )
            return

        interface = ChatbotInterface(
            notes_file_path=args.notes_file,
            similarity_threshold=args.threshold,
        )
        interface.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print_error(f"An unexpected error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
