"""
Music Recommender — CLI entry point.

Modes:
  classic   Rule-based scoring only (no API key needed).
  ai        Claude-powered RAG + agentic workflow.
  demo      Runs both classic profiles AND one AI query side-by-side.

Usage:
  python -m src.main                   # classic mode (batch profiles)
  python -m src.main --mode ai         # AI mode (interactive)
  python -m src.main --mode ai --query "something chill for studying"
  python -m src.main --mode demo
"""

import argparse
import sys

try:
    from src.recommender import load_songs, recommend_songs, confidence
    from src.ai_recommender import AIRecommender
    from src.logger_config import setup_logging
except ImportError:
    from recommender import load_songs, recommend_songs, confidence
    from ai_recommender import AIRecommender
    from logger_config import setup_logging

DATA_PATH = "data/songs.csv"

# ---------------------------------------------------------------------------
# Classic mode display
# ---------------------------------------------------------------------------

CLASSIC_PROFILES = [
    ("High-Energy Pop",
     {"genre": "pop",   "mood": "happy",   "energy": 0.92, "likes_acoustic": False}),
    ("Chill Lofi Study Session",
     {"genre": "lofi",  "mood": "chill",   "energy": 0.38, "likes_acoustic": True}),
    ("Deep Intense Rock",
     {"genre": "rock",  "mood": "intense", "energy": 0.93, "likes_acoustic": False}),
    ("EDGE: Sad but Loud",
     {"genre": "metal", "mood": "sad",     "energy": 0.92, "likes_acoustic": False}),
    ("EDGE: Acoustic but High-Energy",
     {"genre": "jazz",  "mood": "relaxed", "energy": 0.88, "likes_acoustic": True}),
    ("EDGE: Rare Genre (classical / melancholic)",
     {"genre": "classical", "mood": "melancholic", "energy": 0.22, "likes_acoustic": True}),
]


def print_classic(label: str, prefs: dict, recommendations: list) -> None:
    print("\n" + "=" * 60)
    print(f"  {label}")
    print(
        f"  Genre: {prefs.get('genre', '—')}  |  Mood: {prefs.get('mood', '—')}  |  "
        f"Energy: {prefs.get('energy', '—')}  |  Acoustic: {prefs.get('likes_acoustic', False)}"
    )
    print("=" * 60)
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        conf = confidence(score)
        bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        print(f"\n#{rank}  {song['title']} by {song['artist']}")
        print(f"    Score : {score:.2f} / 7.50  |  Confidence: {conf:.2f}  [{bar}]")
        print(f"    Genre : {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")
        for reason in explanation.split(" | "):
            print(f"    + {reason}")
    print("\n" + "=" * 60)


def run_classic(songs: list, logger) -> None:
    logger.info("Running classic (rule-based) mode with %d profiles", len(CLASSIC_PROFILES))
    for label, prefs in CLASSIC_PROFILES:
        recs = recommend_songs(prefs, songs, k=5)
        print_classic(label, prefs, recs)

# ---------------------------------------------------------------------------
# AI mode display
# ---------------------------------------------------------------------------

def print_ai_result(result: dict) -> None:
    recs   = result.get("recommendations", [])
    summary = result.get("summary", "")

    print("\n" + "=" * 60)
    print(f"  AI Recommendations for: \"{result['query']}\"")
    print("=" * 60)

    if not recs:
        print("\n  (No recommendations produced — check logs for details)")
    else:
        for rank, r in enumerate(recs, start=1):
            song = r["song"]
            print(f"\n#{rank}  {song['title']} by {song['artist']}")
            print(f"    Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")
            if r.get("explanation"):
                print(f"    Why: {r['explanation']}")
            if r.get("trade_offs"):
                print(f"    Trade-offs: {r['trade_offs']}")

    if summary:
        print(f"\n  Summary: {summary}")
    print("\n" + "=" * 60)


def run_ai_interactive(songs: list, logger) -> None:
    """REPL loop: user types a query, Claude returns recommendations."""
    try:
        rec = AIRecommender(songs)
    except EnvironmentError as e:
        print(f"\nSetup error: {e}")
        sys.exit(1)

    print("\nAI Music Recommender (type 'quit' to exit)")
    print("Describe what you want to listen to in plain English.\n")

    while True:
        try:
            query = input("Your request: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        logger.info("AI query submitted: %r", query)
        print("\nThinking...\n")
        try:
            result = rec.recommend(query, k=5)
            print_ai_result(result)
        except Exception as exc:
            logger.exception("Error during AI recommendation")
            print(f"\nError: {exc}\n")


def run_ai_single(songs: list, query: str, logger) -> None:
    """Non-interactive: run one AI query from the command line."""
    try:
        rec = AIRecommender(songs)
    except EnvironmentError as e:
        print(f"\nSetup error: {e}")
        sys.exit(1)

    logger.info("AI single query: %r", query)
    result = rec.recommend(query, k=5)
    print_ai_result(result)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Music Recommender")
    parser.add_argument(
        "--mode", choices=["classic", "ai", "demo"], default="classic",
        help="classic = rule-based only; ai = Claude-powered; demo = both"
    )
    parser.add_argument(
        "--query", type=str, default="",
        help="Query string for AI mode (skips interactive prompt if provided)"
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting recommender in mode=%s", args.mode)

    songs = load_songs(DATA_PATH)
    logger.info("Loaded %d songs from %s", len(songs), DATA_PATH)

    if args.mode == "classic":
        run_classic(songs, logger)

    elif args.mode == "ai":
        if args.query:
            run_ai_single(songs, args.query, logger)
        else:
            run_ai_interactive(songs, logger)

    elif args.mode == "demo":
        print("\n=== CLASSIC MODE ===")
        run_classic(songs, logger)
        print("\n\n=== AI MODE (sample query) ===")
        run_ai_single(songs, "something chill and acoustic for a late night study session", logger)


if __name__ == "__main__":
    main()
