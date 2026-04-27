"""
Reliability and correctness tests for the AI recommender.

These tests exercise:
  1. Rule-based components (no API key needed)
  2. Tool execution logic in isolation (no API key needed)
  3. Output schema validation
  4. Edge-case guardrails (max_tool_calls, missing song_id, etc.)
  5. Consistency: same query should always surface the same #1 rule-based pick
     (deterministic scoring — verifies the RAG retrieval layer is stable)

Integration tests that call the real Claude API are skipped automatically
when ANTHROPIC_API_KEY is not set.
"""

import os
import pytest

from src.recommender import Song, UserProfile, Recommender, load_songs, score_song

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_catalog():
    return [
        {"id": 1,  "title": "Neon Blast",    "artist": "A", "genre": "edm",   "mood": "euphoric",
         "energy": 0.95, "tempo_bpm": 130, "valence": 0.90, "danceability": 0.92, "acousticness": 0.02},
        {"id": 2,  "title": "Rainy Pages",   "artist": "B", "genre": "lofi",  "mood": "chill",
         "energy": 0.35, "tempo_bpm": 75,  "valence": 0.55, "danceability": 0.55, "acousticness": 0.80},
        {"id": 3,  "title": "Iron Riff",     "artist": "C", "genre": "metal", "mood": "angry",
         "energy": 0.96, "tempo_bpm": 172, "valence": 0.20, "danceability": 0.60, "acousticness": 0.03},
        {"id": 4,  "title": "Sunset Stroll", "artist": "D", "genre": "jazz",  "mood": "relaxed",
         "energy": 0.38, "tempo_bpm": 90,  "valence": 0.72, "danceability": 0.52, "acousticness": 0.87},
    ]


def make_ai_recommender(catalog=None):
    """Return an AIRecommender with a mock client — skips API calls."""
    songs = catalog or make_catalog()
    # Import here so tests that don't need AI don't fail on missing key
    from src.ai_recommender import AIRecommender
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
    rec = AIRecommender.__new__(AIRecommender)
    rec.songs = songs
    rec.model = "claude-sonnet-4-6"
    rec._index = {s["id"]: s for s in songs}
    # No real client — tool execution tests call internal methods directly
    return rec


# ---------------------------------------------------------------------------
# 1. Recommender class (OOP wrapper) — must pass with no API key
# ---------------------------------------------------------------------------

class TestRecommenderClass:

    def _make_recommender(self):
        songs = [
            Song(1, "Pop Banger", "Art1", "pop",  "happy",    0.85, 120, 0.88, 0.82, 0.12),
            Song(2, "Chill Waves","Art2", "lofi", "chill",    0.38,  80, 0.58, 0.60, 0.78),
            Song(3, "Metal Storm","Art3", "metal","intense",  0.94, 168, 0.25, 0.65, 0.04),
        ]
        return Recommender(songs)

    def test_recommend_returns_k_songs(self):
        rec  = self._make_recommender()
        user = UserProfile("pop", "happy", 0.85, False)
        results = rec.recommend(user, k=2)
        assert len(results) == 2

    def test_recommend_top_song_matches_genre_and_mood(self):
        rec  = self._make_recommender()
        user = UserProfile("pop", "happy", 0.85, False)
        results = rec.recommend(user, k=3)
        assert results[0].genre == "pop"
        assert results[0].mood  == "happy"

    def test_recommend_sorted_descending_by_score(self):
        rec  = self._make_recommender()
        user = UserProfile("pop", "happy", 0.85, False)
        results = rec.recommend(user, k=3)
        prefs = {"genre": "pop", "mood": "happy", "energy": 0.85, "likes_acoustic": False}
        scores = [score_song(prefs, {"id": s.id, "title": s.title, "artist": s.artist,
                                      "genre": s.genre, "mood": s.mood, "energy": s.energy,
                                      "tempo_bpm": s.tempo_bpm, "valence": s.valence,
                                      "danceability": s.danceability, "acousticness": s.acousticness})[0]
                  for s in results]
        assert scores == sorted(scores, reverse=True)

    def test_explain_returns_non_empty_string(self):
        rec  = self._make_recommender()
        user = UserProfile("pop", "happy", 0.85, False)
        expl = rec.explain_recommendation(user, rec.songs[0])
        assert isinstance(expl, str)
        assert expl.strip() != ""

    def test_explain_mentions_matching_features(self):
        rec  = self._make_recommender()
        user = UserProfile("pop", "happy", 0.85, False)
        expl = rec.explain_recommendation(user, rec.songs[0])
        # Should mention at least genre or mood or energy
        assert any(kw in expl.lower() for kw in ("genre", "mood", "energy"))


# ---------------------------------------------------------------------------
# 2. score_song — unit tests for scoring rules
# ---------------------------------------------------------------------------

class TestScoreSong:

    def _song(self, **kwargs):
        base = {"id": 1, "title": "T", "artist": "A", "genre": "pop", "mood": "happy",
                "energy": 0.80, "tempo_bpm": 120, "valence": 0.80, "danceability": 0.75,
                "acousticness": 0.15}
        base.update(kwargs)
        return base

    def test_genre_match_adds_points(self):
        prefs = {"genre": "pop", "mood": "sad", "energy": 0.5, "likes_acoustic": False}
        s_match   = self._song(genre="pop")
        s_nomatch = self._song(genre="rock")
        sc_match, _   = score_song(prefs, s_match)
        sc_nomatch, _ = score_song(prefs, s_nomatch)
        assert sc_match > sc_nomatch

    def test_mood_match_adds_points(self):
        prefs = {"genre": "jazz", "mood": "happy", "energy": 0.5, "likes_acoustic": False}
        s_match   = self._song(genre="jazz", mood="happy")
        s_nomatch = self._song(genre="jazz", mood="sad")
        sc_match, _   = score_song(prefs, s_match)
        sc_nomatch, _ = score_song(prefs, s_nomatch)
        assert sc_match > sc_nomatch

    def test_energy_close_scores_higher(self):
        prefs = {"genre": "rock", "mood": "intense", "energy": 0.90, "likes_acoustic": False}
        s_close = self._song(genre="rock", mood="intense", energy=0.88)
        s_far   = self._song(genre="rock", mood="intense", energy=0.20)
        sc_close, _ = score_song(prefs, s_close)
        sc_far,   _ = score_song(prefs, s_far)
        assert sc_close > sc_far

    def test_acoustic_preference_rewarded(self):
        prefs = {"genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True}
        s_acoustic = self._song(genre="lofi", mood="chill", energy=0.35, acousticness=0.90)
        s_digital  = self._song(genre="lofi", mood="chill", energy=0.35, acousticness=0.10)
        sc_a, _ = score_song(prefs, s_acoustic)
        sc_d, _ = score_song(prefs, s_digital)
        assert sc_a > sc_d

    def test_score_is_non_negative(self):
        prefs = {"genre": "edm", "mood": "euphoric", "energy": 0.99, "likes_acoustic": False}
        s = self._song(genre="folk", mood="sad", energy=0.10, acousticness=0.95)
        sc, _ = score_song(prefs, s)
        assert sc >= 0.0


# ---------------------------------------------------------------------------
# 3. AIRecommender — tool execution (no real API calls)
# ---------------------------------------------------------------------------

class TestToolExecution:

    def test_search_catalog_no_filters_returns_all(self):
        rec = make_ai_recommender()
        result = rec._tool_search_catalog({})
        assert result["count"] == len(rec.songs)

    def test_search_catalog_genre_filter(self):
        rec = make_ai_recommender()
        result = rec._tool_search_catalog({"genre": "lofi"})
        assert result["count"] == 1
        assert result["songs"][0]["genre"] == "lofi"

    def test_search_catalog_energy_range(self):
        rec = make_ai_recommender()
        result = rec._tool_search_catalog({"min_energy": 0.90})
        # songs 1 (0.95) and 3 (0.96) qualify
        assert result["count"] == 2

    def test_search_catalog_acoustic_true(self):
        rec = make_ai_recommender()
        result = rec._tool_search_catalog({"acoustic": True})
        # songs with acousticness >= 0.65: id 2 (0.80) and id 4 (0.87)
        assert result["count"] == 2

    def test_score_song_valid_id(self):
        rec = make_ai_recommender()
        prefs = {"genre": "edm", "mood": "euphoric", "energy": 0.95, "likes_acoustic": False}
        result = rec._tool_score_song({"song_id": 1, "user_prefs": prefs})
        assert "score" in result
        assert result["score"] >= 0.0

    def test_score_song_invalid_id_returns_error(self):
        rec = make_ai_recommender()
        result = rec._tool_score_song({"song_id": 999, "user_prefs": {}})
        assert "error" in result

    def test_score_song_bad_type_returns_error(self):
        rec = make_ai_recommender()
        result = rec._tool_score_song({"song_id": "not-an-int", "user_prefs": {}})
        assert "error" in result


# ---------------------------------------------------------------------------
# 4. RAG retrieval consistency (deterministic — no API)
# ---------------------------------------------------------------------------

class TestRAGRetrieval:

    def test_retrieval_is_deterministic(self):
        """Same query must always return the same ordered candidate list."""
        rec = make_ai_recommender()
        q = "high energy workout music"
        result1 = [c["song"]["id"] for c in rec._rag_retrieve(q, n=3)]
        result2 = [c["song"]["id"] for c in rec._rag_retrieve(q, n=3)]
        assert result1 == result2

    def test_retrieval_count_respects_n(self):
        rec = make_ai_recommender()
        candidates = rec._rag_retrieve("chill lofi beats", n=2)
        assert len(candidates) == 2

    def test_retrieval_returns_sorted_by_rule_score(self):
        rec = make_ai_recommender()
        candidates = rec._rag_retrieve("something energetic", n=4)
        scores = [c["rule_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_high_energy_query_surfaces_energetic_song(self):
        rec = make_ai_recommender()
        candidates = rec._rag_retrieve("intense workout pump hype", n=2)
        # Top candidate should have high energy
        top_energy = candidates[0]["song"]["energy"]
        assert top_energy >= 0.80

    def test_chill_query_surfaces_low_energy_song(self):
        rec = make_ai_recommender()
        candidates = rec._rag_retrieve("calm quiet ambient sleep", n=2)
        top_energy = candidates[0]["song"]["energy"]
        assert top_energy <= 0.50


# ---------------------------------------------------------------------------
# 5. Output schema validation
# ---------------------------------------------------------------------------

class TestOutputSchema:

    def test_build_output_empty_payload(self):
        rec = make_ai_recommender()
        recs, summary = rec._build_output(None, k=5)
        assert recs == []
        assert isinstance(summary, str)

    def test_build_output_enriches_with_song_data(self):
        rec = make_ai_recommender()
        payload = {
            "recommendations": [{"song_id": 1, "explanation": "Great match", "trade_offs": ""}],
            "summary": "Good fit overall.",
        }
        recs, summary = rec._build_output(payload, k=5)
        assert len(recs) == 1
        assert recs[0]["song"]["id"] == 1
        assert recs[0]["explanation"] == "Great match"
        assert summary == "Good fit overall."

    def test_build_output_skips_unknown_song_ids(self):
        rec = make_ai_recommender()
        payload = {
            "recommendations": [
                {"song_id": 999, "explanation": "ghost song"},
                {"song_id": 2,   "explanation": "real song"},
            ],
            "summary": "Mixed.",
        }
        recs, _ = rec._build_output(payload, k=5)
        # id 999 doesn't exist → should be skipped
        assert len(recs) == 1
        assert recs[0]["song"]["id"] == 2

    def test_build_output_respects_k(self):
        rec = make_ai_recommender()
        payload = {
            "recommendations": [
                {"song_id": i, "explanation": f"song {i}"} for i in [1, 2, 3, 4]
            ],
            "summary": "All good.",
        }
        recs, _ = rec._build_output(payload, k=2)
        assert len(recs) == 2


# ---------------------------------------------------------------------------
# 6. Integration test (skipped without real API key)
# ---------------------------------------------------------------------------

SKIP_INTEGRATION = not os.environ.get("ANTHROPIC_API_KEY") or \
                   os.environ.get("ANTHROPIC_API_KEY", "").startswith("test-")


@pytest.mark.skipif(SKIP_INTEGRATION, reason="ANTHROPIC_API_KEY not set or is a test key")
class TestIntegration:

    def test_recommend_returns_expected_shape(self):
        from src.ai_recommender import AIRecommender
        songs = load_songs("data/songs.csv")
        rec = AIRecommender(songs)
        result = rec.recommend("chill lofi beats for studying", k=3)

        assert "query" in result
        assert "recommendations" in result
        assert "summary" in result
        assert 1 <= len(result["recommendations"]) <= 3

        for r in result["recommendations"]:
            assert "song" in r
            assert "explanation" in r
            assert isinstance(r["explanation"], str)
            assert r["explanation"].strip() != ""

    def test_recommend_consistency(self):
        """Running the same query twice should always return the same #1 song."""
        from src.ai_recommender import AIRecommender
        songs = load_songs("data/songs.csv")
        rec = AIRecommender(songs)
        q = "high energy edm to get pumped up"
        r1 = rec.recommend(q, k=3)
        r2 = rec.recommend(q, k=3)
        # The top pick should be the same song (deterministic catalog + consistent Claude)
        assert r1["recommendations"][0]["song"]["id"] == r2["recommendations"][0]["song"]["id"]
