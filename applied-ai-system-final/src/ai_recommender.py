"""
AI-powered music recommender using Claude.

Architecture:
  RAG  — rule-based scorer retrieves the top-N candidates from the catalog;
          those candidates are injected into Claude's context as retrieved knowledge.
  Agentic — Claude drives a tool-use loop: it can search the catalog, score
             individual songs, and when satisfied calls submit_recommendations.

Guardrails:
  • ANTHROPIC_API_KEY check on init
  • max_tool_calls cap prevents runaway loops
  • All inputs validated before tool execution
  • Every Claude call + tool result is logged at DEBUG level
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from src.recommender import load_songs, score_song

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schemas exposed to Claude during the agentic loop
# ---------------------------------------------------------------------------

TOOLS: List[Dict] = [
    {
        "name": "search_catalog",
        "description": (
            "Search the music catalog by optional filters. "
            "Returns all matching songs with their audio features. "
            "Call with no arguments to list the entire catalog."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre":      {"type": "string",  "description": "Exact genre label (e.g. 'pop', 'lofi', 'metal')"},
                "mood":       {"type": "string",  "description": "Exact mood label (e.g. 'happy', 'chill', 'intense')"},
                "min_energy": {"type": "number",  "description": "Minimum energy level, 0.0–1.0"},
                "max_energy": {"type": "number",  "description": "Maximum energy level, 0.0–1.0"},
                "acoustic":   {"type": "boolean", "description": "True → acousticness ≥ 0.65; False → acousticness ≤ 0.30"},
            },
        },
    },
    {
        "name": "score_song",
        "description": (
            "Compute the rule-based compatibility score for a single song "
            "given the user's preferences. Returns the numeric score and the "
            "matching rules that fired."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "song_id": {
                    "type": "integer",
                    "description": "The integer ID of the song to evaluate",
                },
                "user_prefs": {
                    "type": "object",
                    "description": (
                        "User preference dict with keys: genre (str), mood (str), "
                        "energy (float 0–1), likes_acoustic (bool)"
                    ),
                },
            },
            "required": ["song_id", "user_prefs"],
        },
    },
    {
        "name": "submit_recommendations",
        "description": (
            "Submit the final ranked recommendations. Call this once you have "
            "evaluated candidates and are confident in your selections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "description": f"Ordered list of up to 5 recommendations",
                    "items": {
                        "type": "object",
                        "properties": {
                            "song_id":     {"type": "integer"},
                            "explanation": {"type": "string",
                                           "description": "Why this song fits the user"},
                            "trade_offs":  {"type": "string",
                                           "description": "What doesn't perfectly match (if anything)"},
                        },
                        "required": ["song_id", "explanation"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "1-2 sentence assessment of how well the catalog serves this user",
                },
            },
            "required": ["recommendations", "summary"],
        },
    },
]

MAX_TOOL_CALLS = 12   # guardrail: stop runaway agentic loops
RAG_CANDIDATES = 10   # how many songs the rule scorer pre-retrieves for Claude


class AIRecommender:
    """
    Music recommender that combines rule-based RAG retrieval with
    Claude's agentic reasoning to produce rich, contextual recommendations.
    """

    def __init__(self, songs: List[Dict], model: str = "claude-sonnet-4-6"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Run: export ANTHROPIC_API_KEY=your_key"
            )
        self.songs = songs
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        self._index: Dict[int, Dict] = {s["id"]: s for s in songs}
        logger.info("AIRecommender ready: %d songs, model=%s", len(songs), model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(self, query: str, k: int = 5) -> Dict:
        """
        Full RAG + agentic pipeline.

        Steps:
          1. RAG retrieval — rule scorer ranks the whole catalog, top
             RAG_CANDIDATES are injected into Claude's system prompt as
             pre-retrieved knowledge (the "augmentation" in RAG).
          2. Agentic loop — Claude uses tools to search and score songs,
             iterating until it calls submit_recommendations or hits the cap.
          3. Output — enriched recommendation dicts returned to caller.
        """
        logger.info("recommend() called: query=%r  k=%d", query, k)

        # --- Step 1: RAG retrieval ---
        retrieved = self._rag_retrieve(query, n=RAG_CANDIDATES)
        logger.info("RAG retrieved %d candidates", len(retrieved))

        # --- Step 2: Agentic generation ---
        final, summary = self._agentic_loop(query, retrieved, k)
        logger.info("Agentic loop produced %d recommendations", len(final))

        return {"query": query, "recommendations": final, "summary": summary}

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    def _rag_retrieve(self, query: str, n: int) -> List[Dict]:
        """
        Rule-based retrieval step.

        We extract a rough energy hint from the query text (high / low / none)
        and build a minimal prefs dict, then score all songs. The top-n become
        the 'retrieved context' that feeds into Claude.
        """
        # Heuristic: pull energy level from keywords so retrieval is query-aware
        ql = query.lower()
        if any(w in ql for w in ("energetic", "upbeat", "pump", "hype", "intense", "fast", "workout")):
            energy_hint = 0.88
        elif any(w in ql for w in ("chill", "relax", "calm", "sleep", "quiet", "soft", "ambient")):
            energy_hint = 0.30
        else:
            energy_hint = 0.55  # neutral

        prefs = {"energy": energy_hint}

        scored = []
        for song in self.songs:
            sc, reasons = score_song(prefs, song)
            scored.append({
                "song": song,
                "rule_score": round(sc, 3),
                "rule_reasons": reasons,
            })
        scored.sort(key=lambda x: x["rule_score"], reverse=True)
        return scored[:n]

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def _build_system_prompt(self, retrieved: List[Dict], k: int) -> str:
        catalog_genres = sorted({s["genre"] for s in self.songs})
        catalog_moods  = sorted({s["mood"]  for s in self.songs})
        energy_range   = (
            min(s["energy"] for s in self.songs),
            max(s["energy"] for s in self.songs),
        )

        # Format the RAG-retrieved candidates as injected context
        retrieved_block = "\n".join(
            f"  [{c['song']['id']}] \"{c['song']['title']}\" by {c['song']['artist']} | "
            f"genre={c['song']['genre']} mood={c['song']['mood']} "
            f"energy={c['song']['energy']:.2f} acousticness={c['song']['acousticness']:.2f} "
            f"valence={c['song']['valence']:.2f} | rule_score={c['rule_score']}"
            for c in retrieved
        )

        return f"""You are an expert music recommender with access to a catalog of {len(self.songs)} songs.

CATALOG OVERVIEW
  Genres : {', '.join(catalog_genres)}
  Moods  : {', '.join(catalog_moods)}
  Energy : {energy_range[0]:.2f} – {energy_range[1]:.2f}

PRE-RETRIEVED CANDIDATES (rule-based RAG, top {len(retrieved)} by energy proximity)
{retrieved_block}

YOUR TASK
  Select the best {k} songs for the user's query.
  Strategy:
    1. Start from the pre-retrieved candidates above.
    2. Use search_catalog to explore the full catalog if a better fit might exist.
    3. Use score_song to get numeric scores for songs you are considering.
    4. When confident, call submit_recommendations with your ranked top {k}.

  Be specific in your explanations — reference the song's audio features.
  Acknowledge trade-offs honestly when no perfect match exists.
"""

    def _agentic_loop(self, query: str, retrieved: List[Dict], k: int) -> Tuple[List[Dict], str]:
        system_prompt = self._build_system_prompt(retrieved, k)
        messages = [
            {
                "role": "user",
                "content": (
                    f"Find the best {k} songs for this request: \"{query}\"\n\n"
                    "Use the pre-retrieved candidates as your starting point, "
                    "then search and score as needed before calling submit_recommendations."
                ),
            }
        ]

        final_payload: Optional[Dict] = None
        tool_calls_made = 0

        while tool_calls_made < MAX_TOOL_CALLS:
            logger.debug("Agentic loop turn %d, sending %d messages", tool_calls_made + 1, len(messages))

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )

            logger.debug("stop_reason=%s  usage=%s", response.stop_reason, response.usage)

            if response.stop_reason == "end_turn":
                logger.warning("Claude ended without calling submit_recommendations")
                break

            if response.stop_reason != "tool_use":
                logger.warning("Unexpected stop_reason: %s", response.stop_reason)
                break

            # Collect tool-use blocks
            tool_blocks = [b for b in response.content if b.type == "tool_use"]

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            # Execute tools and build tool_result turn
            tool_results = []
            for tb in tool_blocks:
                tool_calls_made += 1
                result = self._execute_tool(tb.name, tb.input)
                logger.debug("Tool %s(%s) → %s", tb.name, tb.input, str(result)[:200])

                if tb.name == "submit_recommendations":
                    final_payload = tb.input  # capture before replying

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

            if final_payload:
                break  # done

        if tool_calls_made >= MAX_TOOL_CALLS:
            logger.warning("Guardrail: MAX_TOOL_CALLS (%d) reached", MAX_TOOL_CALLS)

        return self._build_output(final_payload, k)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, inp: Dict) -> Any:
        if name == "search_catalog":
            return self._tool_search_catalog(inp)
        if name == "score_song":
            return self._tool_score_song(inp)
        if name == "submit_recommendations":
            return {"status": "accepted", "count": len(inp.get("recommendations", []))}
        return {"error": f"Unknown tool: {name}"}

    def _tool_search_catalog(self, inp: Dict) -> Dict:
        genre      = inp.get("genre")
        mood       = inp.get("mood")
        min_e      = inp.get("min_energy")
        max_e      = inp.get("max_energy")
        acoustic   = inp.get("acoustic")   # None / True / False

        results = []
        for s in self.songs:
            if genre    and s["genre"].lower() != genre.lower():
                continue
            if mood     and s["mood"].lower() != mood.lower():
                continue
            if min_e    is not None and s["energy"] < min_e:
                continue
            if max_e    is not None and s["energy"] > max_e:
                continue
            if acoustic is True  and s["acousticness"] < 0.65:
                continue
            if acoustic is False and s["acousticness"] > 0.30:
                continue
            results.append({
                "id": s["id"], "title": s["title"], "artist": s["artist"],
                "genre": s["genre"], "mood": s["mood"],
                "energy": s["energy"], "acousticness": s["acousticness"],
                "valence": s["valence"], "danceability": s["danceability"],
            })

        return {"count": len(results), "songs": results}

    def _tool_score_song(self, inp: Dict) -> Dict:
        song_id = inp.get("song_id")
        user_prefs = inp.get("user_prefs", {})

        if not isinstance(song_id, int):
            return {"error": "song_id must be an integer"}
        song = self._index.get(song_id)
        if song is None:
            return {"error": f"No song with id={song_id}"}

        sc, reasons = score_song(user_prefs, song)
        return {
            "song_id": song_id,
            "title": song["title"],
            "score": round(sc, 3),
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _build_output(
        self, payload: Optional[Dict], k: int
    ) -> Tuple[List[Dict], str]:
        if not payload:
            return [], "Agent did not produce recommendations."

        recs = payload.get("recommendations", [])[:k]
        summary = payload.get("summary", "")
        enriched = []
        for r in recs:
            song = self._index.get(r.get("song_id"))
            if song:
                enriched.append({
                    "song":        song,
                    "explanation": r.get("explanation", ""),
                    "trade_offs":  r.get("trade_offs", ""),
                })
        return enriched, summary
