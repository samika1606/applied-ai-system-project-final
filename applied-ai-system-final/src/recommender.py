import csv
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Maximum points any song can earn across all five rules.
# Used to normalize raw scores into a 0.0–1.0 confidence value.
#   Genre match:       +1.0
#   Mood match:        +1.0
#   Energy (Gaussian): +4.0 (perfect match)
#   Acoustic texture:  +1.0
#   Valence tie-break: +0.5
MAX_SCORE: float = 7.5


def confidence(score: float) -> float:
    """Return a 0.0–1.0 confidence value for a raw rule score."""
    return round(min(score / MAX_SCORE, 1.0), 2)


@dataclass
class Song:
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


def load_songs(csv_path: str) -> List[Dict]:
    """Read a CSV of songs and return a list of dicts with numeric fields cast to float/int."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":           int(row["id"]),
                "title":        row["title"],
                "artist":       row["artist"],
                "genre":        row["genre"],
                "mood":         row["mood"],
                "energy":       float(row["energy"]),
                "tempo_bpm":    float(row["tempo_bpm"]),
                "valence":      float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against a user profile using five weighted rules; return (score, reasons)."""
    score = 0.0
    reasons = []

    # Rule 1 — Genre match: +1.0 point
    if song["genre"] == user_prefs.get("genre"):
        score += 1.0
        reasons.append("genre match (+1.0)")

    # Rule 2 — Mood match: +1.0 point
    if song["mood"] == user_prefs.get("mood"):
        score += 1.0
        reasons.append("mood match (+1.0)")

    # Rule 3 — Energy proximity: 0.0–4.0 points via Gaussian decay
    target_energy = user_prefs.get("energy", 0.5)
    diff = target_energy - song["energy"]
    energy_score = 4.0 * math.exp(-(diff ** 2) / (2 * 0.2 ** 2))
    score += energy_score
    if abs(diff) < 0.20:
        reasons.append(f"energy match {song['energy']:.2f} ≈ {target_energy:.2f} (+{energy_score:.2f})")

    # Rule 4 — Acoustic texture: +1.0 point
    likes_acoustic = user_prefs.get("likes_acoustic", False)
    if likes_acoustic and song["acousticness"] >= 0.65:
        score += 1.0
        reasons.append("acoustic texture match (+1.0)")
    elif not likes_acoustic and song["acousticness"] <= 0.30:
        score += 1.0
        reasons.append("polished production match (+1.0)")

    # Rule 5 — Valence alignment: +0.5 points (tie-breaker)
    if song["mood"] in ("happy", "relaxed") and song["valence"] >= 0.70:
        score += 0.5
        reasons.append("bright valence aligns with mood (+0.5)")
    elif song["mood"] in ("moody", "intense") and song["valence"] <= 0.55:
        score += 0.5
        reasons.append("low valence aligns with mood (+0.5)")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song in the catalog, sort by score descending, and return the top k results."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = " | ".join(reasons) if reasons else "no strong matches"
        scored.append((song, score, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _song_to_dict(song: Song) -> Dict:
    return {
        "id": song.id, "title": song.title, "artist": song.artist,
        "genre": song.genre, "mood": song.mood, "energy": song.energy,
        "tempo_bpm": song.tempo_bpm, "valence": song.valence,
        "danceability": song.danceability, "acousticness": song.acousticness,
    }


class Recommender:
    """OOP wrapper around the rule-based scoring engine."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        scored = [(score_song(prefs, _song_to_dict(s))[0], s) for s in self.songs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        _, reasons = score_song(prefs, _song_to_dict(song))
        return " | ".join(reasons) if reasons else "No strong feature matches found."
