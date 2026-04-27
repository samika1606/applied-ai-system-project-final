# 🎧 Model Card: Music Recommender Simulation

---

## 1. Model Name

**VibeFinder 1.0**

A content-based music recommender that scores songs against a user taste profile and returns the best matches.

---

## 2. Goal / Task

VibeFinder recommends songs by directly comparing a user’s stated preferences to each song’s features and selecting the closest matches. It does not use listening history or other users’ data, and it is designed purely for classroom learning and experimentation rather than real-world use.

---

## 3. How the Model Works

Each song gets a score from 0 to 7.5. Higher score means better match. The score comes from five rules applied one at a time:

1. **Genre match.** If the song's genre matches what the user prefers, it gets 1 point. Genre captures the general style and production feel of the music.

2. **Mood match.** If the song's mood label matches what the user wants, it gets 1 point. Mood captures the emotional tone — happy, chill, intense, etc.

3. **Energy proximity.** This is the most important rule, worth up to 4 points. It rewards songs whose energy level is close to what the user asked for. A song that is exactly on target gets nearly 4 points. A song that is very far off gets almost nothing. The scoring curve drops off quickly, so small differences are forgiven but large ones are not.

4. **Acoustic texture.** If the user prefers acoustic sound and the song is mostly acoustic, it gets 1 point. If the user prefers polished electronic sound and the song fits that, it also gets 1 point. Songs in the middle get nothing from this rule.

5. **Valence tie-breaker.** A small bonus (0.5 points) for songs where the emotional brightness matches the mood label. For example, a "happy" song that also has high musical positivity scores a little higher than a "happy" song that sounds a bit flat.

Once every song has a score, they are sorted from highest to lowest. The top 5 are returned as recommendations, each with a plain-language explanation of which rules fired.

---

## 4. Data Used

The catalog contains 18 songs in a CSV file, each described by 10 attributes including genre, mood, and numerical features like energy, tempo, valence, danceability, and acousticness (all scaled 0.0–1.0). It was expanded from an initial 10-song set to include a wider range of genres, but most genres still only have one example.

---

## 5. Strengths
 It also provides clear and transparent explanations for each recommendation, breaking down how factors like genre and energy contribute to the final score, which helps users understand exactly why a song was selected.

---

## 6. Limitations and Bias
 Most mood labels are ignored by the tie-breaker rule, which only considers 4 of the 14 available moods. Overall, the system struggles with conflicting preferences because label-based rules outweigh numerical scoring like energy, causing contradictory requests to be resolved in biased ways.
---

## 7. Evaluation Process

Six user profiles were tested in total.

Standard profiles (High-Energy Pop, Chill Lofi, Deep Rock) worked well, while adversarial profiles exposed clear weaknesses. A major issue was a low-energy song ranking #1 for a high-energy request because genre and mood labels overpowered energy scoring. The system struggles with conflicting preferences, and adjusting weights fixes one issue but creates others. 

---

## 8. Intended Use and Non-Intended Use

This system is intended as a classroom tool to help students understand how content-based recommendation works by allowing them to read, modify, and intentionally break the scoring rules. It is not designed for real-world use, as it only includes 18 songs and lacks awareness of user context such as listening history, time, culture, or language, which would lead to poor and repetitive recommendations at scale.

---

## 9. Ideas for Improvement

Replace exact mood matching with mood similarity so related moods (like happy and euphoric) receive partial credit instead of scoring zero, making recommendations more nuanced. 

---

## 10. Personal Reflection

The biggest learning moment was realizing the system could be confused by its own rules: in the Acoustic but High-Energy case, a low-energy café jazz song ranked #1 because categorical rules (genre, mood, acoustic) outweighed the energy score, revealing a bias in the weighting design. AI tools helped generate ideas, math, and data, but many outputs still needed careful checking for correctness and consistency.