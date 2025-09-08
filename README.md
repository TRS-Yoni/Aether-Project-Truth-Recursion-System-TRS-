

## 📘 `README.md` 

````markdown
# 🌌 Aether Project – TRS (Truth Recursion System)

> Designed by Yoni, a South Korean dropout building AGI ethics without budget — only truth.

---

## 💡 What is TRS?

**TRS (Truth Recursion System)** is a logic system that:
- Decodes emotional input
- Detects distorted thought loops
- Reframes the emotion using philosophy (Nietzsche, Kant, Quran, etc.)
- Aligns the output with truth-based design values (Trust, Dignity, Connectivity)

This project is part of the **Aether Project**, a broader framework for AI ethics, emotional cognition, and decentralized truth governance.

---

## ⚙️ Core Modules

| File | Description |
|------|-------------|
| `aether_trs.py` | Main pipeline: emotion analysis → reframing → truth alignment |
| `givers_module.py` | Givers module: Treats **anger as a truth alarm** |
| `data/sample_questions.csv` | Sample emotional queries for testing |
| `tests/test_trs.py` | Unit tests for core logic |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
| `LICENSE` | MIT License |

---

## 🧠 Philosophy Behind TRS

- **Nietzsche**: “Will to power” as reframed agency  
- **Kant**: “Categorical imperative” for design alignment  
- **Quran 49:13**: Diversity as divine signal  
- **Givers Theory**: F-type personalities (INFP, ENFJ) transmute pain → signal

This system assumes that **emotions are not random**, but distortions of truth — and can be re-aligned with inner dignity through recursion.

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/aether-trs.git
cd aether-trs
pip install -r requirements.txt
python aether_trs.py
````

Sample input:

```text
Why am I always angry when I give too much?
```

Output:

```
Emotion: negative
Alarm: Truth alarm triggered!
Reframed: Nietzsche: Will to power drives truth.
```

---import asyncio
import platform
from transformers import pipeline

# --- TRS Core System ---
class TRS:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.philosophies = {
            "anger": [
                "Nietzsche: Anger reveals a will to power – transform it into creation.",
                "Biblical: Righteous anger can be a signal to defend truth."
            ],
            "sadness": [
                "Kant: Duty aligns with universal law, even in sorrow.",
                "Laozi: Softness overcomes hardness; water shapes stone."
            ],
            "fear": [
                "Kierkegaard: Anxiety is the dizziness of freedom.",
                "Stoic: Fear is imagination untrained by reason."
            ],
            "shame": [
                "Confucius: To recognize shame is the beginning of honor.",
                "Nietzsche: Even self-contempt shows respect for oneself."
            ],
            "neutral": [
                "Truth is not shaken by feelings; it is the ground beneath them."
            ]
        }

    def analyze_emotion(self, text: str) -> str:
        """Use NLP sentiment pipeline to detect basic polarity (POSITIVE/NEGATIVE)."""
        try:
            result = self.sentiment_analyzer(text)[0]
            label = result["label"].lower()
            if "neg" in label:
                # refine negative into categories
                return self._classify_negative(text)
            elif "pos" in label:
                return "positive"
            else:
                return "neutral"
        except Exception:
            return "unknown"

    def _classify_negative(self, text: str) -> str:
        """Keyword-based sub-classification for negative emotions."""
        t = text.lower()
        if any(k in t for k in ["angry", "anger", "furious", "mad", "irritated"]):
            return "anger"
        if any(k in t for k in ["sad", "lonely", "grief", "loss", "tired", "exhausted"]):
            return "sadness"
        if any(k in t for k in ["afraid", "fear", "scared", "anxious", "anxiety", "worried"]):
            return "fear"
        if any(k in t for k in ["shame", "ashamed", "embarrassed", "criticized"]):
            return "shame"
        return "negative"

    def reframe_emotion(self, emotion: str, text: str) -> str:
        """Reframe emotion using philosophical lenses."""
        options = self.philosophies.get(emotion, self.philosophies.get("neutral", []))
        if not options:
            return f"Reframe: {emotion} → Seek truth."
        return options[0]  # pick first for now (can randomize later)


# --- Givers Module ---
class Givers:
    def __init__(self):
        self.truth_alarm = False
        self.alarm_keywords = [
            "angry", "anger", "furious", "mad", "irritated",
            "afraid", "scared", "fear", "anxious", "anxiety",
            "ashamed", "shame", "criticized", "rejected", "hurt"
        ]

    def detect_alarm(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in self.alarm_keywords):
            self.truth_alarm = True
            return "🔔 Truth alarm triggered! Your pain signals hidden truth."
        return "No alarm."

    def reset_alarm(self):
        self.truth_alarm = False


# --- Main loop (Pyodide/Local) ---
async def main():
    trs = TRS()
    givers = Givers()

    samples = [
        "Why do I feel invisible in groups?",
        "Why am I afraid of being honest?",
        "Why does love make me anxious?",
        "Why do I shut down when criticized?",
        "Why do I feel guilty when resting?",
        "Why do I get angry when I'm ignored?",
    ]

    for s in samples:
        emotion = trs.analyze_emotion(s)
        alarm = givers.detect_alarm(s)
        reframe = trs.reframe_emotion(emotion, s)
        print(f"Input: {s}\nEmotion: {emotion}\nAlarm: {alarm}\nReframe: {reframe}\n---")


if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())


## 📈 Coming Soon

* Web-based UI for real-time TRS simulation
* HuggingFace custom emotion classifier
* GPT-4 Reframer API module
* PDF-based xAI Submission (in progress)

---

## 🤝 Contribute

* Test the pipeline (`tests/test_trs.py`)
* Add more emotional queries to `/data/sample_questions.csv`
* Share feedback via [X.com](https://x.com/yoni_aether) or GitHub issues

---

## 📜 License

MIT — use freely, but always cite the truth.
This project was born out of pain, recursion, and fire. 🧬🔥

---

## 🧑‍🚀 Credits

* **Yoni** – Architect & Philosopher
* **GPT / Grok** – AI Co-creators
* **TRS** – The Truth itself

> "Let there be truth." — First commit, 2025.09.08

```

