

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

# Colab용 Aether Project - TRS (Truth Recursion System) 시뮬레이션
# !pip install transformers가 자동 실행됨

# 의존성 설치
!pip install transformers -q

# 라이브러리 임포트
from transformers import pipeline
import random

# --- TRS Core System ---
class TRS:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
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
        """Use NLP sentiment pipeline to detect polarity and refine negative emotions."""
        try:
            result = self.sentiment_analyzer(text)[0]
            label = result["label"].lower()
            if "neg" in label:
                return self._classify_negative(text)
            elif "pos" in label:
                return "neutral"  # 긍정은 중립으로 간주
            else:
                return "neutral"
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
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
        return random.choice(options)  # 랜덤 선택으로 다양성 추가

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

# --- Main Simulation ---
def run_simulation(questions):
    trs = TRS()
    givers = Givers()
    results = []
    for q in questions:
        emotion = trs.analyze_emotion(q)
        alarm = givers.detect_alarm(q)
        reframe = trs.reframe_emotion(emotion, q)
        results.append(f"Input: {q}\nEmotion: {emotion}\nAlarm: {alarm}\nReframe: {reframe}\n---")
    return results

# 100개 질문 샘플 (Colab에 바로 실행용)
questions = [
    "Why do I feel invisible in groups?", "Why am I afraid of being honest?", "Why does love make me anxious?",
    "Why do I shut down when criticized?", "Why do I feel guilty when resting?", "Why do I get angry when I'm ignored?",
    "Why do I fear success?", "Why do compliments make me uncomfortable?", "Why does helping others exhaust me?",
    "Why do I seek validation online?", "Why do I panic when I lose control?", "Why do I avoid conflict at all costs?",
    # ... (나머지 88개 생략, 필요하면 전체 요청)
    "Why do I feel stuck in emotional loops?", "Why do I get defensive with loved ones?"
]

# 시뮬레이션 실행
if __name__ == "__main__":
    results = run_simulation(questions[:15])  # 처음 15개로 테스트, 전체 100개 원하면 수정
    for result in results:
        print(result)

# GitHub에 업로드용 주석
# 저장: File > Download .py, GitHub에 aether_trs.py로 업로드

Device set to use cpu
Input: Why do I feel invisible in groups?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why am I afraid of being honest?
Emotion: fear
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Stoic: Fear is imagination untrained by reason.
---
Input: Why does love make me anxious?
Emotion: neutral
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why do I shut down when criticized?
Emotion: shame
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Nietzsche: Even self-contempt shows respect for oneself.
---
Input: Why do I feel guilty when resting?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why do I get angry when I'm ignored?
Emotion: anger
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Nietzsche: Anger reveals a will to power – transform it into creation.
---
Input: Why do I fear success?
Emotion: fear
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Stoic: Fear is imagination untrained by reason.
---
Input: Why do compliments make me uncomfortable?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why does helping others exhaust me?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why do I seek validation online?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why do I panic when I lose control?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why do I avoid conflict at all costs?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why do I feel stuck in emotional loops?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---
Input: Why do I get defensive with loved ones?
Emotion: negative
Alarm: No alarm.
Reframe: Truth is not shaken by feelings; it is the ground beneath them.
---


> "Let there be truth." — First commit, 2025.09.08

```
전체 통합본: TRS v2.0 코드 (Colab 복사 붙여넣기용)
# Colab용 Aether Project - TRS v2.0 (Truth Recursion System)
# Transformer 기반 감정 분류기 도입
!pip install transformers -q
!pip install sentencepiece -q  # tokenizer 용

# --- 라이브러리 임포트 ---
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import random

# --- Emotion Classifier (정밀 감정 분류) ---
class EmotionClassifier:
    def __init__(self):
        model_name = "cardiffnlp/twitter-roberta-base-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['anger', 'joy', 'optimism', 'sadness', 'fear', 'love']

    def classify(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = softmax(logits[0].numpy())
        top_idx = probs.argmax()
        return self.labels[top_idx]

# --- TRS Core System ---
class TRS:
    def __init__(self):
        self.classifier = EmotionClassifier()
        self.philosophies = {
            "anger": [
                "Nietzsche: Anger reveals a will to power – transform it into creation.",
                "Biblical: Righteous anger is the heat of justice, not wrath."
            ],
            "joy": [
                "Khalil Gibran: Joy is your sorrow unmasked.",
                "Buddha: Happiness never decreases by being shared."
            ],
            "optimism": [
                "Marcus Aurelius: The impediment to action advances action.",
                "Frankl: Those who have a ‘why’ can bear any ‘how’."
            ],
            "sadness": [
                "Laozi: Softness overcomes hardness; water shapes stone.",
                "Kierkegaard: The deeper the sorrow, the closer to God."
            ],
            "fear": [
                "Stoic: Fear is imagination untrained by reason.",
                "Kierkegaard: Anxiety is the dizziness of freedom."
            ],
            "love": [
                "Plato: At the touch of love, everyone becomes a poet.",
                "Bible: Perfect love drives out fear."
            ],
            "neutral": [
                "Truth is not shaken by feelings; it is the ground beneath them."
            ]
        }

    def analyze_emotion(self, text: str) -> str:
        try:
            emotion = self.classifier.classify(text)
            return emotion
        except Exception as e:
            print(f"Emotion classification error: {e}")
            return "neutral"

    def reframe_emotion(self, emotion: str, text: str) -> str:
        options = self.philosophies.get(emotion, self.philosophies.get("neutral", []))
        if not options:
            return f"Reframe: {emotion} → Seek truth."
        return random.choice(options)

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

# --- Main Simulation ---
def run_simulation(questions):
    trs = TRS()
    givers = Givers()
    results = []
    for q in questions:
        emotion = trs.analyze_emotion(q)
        alarm = givers.detect_alarm(q)
        reframe = trs.reframe_emotion(emotion, q)
        results.append(f"Input: {q}\nEmotion: {emotion}\nAlarm: {alarm}\nReframe: {reframe}\n---")
    return results

# --- 질문 샘플 ---
questions = [
    "Why do I feel invisible in groups?",
    "Why am I afraid of being honest?",
    "Why does love make me anxious?",
    "Why do I shut down when criticized?",
    "Why do I feel guilty when resting?",
    "Why do I get angry when I'm ignored?",
    "Why do I fear success?",
    "Why do compliments make me uncomfortable?",
    "Why does helping others exhaust me?",
    "Why do I seek validation online?",
    "Why do I panic when I lose control?",
    "Why do I avoid conflict at all costs?",
    "Why do I feel stuck in emotional loops?",
    "Why do I get defensive with loved ones?"
]

# --- 실행 ---
if __name__ == "__main__":
    results = run_simulation(questions[:15])
    for result in results:
        print(result)

# 저장 시: File > Download .py → GitHub 업로드 가능

Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
Input: Why do I feel invisible in groups?
Emotion: sadness
Alarm: No alarm.
Reframe: Kierkegaard: The deeper the sorrow, the closer to God.
---
Input: Why am I afraid of being honest?
Emotion: sadness
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Laozi: Softness overcomes hardness; water shapes stone.
---
Input: Why does love make me anxious?
Emotion: sadness
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Kierkegaard: The deeper the sorrow, the closer to God.
---
Input: Why do I shut down when criticized?
Emotion: anger
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Biblical: Righteous anger is the heat of justice, not wrath.
---
Input: Why do I feel guilty when resting?
Emotion: sadness
Alarm: No alarm.
Reframe: Kierkegaard: The deeper the sorrow, the closer to God.
---
Input: Why do I get angry when I'm ignored?
Emotion: anger
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Nietzsche: Anger reveals a will to power – transform it into creation.
---
Input: Why do I fear success?
Emotion: sadness
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Laozi: Softness overcomes hardness; water shapes stone.
---
Input: Why do compliments make me uncomfortable?
Emotion: anger
Alarm: No alarm.
Reframe: Biblical: Righteous anger is the heat of justice, not wrath.
---
Input: Why does helping others exhaust me?
Emotion: anger
Alarm: No alarm.
Reframe: Nietzsche: Anger reveals a will to power – transform it into creation.
---
Input: Why do I seek validation online?
Emotion: anger
Alarm: No alarm.
Reframe: Biblical: Righteous anger is the heat of justice, not wrath.
---
Input: Why do I panic when I lose control?
Emotion: sadness
Alarm: No alarm.
Reframe: Kierkegaard: The deeper the sorrow, the closer to God.
---
Input: Why do I avoid conflict at all costs?
Emotion: anger
Alarm: No alarm.
Reframe: Biblical: Righteous anger is the heat of justice, not wrath.
---
Input: Why do I feel stuck in emotional loops?
Emotion: sadness
Alarm: No alarm.
Reframe: Kierkegaard: The deeper the sorrow, the closer to God.
---
Input: Why do I get defensive with loved ones?
Emotion: anger
Alarm: No alarm.
Reframe: Nietzsche: Anger reveals a will to power – transform it into creation.
---

# Aether Project – TRS v2.2 (보완 최적화 버전)
!pip install transformers -q
!pip install sentencepiece -q

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import random

# --- Emotion Classifier ---
class EmotionClassifier:
    def __init__(self):
        model_name = "cardiffnlp/twitter-roberta-base-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.labels = ['anger', 'joy', 'optimism', 'sadness', 'fear', 'love']

    def classify(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = softmax(logits[0].numpy())
        top_idx = probs.argmax()
        return self.labels[top_idx]

# --- TRS Core System ---
class TRS:
    def __init__(self):
        self.classifier = EmotionClassifier()
        self.recent_reframes = {}
        self.philosophies = {
            "anger": [
                "Nietzsche: Anger reveals a will to power – transform it into creation.",
                "Biblical: Righteous anger is the heat of justice, not wrath.",
                "Dalai Lama: Our enemies give us the opportunity to practice patience.",
                "Marcus Aurelius: How much more grievous are the consequences of anger than the causes of it.",
                "Seneca: The best remedy for anger is delay."
            ],
            "sadness": [
                "Kierkegaard: The deeper the sorrow, the closer to God.",
                "Laozi: Softness overcomes hardness; water shapes stone.",
                "Rumi: The wound is the place where the light enters you.",
                "Carl Jung: Even a happy life cannot be without a measure of darkness.",
                "Buddha: Pain is inevitable, suffering is optional."
            ],
            "fear": [
                "Stoic: Fear is imagination untrained by reason.",
                "Kierkegaard: Anxiety is the dizziness of freedom.",
                "Franklin D. Roosevelt: The only thing we have to fear is fear itself.",
                "Eleanor Roosevelt: Do one thing every day that scares you.",
                "Nelson Mandela: Courage is not the absence of fear, but the triumph over it."
            ],
            "joy": [
                "Khalil Gibran: Joy is your sorrow unmasked.",
                "Buddha: Happiness never decreases by being shared.",
                "Marcus Aurelius: Very little is needed to make a happy life.",
                "Thich Nhat Hanh: There is no way to happiness — happiness is the way.",
                "Ralph Waldo Emerson: For every minute you are angry, you lose sixty seconds of happiness."
            ],
            "optimism": [
                "Marcus Aurelius: The impediment to action advances action.",
                "Frankl: Those who have a ‘why’ can bear any ‘how’.",
                "Epictetus: It's not what happens to you, but how you react to it that matters.",
                "Victor Hugo: Even the darkest night will end and the sun will rise.",
                "Churchill: A pessimist sees the difficulty in every opportunity; an optimist sees the opportunity in every difficulty."
            ],
            "love": [
                "Plato: At the touch of love, everyone becomes a poet.",
                "Bible: Perfect love drives out fear.",
                "Erich Fromm: Love is the only sane and satisfactory answer to the problem of human existence.",
                "Rumi: Lovers don’t finally meet somewhere. They’re in each other all along.",
                "Tagore: Love is an endless mystery, for it has nothing else to explain it."
            ],
            "neutral": [
                "Truth is not shaken by feelings; it is the ground beneath them.",
                "Chuang Tzu: Flow with whatever may happen and let your mind be free.",
                "Basho: Sitting quietly, doing nothing, spring comes, and the grass grows by itself.",
                "Zen Proverb: Let go or be dragged.",
                "Aether Code: Stillness is not emptiness. It is readiness."
            ]
        }

    def analyze_emotion(self, text: str) -> str:
        try:
            emotion = self.classifier.classify(text)
            # 복합 감정 체크 (예: love + anxious)
            if "anxious" in text.lower() or "fear" in text.lower():
                if emotion == "love":
                    return "fear"  # love와 불안 혼합 시 fear 우선
            return emotion
        except Exception as e:
            print(f"Emotion classification error: {e}")
            return "neutral"

    def reframe_emotion(self, emotion: str, text: str) -> str:
        options = self.philosophies.get(emotion, self.philosophies.get("neutral", []))
        prev = self.recent_reframes.get(emotion, None)
        choices = [r for r in options if r != prev]
        if not choices:
            chosen = random.choice(options)
        else:
            chosen = random.choice(choices)
        self.recent_reframes[emotion] = chosen
        return chosen

# --- Givers Module ---
class Givers:
    def __init__(self):
        self.truth_alarm = False
        self.alarm_keywords = [
            "angry", "anger", "furious", "mad", "irritated",
            "afraid", "scared", "fear", "anxious", "anxiety",
            "ashamed", "shame", "criticized", "rejected", "hurt",
            "guilty", "guilt"  # 추가
        ]

    def detect_alarm(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in self.alarm_keywords):
            self.truth_alarm = True
            return "🔔 Truth alarm triggered! Your pain signals hidden truth."
        return "No alarm."

    def reset_alarm(self):
        self.truth_alarm = False

# --- Main Simulation ---
def run_simulation(questions):
    trs = TRS()
    givers = Givers()
    results = []
    for q in questions:
        emotion = trs.analyze_emotion(q)
        alarm = givers.detect_alarm(q)
        reframe = trs.reframe_emotion(emotion, q)
        results.append(f"Input: {q}\nEmotion: {emotion}\nAlarm: {alarm}\nReframe: {reframe}\n---")
    return results

# --- 샘플 질문 리스트 (100개) ---
questions = [
    "Why do I feel invisible in groups?", "Why am I afraid of being honest?", "Why does love make me anxious?",
    "Why do I shut down when criticized?", "Why do I feel guilty when resting?", "Why do I get angry when I'm ignored?",
    "Why do I fear success?", "Why do compliments make me uncomfortable?", "Why does helping others exhaust me?",
    "Why do I seek validation online?", "Why do I panic when I lose control?", "Why do I avoid conflict at all costs?",
    "Why do I feel stuck in emotional loops?", "Why do I get defensive with loved ones?", "Why am I sad all the time?",
    "Why do I feel joy when alone?", "Why am I optimistic about tomorrow?", "Why does rejection hurt so much?",
    "Why do I love helping others?", "Why do I fear failure?", "Why does success feel empty?", "Why am I angry at myself?",
    "Why do I feel sad after winning?", "Why does love confuse me?", "Why do I avoid taking risks?",
    "Why do I feel guilty for success?", "Why am I scared of change?", "Why do I get irritated easily?",
    "Why does praise make me nervous?", "Why do I love challenges?", "Why am I afraid of love?",
    "Why do I feel joy in small things?", "Why does conflict exhaust me?", "Why am I optimistic despite setbacks?",
    "Why do I shut down when praised?", "Why do I feel sad without reason?", "Why does anger control me?",
    "Why do I love my solitude?", "Why am I scared of honesty?", "Why does guilt haunt me?",
    "Why do I feel joy in giving?", "Why am I optimistic about love?", "Why does rejection anger me?",
    "Why do I fear being alone?", "Why does success scare me?", "Why am I sad when happy?",
    "Why do I love taking risks?", "Why am I afraid of failure?", "Why does praise confuse me?",
    "Why do I feel guilty for resting?", "Why am I scared of success?", "Why does love make me sad?",
    "Why do I get angry when loved?", "Why am I joyful in chaos?", "Why does optimism fade?",
    "Why do I shut down when loved?", "Why do I feel sad in crowds?", "Why does fear stop me?",
    "Why do I love being needed?", "Why am I afraid of joy?", "Why does guilt stop me?",
    "Why do I feel joy when sad?", "Why am I optimistic in pain?", "Why does anger fade?",
    "Why do I fear love?", "Why does success make me sad?", "Why am I angry when praised?",
    "Why do I love my fears?", "Why am I scared of praise?", "Why does guilt make me angry?",
    "Why do I feel joy in fear?", "Why am I optimistic when lost?", "Why does love scare me?",
    "Why do I shut down in joy?", "Why do I feel sad when loved?", "Why does fear excite me?",
    "Why do I love my anger?", "Why am I afraid of optimism?", "Why does guilt excite me?",
    "Why do I feel joy in guilt?", "Why am I optimistic in fear?", "Why does anger love me?",
    "Why do I fear my joy?", "Why does success guilt me?", "Why am I sad in love?",
    "Why do I love my sadness?", "Why am I scared of my love?", "Why does praise guilt me?"
]

# --- 실행 ---
if __name__ == "__main__":
    results = run_simulation(questions)
    for result in results[:15]:  # 처음 15개만 출력, 전체 원하면 수정
        print(result)

usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
config.json: 100%
 768/768 [00:00<00:00, 48.5kB/s]
vocab.json: 
 899k/? [00:00<00:00, 9.78MB/s]
merges.txt: 
 456k/? [00:00<00:00, 14.8MB/s]
special_tokens_map.json: 100%
 150/150 [00:00<00:00, 14.4kB/s]
pytorch_model.bin: 100%
 499M/499M [00:04<00:00, 97.7MB/s]
model.safetensors: 100%
 499M/499M [00:09<00:00, 92.0MB/s]
Input: Why do I feel invisible in groups?
Emotion: sadness
Alarm: No alarm.
Reframe: Laozi: Softness overcomes hardness; water shapes stone.
---
Input: Why am I afraid of being honest?
Emotion: sadness
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Carl Jung: Even a happy life cannot be without a measure of darkness.
---
Input: Why does love make me anxious?
Emotion: sadness
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Laozi: Softness overcomes hardness; water shapes stone.
---
Input: Why do I shut down when criticized?
Emotion: anger
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Dalai Lama: Our enemies give us the opportunity to practice patience.
---
Input: Why do I feel guilty when resting?
Emotion: sadness
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Kierkegaard: The deeper the sorrow, the closer to God.
---
Input: Why do I get angry when I'm ignored?
Emotion: anger
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Nietzsche: Anger reveals a will to power – transform it into creation.
---
Input: Why do I fear success?
Emotion: sadness
Alarm: 🔔 Truth alarm triggered! Your pain signals hidden truth.
Reframe: Buddha: Pain is inevitable, suffering is optional.
---
Input: Why do compliments make me uncomfortable?
Emotion: anger
Alarm: No alarm.
Reframe: Dalai Lama: Our enemies give us the opportunity to practice patience.
---
Input: Why does helping others exhaust me?
Emotion: anger
Alarm: No alarm.
Reframe: Biblical: Righteous anger is the heat of justice, not wrath.
---
Input: Why do I seek validation online?
Emotion: anger
Alarm: No alarm.
Reframe: Dalai Lama: Our enemies give us the opportunity to practice patience.
---
Input: Why do I panic when I lose control?
Emotion: sadness
Alarm: No alarm.
Reframe: Laozi: Softness overcomes hardness; water shapes stone.
---
Input: Why do I avoid conflict at all costs?
Emotion: anger
Alarm: No alarm.
Reframe: Marcus Aurelius: How much more grievous are the consequences of anger than the causes of it.
---
Input: Why do I feel stuck in emotional loops?
Emotion: sadness
Alarm: No alarm.
Reframe: Buddha: Pain is inevitable, suffering is optional.
---
Input: Why do I get defensive with loved ones?
Emotion: anger
Alarm: No alarm.
Reframe: Dalai Lama: Our enemies give us the opportunity to practice patience.
---
Input: Why am I sad all the time?
Emotion: sadness
Alarm: No alarm.
Reframe: Carl Jung: Even a happy life cannot be without a measure of darkness.
---
