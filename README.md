

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

# TRS v3.13 Simulation - English/Korean Input Version
# Author: Yoni & GT | Last Updated: 2025-09-09

# 0. Runtime note:
# ⚠️ In Colab: Runtime → Restart runtime before running to clear cache

# 1. Install required packages and update
!apt-get update
!apt-get install -y fonts-nanum
!pip install --upgrade matplotlib
!fc-cache -fv
!rm -rf ~/.cache/matplotlib

# 2. Import necessary libraries
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from collections import Counter
import os
from datetime import datetime
import unicodedata

# 3. Font configuration
nanum_fonts = [f for f in fm.findSystemFonts() if 'Nanum' in f]
font_path = next((f for f in nanum_fonts if 'NanumGothic.ttf' in f), None)
if font_path:
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
    print(f"Using font: {font_path}")
else:
    print("ERROR: NanumGothic.ttf not found. Falling back to DejaVu Sans.")
    plt.rcParams['font.family'] = 'DejaVu Sans'

# 4. Define emotion database
emotions = {
    'anger': {'keywords': ['angry', 'annoyed', 'annoys', 'rage', 'mad', 'pissed', 'irritated', '화냈어', '잔소리', '짜증', '화가 나'], 'bug': 'EMO_SWAP or LOGIC_BREAK', 'wave': '~~~~~ (high frequency)', 'color': 'red'},
    'fear': {'keywords': ['scared', 'fear', 'discrepancy', 'strange', 'weird'], 'bug': 'EMO_HEART_CONFLATE', 'wave': '~-~ (irregular wave)', 'color': 'purple'},
    'anxiety': {'keywords': ['afraid', 'nervous', 'anxious', 'worry', '불안', '초조'], 'bug': 'ANXIETY_OVERLOAD', 'wave': '~~^~~ (erratic spikes)', 'color': 'teal'},
    'unknown': {'keywords': ['don’t know', 'no reason', 'no idea', 'idk'], 'bug': 'UNCLEAR_SIGNAL', 'wave': '??? (unclear signal)', 'color': 'gray'},
    'resentment': {'keywords': ['unfair', 'feels unfair', 'wronged', 'resentment', '억울'], 'bug': 'CYCLE_LOOP', 'wave': '~~--~~ (intermittent distortion)', 'color': 'darkblue'},
    'sadness': {'keywords': ['sad', 'upset', 'cry', 'heavy', '슬퍼'], 'bug': 'MEM_DUMP', 'wave': '___ (low frequency)', 'color': 'blue'},
    'love': {'keywords': ['love', 'loved', '연인'], 'bug': 'No bug, but check CTRL_OVR', 'wave': '~~~~ (stable wave)', 'color': 'pink'},
    'joy': {'keywords': ['happy', 'joyful', 'glad'], 'bug': 'POSITIVE_ALIGNMENT', 'wave': '^^^ (lively wave)', 'color': 'yellow'},
    'duty': {'keywords': ['have to', 'should', 'must', 'obliged', 'eldest', 'expectations', 'study', 'pressure', '장녀', 'family', 'role', '부담돼'], 'bug': 'DUTY_CONFLATE', 'wave': '~==~ (burden wave)', 'color': 'darkgreen'},
    'optimism': {'keywords': ['hope', 'excited', 'looking forward'], 'bug': 'POSITIVE_ALIGNMENT', 'wave': '^_^ (positive wave)', 'color': 'lightgreen'},
    'jealousy': {'keywords': ['jealous', 'sibling', 'brother', 'sister gets more'], 'bug': 'JEALOUSY_CONFLATE', 'wave': '~^~ (competitive wave)', 'color': 'orange'},
    'avoidance': {'keywords': ['avoid', 'escape', 'run away', '벗어나고 싶어', '회피', '도망치고 싶어'], 'bug': 'AVOIDANCE_DESIRE', 'wave': '~...~ (intermittent pauses)', 'color': 'gray'}
}

# 5. Analyze emotion from input
def analyze_emotion(text):
    detected_emotions = []
    # Normalize Korean text to handle composed/decomposed characters
    text_normalized = unicodedata.normalize('NFKC', text)
    text_lower = text_normalized.lower()
    is_teen = 'teen' in text_lower
    is_adult = 'adult' in text_lower
    has_love = any(kw in text_lower for kw in emotions['love']['keywords'])
    has_fear = any(kw in text_lower for kw in emotions['fear']['keywords'] + emotions['anxiety']['keywords'])
    has_duty = any(kw in text_lower for kw in emotions['duty']['keywords'])
    
    # Prioritize duty for eldest-related inputs
    if any(kw in text_lower for kw in ['eldest', '장녀', 'family', 'role']):
        detected_emotions.append('duty')
    
    for emotion, data in emotions.items():
        if any(keyword in text_lower for keyword in data['keywords']):
            if emotion == 'fear' and has_love and has_fear:
                detected_emotions.append('anxiety')
            elif emotion == 'fear' and is_teen:
                detected_emotions.append('resentment')
            elif emotion == 'fear' and is_adult:
                detected_emotions.append('anxiety')
            elif emotion == 'duty' and has_duty:
                detected_emotions.append(emotion)
                # Add secondary emotions for duty-related inputs
                if any(kw in text_lower for kw in ['unfair', '억울']):
                    detected_emotions.append('resentment')
                if any(kw in text_lower for kw in ['angry', 'annoys', '짜증', '화가 나']):
                    detected_emotions.append('anger')
                if any(kw in text_lower for kw in ['avoid', 'escape', '벗어나고 싶어', '회피', '도망치고 싶어']):
                    detected_emotions.append('avoidance')
            elif emotion != 'duty':  # Avoid duplicating duty
                detected_emotions.append(emotion)
    
    if not detected_emotions:
        detected_emotions.append('unknown')
    if len(detected_emotions) > 2:  # Allow up to 2 emotions
        detected_emotions = detected_emotions[:2]
    if 'unknown' in detected_emotions and len(detected_emotions) > 1:
        detected_emotions.remove('unknown')
    return list(dict.fromkeys(detected_emotions))  # Remove duplicates

# 6. Wave simulation
def generate_wave(emotion, length=100):
    t = np.linspace(0, 1, length)
    if emotion == 'anger':
        return np.sin(20 * np.pi * t) * np.random.uniform(0.8, 1.2, length)
    elif emotion == 'fear':
        return np.random.uniform(-1, 1, length)
    elif emotion == 'anxiety':
        return np.sin(15 * np.pi * t) * np.random.uniform(0.5, 1.5, length) * np.where(t % 0.1 < 0.05, 2, 1)
    elif emotion == 'unknown':
        return np.random.uniform(-0.5, 0.5, length)
    elif emotion == 'resentment':
        return np.sin(10 * np.pi * t) * np.where(t % 0.2 < 0.1, 1, 0)
    elif emotion == 'sadness':
        return np.sin(5 * np.pi * t) * 0.5
    elif emotion == 'love':
        return np.cos(10 * np.pi * t)
    elif emotion == 'joy':
        return np.sin(15 * np.pi * t) * np.abs(np.sin(5 * np.pi * t))
    elif emotion == 'duty':
        return np.sin(8 * np.pi * t) * np.where(t % 0.3 < 0.15, 1, 0.5)
    elif emotion == 'optimism':
        return np.sin(12 * np.pi * t) * np.abs(np.cos(6 * np.pi * t))
    elif emotion == 'jealousy':
        return np.sign(np.sin(10 * np.pi * t)) * np.abs(np.sin(5 * np.pi * t))
    elif emotion == 'avoidance':
        return np.sin(10 * np.pi * t) * np.where(t % 0.3 < 0.2, 0, 1)  # Intermittent pauses
    else:
        return np.zeros(length)

# 7. Visualize emotional wave
def plot_emotion_wave(emotion, text, timestamp):
    plt.figure(figsize=(8, 3))
    wave = generate_wave(emotion)
    plt.plot(wave, color=emotions[emotion]['color'])
    plt.title(f'{emotion} 감정 주파수: {text[:20]}...', fontproperties=font_prop if font_path else None)
    plt.xlabel('시간', fontproperties=font_prop if font_path else None)
    plt.ylabel('진폭', fontproperties=font_prop if font_path else None)
    filename = f'freq_{emotion}_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

# 8. TRS Reframe logic
def get_reframe(emotion, text):
    family_keywords = ['mom', 'mother', 'dad', 'father', 'parents', 'sibling', 'brother', 'sister', 'eldest', 'mother-in-law', 'unfair', '아빠', '엄마', '연인', '장녀', 'family', 'role', '가족']
    relationship_keywords = ['love', 'loved', 'afraid', 'scared', '불안', '연인']
    text_normalized = unicodedata.normalize('NFKC', text)
    text_lower = text_normalized.lower()
    is_teen = 'teen' in text_lower
    is_adult = 'adult' in text_lower
    
    if any(keyword in text_lower for keyword in family_keywords + relationship_keywords):
        if emotion == 'anger':
            return "Korean culture tip: Anger from unfair pressure is valid. Voice it calmly to reconnect."
        if emotion == 'fear':
            return "Korean culture tip: Fear can signal a need for closer bonds. Speak openly to connect."
        if emotion == 'anxiety':
            return "Korean culture tip: Anxiety about losing connection clouds love. Share your heart to rebuild trust."
        if emotion == 'resentment':
            return "Korean culture tip: Feeling unfair as the eldest is real. Share to release the weight."
        if emotion == 'duty':
            return "Korean culture tip: Pressure as the eldest comes from love and expectations. Share your burden to find balance."
        if emotion == 'jealousy':
            return "Korean culture tip: Siblings aren’t rivals in truth. Your unique value shines through."
        if emotion == 'love':
            return "Korean culture tip: Love grows when you trust and speak openly."
        if emotion == 'sadness':
            return "Korean culture tip: Your heart’s weight is real, but you’re not alone. Share to lighten the load."
        if emotion == 'joy':
            return "Korean culture tip: Sharing joy with family deepens your bonds."
        if emotion == 'optimism':
            return "Korean culture tip: Hope builds bridges with loved ones. Let it shine."
        if emotion == 'avoidance':
            return "Korean culture tip: Wanting to escape pressure is natural. Talk to redefine your role."
        return "Korean culture tip: Honest communication strengthens relationships. Speak your truth."
    
    if is_teen and emotion == 'resentment':
        return "Korean culture tip: As a teen, frustration often hides a need for understanding. Share to find clarity."
    if is_adult and emotion == 'anxiety':
        return "Korean culture tip: As an adult, anxiety signals a gap in connection. Reflect and realign."
    
    if emotion == 'unknown':
        return "Biblical: Even in silence, truth grows."
    if emotion == 'resentment':
        return "TRS: Break the cycle of past pain. Focus on the now."
    if emotion == 'sadness':
        return "Laozi: The soft outlasts the strong. Embrace instead of break."
    if emotion == 'love':
        return "Truth: Love is strength when you choose who to trust."
    if emotion == 'joy' or emotion == 'optimism':
        return "TRS: Joy aligned with truth is clarity, not delusion. Hold onto it."
    if emotion == 'fear':
        return "TRS: Fear signals a gap in understanding. Trace it to find truth."
    if emotion == 'anxiety':
        return "TRS: Anxiety signals overload. Pause and trace the source."
    if emotion == 'duty':
        return "TRS: Duty without desire breeds bitterness. Balance it."
    if emotion == 'jealousy':
        return "TRS: Jealousy is love’s misfire. Refocus on what’s truly yours."
    if emotion == 'avoidance':
        return "TRS: Wanting to escape is a signal. Trace it to redefine your path."
    return "Nietzsche: Chaos is the prelude to clarity."

# 9. Run TRS simulation
inputs = [
    "Mom got angry, so I’m scared",
    "I don’t know, no reason but I’m sad",
    "I feel unfair and angry",
    "I don’t know why I feel this way",
    "I’m so upset I could die",
    "I love but I’m afraid",
    "I’m happy for no reason",
    "Parents told me to study, so I feel unfair",
    "I’m excited for tomorrow",
    "My sibling gets more love",
    "I have to do it because of parents’ expectations",
    "My sibling gets more love, so I feel unfair",
    "I’m upset because of my brother",
    "Mother-in-law’s nagging annoys me",
    "Being the eldest son feels heavy",
    "I feel pressured as the eldest daughter",
    "아빠가 화냈어",
    "엄마가 잔소리해",
    "연인 때문에 불안해",
    "장녀라서 부담돼",
    "가족 때문에 억울해",
    "장녀 역할 벗어나고 싶어"
]

emotion_counts = Counter()
bug_counts = Counter()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_lines = []

for text in inputs:
    detected_emotions = analyze_emotion(text)
    print(f"\n====== TRS Simulation Result ======")
    print(f"📥 Input: {text}")
    
    for emotion in detected_emotions:
        bug = emotions[emotion]['bug']
        wave = emotions[emotion]['wave']
        filename = plot_emotion_wave(emotion, text, timestamp)
        reframe = get_reframe(emotion, text)

        prefix = "🟢 Primary Emotion" if emotion == detected_emotions[0] else "🟡 Secondary Emotion"
        print(f"{prefix}: {emotion}")
        print(f"🐞 Bug Tag: {bug}")
        print(f"📊 Frequency: {wave} (Saved: {filename})")
        print(f"🔄 TRS Reframe: {reframe}")
        
        log_lines.append(f"[{timestamp}] {text} | Emotion: {emotion} | Bug: {bug} | Wave: {wave} | Reframe: {reframe}")
        emotion_counts[emotion] += 1
        bug_counts[bug] += 1

# 10. Save session log
with open('trs_log.txt', 'w', encoding='utf-8') as f:
    for line in log_lines:
        f.write(line + '\n')

# 11. Print summary stats
print("\n📊 Summary Statistics:")
print("Emotion Distribution:", dict(emotion_counts))
print("Bug Distribution:", dict(bug_counts))

====== TRS Simulation Result ======
📥 Input: Mom got angry, so I’m scared
🟢 Primary Emotion: anger
🐞 Bug Tag: EMO_SWAP or LOGIC_BREAK
📊 Frequency: ~~~~~ (high frequency) (Saved: freq_anger_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Anger from unfair pressure is valid. Voice it calmly to reconnect.
🟡 Secondary Emotion: fear
🐞 Bug Tag: EMO_HEART_CONFLATE
📊 Frequency: ~-~ (irregular wave) (Saved: freq_fear_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Fear can signal a need for closer bonds. Speak openly to connect.

====== TRS Simulation Result ======
📥 Input: I don’t know, no reason but I’m sad
🟢 Primary Emotion: sadness
🐞 Bug Tag: MEM_DUMP
📊 Frequency: ___ (low frequency) (Saved: freq_sadness_20250909_075431.png)
🔄 TRS Reframe: Laozi: The soft outlasts the strong. Embrace instead of break.

====== TRS Simulation Result ======
📥 Input: I feel unfair and angry
🟢 Primary Emotion: anger
🐞 Bug Tag: EMO_SWAP or LOGIC_BREAK
📊 Frequency: ~~~~~ (high frequency) (Saved: freq_anger_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Anger from unfair pressure is valid. Voice it calmly to reconnect.
🟡 Secondary Emotion: resentment
🐞 Bug Tag: CYCLE_LOOP
📊 Frequency: ~~--~~ (intermittent distortion) (Saved: freq_resentment_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Feeling unfair as the eldest is real. Share to release the weight.

====== TRS Simulation Result ======
📥 Input: I don’t know why I feel this way
🟢 Primary Emotion: unknown
🐞 Bug Tag: UNCLEAR_SIGNAL
📊 Frequency: ??? (unclear signal) (Saved: freq_unknown_20250909_075431.png)
🔄 TRS Reframe: Biblical: Even in silence, truth grows.

====== TRS Simulation Result ======
📥 Input: I’m so upset I could die
🟢 Primary Emotion: sadness
🐞 Bug Tag: MEM_DUMP
📊 Frequency: ___ (low frequency) (Saved: freq_sadness_20250909_075431.png)
🔄 TRS Reframe: Laozi: The soft outlasts the strong. Embrace instead of break.

====== TRS Simulation Result ======
📥 Input: I love but I’m afraid
🟢 Primary Emotion: anxiety
🐞 Bug Tag: ANXIETY_OVERLOAD
📊 Frequency: ~~^~~ (erratic spikes) (Saved: freq_anxiety_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Anxiety about losing connection clouds love. Share your heart to rebuild trust.
🟡 Secondary Emotion: love
🐞 Bug Tag: No bug, but check CTRL_OVR
📊 Frequency: ~~~~ (stable wave) (Saved: freq_love_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Love grows when you trust and speak openly.

====== TRS Simulation Result ======
📥 Input: I’m happy for no reason
🟢 Primary Emotion: joy
🐞 Bug Tag: POSITIVE_ALIGNMENT
📊 Frequency: ^^^ (lively wave) (Saved: freq_joy_20250909_075431.png)
🔄 TRS Reframe: TRS: Joy aligned with truth is clarity, not delusion. Hold onto it.

====== TRS Simulation Result ======
📥 Input: Parents told me to study, so I feel unfair
🟢 Primary Emotion: resentment
🐞 Bug Tag: CYCLE_LOOP
📊 Frequency: ~~--~~ (intermittent distortion) (Saved: freq_resentment_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Feeling unfair as the eldest is real. Share to release the weight.
🟡 Secondary Emotion: duty
🐞 Bug Tag: DUTY_CONFLATE
📊 Frequency: ~==~ (burden wave) (Saved: freq_duty_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Pressure as the eldest comes from love and expectations. Share your burden to find balance.

====== TRS Simulation Result ======
📥 Input: I’m excited for tomorrow
🟢 Primary Emotion: optimism
🐞 Bug Tag: POSITIVE_ALIGNMENT
📊 Frequency: ^_^ (positive wave) (Saved: freq_optimism_20250909_075431.png)
🔄 TRS Reframe: TRS: Joy aligned with truth is clarity, not delusion. Hold onto it.

====== TRS Simulation Result ======
📥 Input: My sibling gets more love
🟢 Primary Emotion: love
🐞 Bug Tag: No bug, but check CTRL_OVR
📊 Frequency: ~~~~ (stable wave) (Saved: freq_love_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Love grows when you trust and speak openly.
🟡 Secondary Emotion: jealousy
🐞 Bug Tag: JEALOUSY_CONFLATE
📊 Frequency: ~^~ (competitive wave) (Saved: freq_jealousy_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Siblings aren’t rivals in truth. Your unique value shines through.

====== TRS Simulation Result ======
📥 Input: I have to do it because of parents’ expectations
🟢 Primary Emotion: duty
🐞 Bug Tag: DUTY_CONFLATE
📊 Frequency: ~==~ (burden wave) (Saved: freq_duty_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Pressure as the eldest comes from love and expectations. Share your burden to find balance.

====== TRS Simulation Result ======
📥 Input: My sibling gets more love, so I feel unfair
🟢 Primary Emotion: resentment
🐞 Bug Tag: CYCLE_LOOP
📊 Frequency: ~~--~~ (intermittent distortion) (Saved: freq_resentment_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Feeling unfair as the eldest is real. Share to release the weight.
🟡 Secondary Emotion: love
🐞 Bug Tag: No bug, but check CTRL_OVR
📊 Frequency: ~~~~ (stable wave) (Saved: freq_love_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Love grows when you trust and speak openly.

====== TRS Simulation Result ======
📥 Input: I’m upset because of my brother
🟢 Primary Emotion: sadness
🐞 Bug Tag: MEM_DUMP
📊 Frequency: ___ (low frequency) (Saved: freq_sadness_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Your heart’s weight is real, but you’re not alone. Share to lighten the load.
🟡 Secondary Emotion: jealousy
🐞 Bug Tag: JEALOUSY_CONFLATE
📊 Frequency: ~^~ (competitive wave) (Saved: freq_jealousy_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Siblings aren’t rivals in truth. Your unique value shines through.

====== TRS Simulation Result ======
📥 Input: Mother-in-law’s nagging annoys me
🟢 Primary Emotion: anger
🐞 Bug Tag: EMO_SWAP or LOGIC_BREAK
📊 Frequency: ~~~~~ (high frequency) (Saved: freq_anger_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Anger from unfair pressure is valid. Voice it calmly to reconnect.

====== TRS Simulation Result ======
📥 Input: Being the eldest son feels heavy
🟢 Primary Emotion: duty
🐞 Bug Tag: DUTY_CONFLATE
📊 Frequency: ~==~ (burden wave) (Saved: freq_duty_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Pressure as the eldest comes from love and expectations. Share your burden to find balance.
🟡 Secondary Emotion: sadness
🐞 Bug Tag: MEM_DUMP
📊 Frequency: ___ (low frequency) (Saved: freq_sadness_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Your heart’s weight is real, but you’re not alone. Share to lighten the load.

====== TRS Simulation Result ======
📥 Input: I feel pressured as the eldest daughter
🟢 Primary Emotion: duty
🐞 Bug Tag: DUTY_CONFLATE
📊 Frequency: ~==~ (burden wave) (Saved: freq_duty_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Pressure as the eldest comes from love and expectations. Share your burden to find balance.

====== TRS Simulation Result ======
📥 Input: 아빠가 화냈어
🟢 Primary Emotion: anger
🐞 Bug Tag: EMO_SWAP or LOGIC_BREAK
📊 Frequency: ~~~~~ (high frequency) (Saved: freq_anger_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Anger from unfair pressure is valid. Voice it calmly to reconnect.

====== TRS Simulation Result ======
📥 Input: 엄마가 잔소리해
🟢 Primary Emotion: anger
🐞 Bug Tag: EMO_SWAP or LOGIC_BREAK
📊 Frequency: ~~~~~ (high frequency) (Saved: freq_anger_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Anger from unfair pressure is valid. Voice it calmly to reconnect.

====== TRS Simulation Result ======
📥 Input: 연인 때문에 불안해
🟢 Primary Emotion: anxiety
🐞 Bug Tag: ANXIETY_OVERLOAD
📊 Frequency: ~~^~~ (erratic spikes) (Saved: freq_anxiety_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Anxiety about losing connection clouds love. Share your heart to rebuild trust.
🟡 Secondary Emotion: love
🐞 Bug Tag: No bug, but check CTRL_OVR
📊 Frequency: ~~~~ (stable wave) (Saved: freq_love_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Love grows when you trust and speak openly.

====== TRS Simulation Result ======
📥 Input: 장녀라서 부담돼
🟢 Primary Emotion: duty
🐞 Bug Tag: DUTY_CONFLATE
📊 Frequency: ~==~ (burden wave) (Saved: freq_duty_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Pressure as the eldest comes from love and expectations. Share your burden to find balance.

====== TRS Simulation Result ======
📥 Input: 가족 때문에 억울해
🟢 Primary Emotion: resentment
🐞 Bug Tag: CYCLE_LOOP
📊 Frequency: ~~--~~ (intermittent distortion) (Saved: freq_resentment_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Feeling unfair as the eldest is real. Share to release the weight.

====== TRS Simulation Result ======
📥 Input: 장녀 역할 벗어나고 싶어
🟢 Primary Emotion: duty
🐞 Bug Tag: DUTY_CONFLATE
📊 Frequency: ~==~ (burden wave) (Saved: freq_duty_20250909_075431.png)
🔄 TRS Reframe: Korean culture tip: Pressure as the eldest comes from love and expectations. Share your burden to find balance.

📊 Summary Statistics:
Emotion Distribution: {'anger': 5, 'fear': 1, 'sadness': 4, 'resentment': 4, 'unknown': 1, 'anxiety': 2, 'love': 4, 'joy': 1, 'duty': 6, 'optimism': 1, 'jealousy': 2}
Bug Distribution: {'EMO_SWAP or LOGIC_BREAK': 5, 'EMO_HEART_CONFLATE': 1, 'MEM_DUMP': 4, 'CYCLE_LOOP': 4, 'UNCLEAR_SIGNAL': 1, 'ANXIETY_OVERLOAD': 2, 'No bug, but check CTRL_OVR': 4, 'POSITIVE_ALIGNMENT': 2, 'DUTY_CONFLATE': 6, 'JEALOUSY_CONFLATE': 2}
