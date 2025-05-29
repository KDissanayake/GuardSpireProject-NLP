# GuardSpireProject-NLP

This repository contains the **Natural Language Processing (NLP) module** for the GuardSpire project.  
It is responsible for analyzing user text input and detecting potential scam-related content using machine learning models.

---

## 🧠 Overview

This module includes:
- A pretrained **BERT model** for contextual language understanding.
- A **Logistic Regression** classifier trained on synthetic scam data.
- Supporting tools and tokenizers to preprocess and evaluate user input.

---

## 🚀 How to Run the NLP Module

### ✅ Prerequisites

- Python 3.x
- `pip` installed
- Flask
- (Optional) Virtual environment

---

### 🛠️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/KDissanayake/GuardSpireProject-NLP.git
cd GuardSpireProject-NLP
````

2. **(Optional) Create and activate a virtual environment:**

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. **Install dependencies (including Flask):**

```bash
pip install -r requirements.txt
```

4. **Download necessary model files (if not already present):**

```bash
python download_models.py
```

5. **Run the NLP Flask server:**

```bash
python app.py
```

📍 The NLP server will run on:
[http://localhost:5001](http://localhost:5001)


---

## 📝 Notes

* Ensure `saved_model/` folder contains all model artifacts (config, tokenizer, model, etc).
* Large files like `.safetensors` are excluded via `.gitignore` and should be manually added or downloaded.

---


