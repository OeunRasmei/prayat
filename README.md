# Prayat (ប្រយ័ត្ន): Scam Message Detector for English, Khmer and Chinese

Machine Learning group project. *Prayat* means "be careful" in Khmer.

Paste a message or upload a screenshot. Prayat tells you whether it looks like a **scam**,
**spam / an advert**, or a **normal** message, underlines the words that made it suspicious, and
says what to do. Everything runs in the browser: messages and screenshots never leave the device.

**Live app:** https://oeunrasmei.github.io/prayat/

## Results (3,209 test messages never seen in training)

| Model | Accuracy | Macro F1 | Scams caught | Any warning on spam/scam | False alarms |
|---|---:|---:|---:|---:|---:|
| Naive Bayes (characters) | 96.6% | 95.1% | 87.9% | 97.5% | 1.8% |
| Logistic Regression (words) | 84.4% | 78.4% | 64.6% | 62.2% | 1.0% |
| **Logistic Regression (characters)**, deployed | **98.2%** | **97.2%** | **93.0%** | **98.3%** | **0.3%** |
| Neural network / MLP (characters) | 97.6% | 96.3% | 89.1% | 96.7% | 0.1% |

Macro F1 per language for the deployed model: English 91.0%, Khmer 90.1%, Chinese 98.7%.
Word features only reach 77.8% on Khmer and 75.2% on Chinese, because those languages do not put
spaces between words. Character n-grams work for all three scripts.

## Data

| Language | Source | Used for |
|---|---|---|
| English | SMS Phishing Dataset for Machine Learning and Pattern Recognition, Mishra & Soni, Mendeley Data 2022 (5,971 SMS: ham / spam / smishing) | normal, spam, scam |
| Chinese | FBS SMS Dataset, Zhang et al., "Lies in the Air", ACM CCS 2020 (real spam and fraud SMS from fake base stations) | spam, scam |
| Chinese | NUS SMS Corpus, Chen & Kan 2013 (real personal SMS) | normal |
| Khmer, English, Chinese | Written by our team: `data/authored/generate_authored.py` (template-based, common Cambodian scam patterns plus modern English and Chinese scams) | all classes |

No public Khmer scam dataset exists, so **all Khmer data is authored**. The test set is split by
template, so Khmer scores measure performance on message patterns the model has never seen, not on
real Khmer scams. Native speakers in the group should review and extend the Khmer messages.

## How it works

1. **OCR** (screenshots only): Tesseract.js reads English, Khmer and Chinese text in the browser.
2. **Normalisation** (`src/textprep.py`, ported to `docs/js/textprep.js`): NFKC, lower-case,
   traditional → simplified Chinese, Khmer digits → 0-9, links / phone numbers / e-mails → placeholders,
   numbers → 0, punctuation and emoji removed.
3. **Features**: TF-IDF over 40,000 character n-grams (1 to 4 characters).
4. **Model**: multinomial Logistic Regression (C = 10, each language × class group weighted equally).
5. **Explanation**: each n-gram's contribution is spread back over its characters to underline suspicious phrases.

`tests/check_parity.mjs` checks that the JavaScript gives the same results as Python
(identical normalisation, probabilities within 0.000001).

## Project structure

```
notebooks/scam_detector.ipynb    EDA, preprocessing, 4 models, evaluation, export (executed)
notebooks/scam_detector.py       same notebook as a plain script (jupytext)
src/textprep.py                  shared text normalisation
data/download_data.py            downloads the three public datasets into data/raw/
data/authored/                   our authored messages and their generator
reports/figures/                 charts used in the slides
docs/                            the web app (GitHub Pages)
tests/check_parity.mjs           Python vs JavaScript check
```

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python data/download_data.py
.venv/bin/python data/authored/generate_authored.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/scam_detector.ipynb
node tests/check_parity.mjs
python3 -m http.server 8000 --directory docs
```

## Team

- Oeun Rasmei
- _add group members_
