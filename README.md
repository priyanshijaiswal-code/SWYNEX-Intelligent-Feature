# Intelligent Feature — Error Handling, Failure Cases & Demo Interface

**Internship:** SWYNEX Technologies — Artificial Intelligence Internship
**Task:** Task 3 — Intelligent Feature

## What This Adds (on top of Task 2)
This builds on the scam-message classifier from Task 2 (`scam_classifier.py`) by adding:

1. **Input error handling** — `safe_predict()` validates input before running it through the model, and raises a clear `InvalidInputError` for bad input instead of letting the underlying code crash.
2. **Documented, verified failure cases** — real messages the model gets wrong, tested against the actual trained model (not hypothetical).
3. **A simple command-line interface** — type a message, get a live classification, explanation, and risk level.

## Error Handling
| Input | Behavior |
|---|---|
| `None` | Raises `InvalidInputError: No message provided` |
| `""` / whitespace only | Raises `InvalidInputError: Message is empty` |
| Non-string (e.g. `12345`) | Raises `InvalidInputError: Expected a string message, got int` |
| Too short (e.g. `"hi"`) | Raises `InvalidInputError: Message is too short to classify reliably` |
| Message with no recognized words | Returns `UNKNOWN` with an explicit note, rather than a falsely confident guess |

## Documented Failure Cases
These are real, verified misclassifications — included for transparency about the prototype's current limitations, not cherry-picked successes.

1. **"Congratulations on completing your certification course!"**
   Expected: Legitimate → Model says: **Scam** (false positive)
   The word "congratulations" is over-weighted toward scam because most scam examples in the small training set open that way.

2. **"You have an important update regarding your recent order."**
   Expected: Legitimate → Model says: **Scam** (false positive)
   Words like "important" and "update" overlap with urgency language the model associates with scams.

3. **"Act now, this offer expires soon, don't miss out on huge savings this weekend."**
   Expected: Scam → Model says: **Legitimate** (false negative)
   A promotional scam without the specific keywords ("click", "verify", "account") the model was trained to recognize — arguably the more concerning failure mode, since it lets a real scam through.

**Takeaway:** With only ~20 training examples, the model over-relies on a handful of keywords rather than deeper context. A larger, more diverse dataset would be the main next step to reduce both false positives and false negatives.

## Demo Interface
Running the script drops into an interactive CLI:
```
Enter a message: Your account will be suspended, click here to verify now
  -> Prediction: SCAM (risk: Medium, confidence: 0.58)
  -> Flagged words: ['click', 'verify', 'suspended', 'account']
```
Type `quit` to exit.

## Files
- `scam_classifier.py` — Task 2's base model (training + evaluation)
- `intelligent_feature.py` — Task 3's error handling, failure case report, and CLI demo

## How to Run
```bash
pip install scikit-learn
python intelligent_feature.py
```
