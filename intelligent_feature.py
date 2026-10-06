"""
Scam Message Classifier — Task 3: Intelligent Feature
SWYNEX Technologies — Artificial Intelligence Internship

Builds on the Task 2 prototype (scam_classifier.py) by adding:
  1. Input error handling (empty input, non-string input, overly short input)
  2. A documented set of failure cases where the model gets it wrong
  3. A simple command-line interface for interactive testing / demo


"""

from scam_classifier import vectorizer, model, feature_names, coefficients
import numpy as np


# ---------------------------------------------------------------------------
# 1. Error handling wrapper around the core prediction logic
# ---------------------------------------------------------------------------
class InvalidInputError(Exception):
    """Raised when the input message cannot be classified."""
    pass


def safe_predict(text, top_n=5):
    """
    Wraps the model's prediction with input validation and error handling.
    Raises InvalidInputError with a clear message on bad input, rather than
    letting the underlying vectorizer/model throw an unhandled exception.
    """
    if text is None:
        raise InvalidInputError("No message provided (input was None).")

    if not isinstance(text, str):
        raise InvalidInputError(f"Expected a string message, got {type(text).__name__}.")

    cleaned = text.strip()

    if cleaned == "":
        raise InvalidInputError("Message is empty. Please provide some text to classify.")

    if len(cleaned) < 3:
        raise InvalidInputError("Message is too short to classify reliably (minimum ~3 characters).")

    vec = vectorizer.transform([cleaned])

    # If none of the words in the message were seen during training, the
    # vector will be all zeros — the model can still technically predict,
    # but the result is effectively a guess. We flag this explicitly rather
    # than silently returning a falsely confident answer.
    if vec.nnz == 0:
        return {
            "text": cleaned,
            "prediction": "UNKNOWN",
            "confidence": None,
            "risk_level": "Undetermined",
            "flagged_words": [],
            "note": "None of the words in this message were seen in training data. "
                    "Prediction would be unreliable, so no verdict is given.",
        }

    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]

    present_idx = vec.nonzero()[1]
    word_scores = [(feature_names[i], coefficients[i]) for i in present_idx]
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    top_words = [w for w, s in word_scores[:top_n]]

    label = "SCAM" if pred == 1 else "LEGITIMATE"
    risk = "High" if prob > 0.75 else "Medium" if prob > 0.5 else "Low"

    return {
        "text": cleaned,
        "prediction": label,
        "confidence": round(float(prob), 2),
        "risk_level": risk,
        "flagged_words": top_words,
        "note": None,
    }


# ---------------------------------------------------------------------------
# 2. Documented failure cases
#    These are messages the model gets WRONG, included deliberately to be
#    transparent about the prototype's current limitations.
# ---------------------------------------------------------------------------
FAILURE_CASES = [
    {
        "text": "Congratulations on completing your certification course!",
        "expected": "LEGITIMATE",
        "why_it_fails": "The word 'congratulations' is heavily weighted toward SCAM in the trained model "
                        "since most scam examples in the training set open that way, causing a false "
                        "positive on genuine celebratory messages.",
    },
    {
        "text": "You have an important update regarding your recent order.",
        "expected": "LEGITIMATE",
        "why_it_fails": "Words like 'important' and 'update' overlap with urgency language seen in the "
                        "scam training examples, so a neutral order-status message is misread as a threat.",
    },
    {
        "text": "Act now, this offer expires soon, don't miss out on huge savings this weekend.",
        "expected": "SCAM",
        "why_it_fails": "This is a promotional/marketing message with urgency cues but none of the specific "
                        "words ('click', 'verify', 'account', 'bank') the model learned to associate with "
                        "scams, so it's misclassified as legitimate. A false negative — arguably the more "
                        "dangerous failure mode for a scam detector.",
    },
]


def run_failure_case_report():
    print("=" * 60)
    print("DOCUMENTED FAILURE CASES")
    print("=" * 60)
    for case in FAILURE_CASES:
        result = safe_predict(case["text"])
        actual = result["prediction"]
        status = "MATCHES expected" if actual == case["expected"] else "MISCLASSIFIED"
        print(f"\nInput: {case['text']}")
        print(f"  Expected: {case['expected']} | Model said: {actual}  ->  {status}")
        print(f"  Why this is hard for the model: {case['why_it_fails']}")


# ---------------------------------------------------------------------------
# 3. Error-handling demonstration
# ---------------------------------------------------------------------------
def run_error_handling_demo():
    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMONSTRATION")
    print("=" * 60)
    bad_inputs = [None, "", "   ", 12345, "hi"]
    for bad in bad_inputs:
        try:
            result = safe_predict(bad)
            print(f"\nInput: {bad!r} -> Result: {result}")
        except InvalidInputError as e:
            print(f"\nInput: {bad!r} -> Handled gracefully: {e}")


# ---------------------------------------------------------------------------
# 4. Simple command-line interface for interactive demo
# ---------------------------------------------------------------------------
def run_cli():
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO — type a message to classify, or 'quit' to exit")
    print("=" * 60)
    while True:
        try:
            user_input = input("\nEnter a message: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.strip().lower() == "quit":
            print("Exiting.")
            break

        try:
            result = safe_predict(user_input)
            if result["prediction"] == "UNKNOWN":
                print(f"  -> {result['note']}")
            else:
                print(f"  -> Prediction: {result['prediction']} "
                      f"(risk: {result['risk_level']}, confidence: {result['confidence']})")
                print(f"  -> Flagged words: {result['flagged_words']}")
        except InvalidInputError as e:
            print(f"  -> Error: {e}")


if __name__ == "__main__":
    run_error_handling_demo()
    run_failure_case_report()
    run_cli()
