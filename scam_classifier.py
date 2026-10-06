"""
Scam Message Classifier — Task 2 Prototype
SWYNEX Technologies — Artificial Intelligence Internship

Integrates a TF-IDF + Logistic Regression model (scikit-learn) into a small
working prototype for the problem defined in Task 1: classifying SMS/text
messages as scam or legitimate, with a basic explainability layer showing
which words most influenced the prediction.

No external API keys are used — this runs entirely on a local scikit-learn
model, so there is nothing secret to exclude from the repo.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# ---------------------------------------------------------------------------
# 1. Sample labeled dataset (small, for prototype purposes)
#    label: 1 = scam, 0 = legitimate
# ---------------------------------------------------------------------------
messages = [
    # Scam examples
    ("Congratulations! You've won a $1000 Walmart gift card. Click here to claim now: bit.ly/claim123", 1),
    ("URGENT: Your bank account has been suspended. Verify your details immediately at secure-bank-verify.com", 1),
    ("You have been selected for a free iPhone 15. Confirm your address to receive it today!", 1),
    ("Your account will be blocked in 24 hours. Update your KYC now: kyc-update-alert.in", 1),
    ("Dear customer, your parcel is on hold due to unpaid customs fee. Pay $2.99 here: track-parcel-pay.com", 1),
    ("You've won a lottery of $500,000! Send your bank details to claim your prize immediately.", 1),
    ("Limited time offer: Get a personal loan approved instantly, no documents required. Apply now!", 1),
    ("Your OTP is required to reverse a fraud transaction. Share it now with our support agent.", 1),
    ("Congratulations, you are the lucky winner! Click the link and enter your card details to redeem.", 1),
    ("Your electricity connection will be disconnected tonight. Pay pending bill immediately via this link.", 1),

    # Legitimate examples
    ("Hey, are we still on for lunch tomorrow at 1pm?", 0),
    ("Your OTP for login is 483920. Do not share this with anyone.", 0),
    ("Reminder: Your appointment with Dr. Sharma is scheduled for 10 AM on Friday.", 0),
    ("Your Amazon order #45213 has been shipped and will arrive by Thursday.", 0),
    ("Hi mom, I'll be home by 8, don't wait for dinner.", 0),
    ("Your electricity bill of Rs. 1200 is due on the 15th. Pay via the official app.", 0),
    ("Meeting moved to 3 PM in Conference Room B. Please confirm attendance.", 0),
    ("Thank you for your payment of Rs. 999. Your subscription is now active.", 0),
    ("Your flight PNR 8X92JZ is confirmed for 21 Oct, departure 6:45 AM.", 0),
    ("Don't forget to submit the assignment by midnight, professor mentioned it in class.", 0),
]

texts = [m[0] for m in messages]
labels = [m[1] for m in messages]

# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# ---------------------------------------------------------------------------
# 3. TF-IDF + Logistic Regression pipeline
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_vec)
print("=" * 60)
print("EVALUATION ON HELD-OUT TEST SET")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Scam"]))

# ---------------------------------------------------------------------------
# 5. Explainability helper — top contributing words for a prediction
# ---------------------------------------------------------------------------
feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = model.coef_[0]

def explain_prediction(text, top_n=5):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]

    # words present in this message, weighted by their trained coefficient
    present_idx = vec.nonzero()[1]
    word_scores = [(feature_names[i], coefficients[i]) for i in present_idx]
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    top_words = word_scores[:top_n]

    label = "SCAM" if pred == 1 else "LEGITIMATE"
    risk = "High" if prob > 0.75 else "Medium" if prob > 0.5 else "Low"

    return {
        "text": text,
        "prediction": label,
        "confidence": round(float(prob), 2),
        "risk_level": risk,
        "flagged_words": [w for w, s in top_words],
    }

# ---------------------------------------------------------------------------
# 6. Example inputs and outputs (as required by the task)
# ---------------------------------------------------------------------------
example_messages = [
    "Congratulations! You've been selected to receive a free vacation package. Click now to claim.",
    "Hey, can you send me the notes from today's lecture?",
    "URGENT: Your card has been blocked. Verify your PIN immediately at this link.",
    "Your Zomato order has been delivered. Enjoy your meal!",
]

print("\n" + "=" * 60)
print("EXAMPLE INPUTS AND OUTPUTS")
print("=" * 60)
for msg in example_messages:
    result = explain_prediction(msg)
    print(f"\nInput: {result['text']}")
    print(f"  -> Prediction: {result['prediction']} (risk: {result['risk_level']}, confidence: {result['confidence']})")
    print(f"  -> Flagged words: {result['flagged_words']}")
