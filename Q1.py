# dataset.py
# E-Commerce Sentiment Dataset for AI Agent Training

# ---------------------------------------------------------
# TRAINING DATA: Use this to build vocabulary and probabilities
# ---------------------------------------------------------
TRAINING_DATA = [
    # --- Happy Class ---
    {"text": "Absolutely love this product, it works perfectly!", "label": "Happy"},
    {"text": "Fast delivery, very satisfied with my purchase.", "label": "Happy"},
    {"text": "Great value for the price, highly recommend.", "label": "Happy"},
    {"text": "The quality is outstanding and it fits perfectly.", "label": "Happy"},
    {"text": "Excellent customer support, they resolved my issue immediately.", "label": "Happy"},
    {"text": "Five stars, exactly what I was looking for.", "label": "Happy"},
    {"text": "Beautiful design and very sturdy materials.", "label": "Happy"},
    {"text": "Arrived earlier than expected, very happy.", "label": "Happy"},
    {"text": "Best purchase I have made this year.", "label": "Happy"},
    {"text": "So glad I bought this, it has been a game changer.", "label": "Happy"},
    
    # --- Frustrated Class ---
    {"text": "Terrible experience, the item broke on the first day.", "label": "Frustrated"},
    {"text": "Shipping took way too long and the box was damaged.", "label": "Frustrated"},
    {"text": "Customer service was incredibly rude and unhelpful.", "label": "Frustrated"},
    {"text": "Completely useless, does not match the description at all.", "label": "Frustrated"},
    {"text": "Waste of money, poor quality materials.", "label": "Frustrated"},
    {"text": "I am still waiting for my refund after three weeks.", "label": "Frustrated"},
    {"text": "Do not buy this, it is a complete scam.", "label": "Frustrated"},
    {"text": "The sizing is completely wrong and they refuse returns.", "label": "Frustrated"},
    {"text": "Extremely disappointed with the lack of communication.", "label": "Frustrated"},
    {"text": "Defective right out of the box, frustrating.", "label": "Frustrated"}
]

# ---------------------------------------------------------
# INFERENCE / TEST DATA: Use this to test your final classifier
# NOTE: These strings contain words that DO NOT appear in the training data.
# ---------------------------------------------------------
TEST_DATA = [
    "The delivery was incredibly quick and the item is brilliant!", # Expected: Happy
    "Horrible scam, totally destroyed my trust in this site.",      # Expected: Frustrated
    "Customer support was fast but the product is useless.",        # Expected: Frustrated (Mixed signals)
    "Outstanding experience, beautiful and sturdy packaging."       # Expected: Happy
]

class PredictSentiment:
    def __init__(self):
        pass

    def calculate_class_proportions(self):
        happy_count = sum(1 for item in self.training_data if item["label"] == "Happy")
        frustrated_count = sum(1 for item in self.training_data if item["label"] == "Frustrated")
        total_count = happy_count + frustrated_count
        self.happy_proportion = happy_count / total_count
        self.frustrated_proportion = frustrated_count / total_count

    def vocab_count(self):
        self.happy_count = {}
        self.frustrated_count = {}
        for sequence in self.training_data:
            label = sequence["label"]
            text = sequence["text"].lower()
            words = text.replace(".", "").replace(",", "").split()
            if label == "Happy":
                for word in words:
                    self.happy_count[word] = self.happy_count.get(word, 0) + 1
            elif label == "Frustrated":
                for word in words:
                    self.frustrated_count[word] = self.frustrated_count.get(word, 0) + 1
        self.happy_total_count = sum(self.happy_count.values())
        self.frustrated_total_count = sum(self.frustrated_count.values())
    
    def train(self, training_data):
        self.training_data = training_data
        self.calculate_class_proportions()
        self.vocab_count()

    def predict(self, text):
        words = text.lower().replace(".", "").replace(",", "").split()
        happy_prob = 1
        frustrated_prob = 1
        for word in words:
            if word in self.happy_count: # Only consider words seen in training
                happy_prob *= (self.happy_count[word] + 1) / (self.happy_total_count + len(self.happy_count) + len(self.frustrated_count)) # Word is seen
            else:
                happy_prob *= 1 / (self.happy_total_count + len(self.happy_count) + len(self.frustrated_count)) # Unseen word, apply smoothing
            if word in self.frustrated_count: # Only consider words seen in training
                frustrated_prob *= (self.frustrated_count[word] + 1) / (self.frustrated_total_count + len(self.happy_count) + len(self.frustrated_count)) # Word is seen
            else:
                frustrated_prob *= 1 / (self.frustrated_total_count + len(self.happy_count) + len(self.frustrated_count)) # Unseen word, apply smoothing
        happy_prob *= self.happy_proportion
        frustrated_prob *= self.frustrated_proportion
        return ("Happy", happy_prob) if happy_prob > frustrated_prob else ("Frustrated", frustrated_prob)

model = PredictSentiment()
model.train(TRAINING_DATA)
for test in TEST_DATA:
    prediction, probability = model.predict(test)
    print(f"Text: '{test}' => Predicted Sentiment: {prediction}, Probability: {probability}")

