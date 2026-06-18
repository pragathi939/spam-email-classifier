import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load UCI dataset
data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

X = data["message"]
y = data["label"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy * 100, 2), "%")

while True:
    msg = input("Enter message (or exit): ")

    if msg.lower() == "exit":
        break

    msg_vector = vectorizer.transform([msg])

    result = model.predict(msg_vector)

    if result[0] == 1:
        print("SPAM\n")
    else:
        print("NOT SPAM\n")