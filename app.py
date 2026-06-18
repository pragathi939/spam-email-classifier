import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Convert text to numbers
cv = CountVectorizer()
X = cv.fit_transform(df["message"])

# Convert labels
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Web page
st.title("Spam Email Classifier")

msg = st.text_area("Enter your message")

if st.button("Check"):
    data = cv.transform([msg])
    prediction = model.predict(data)

    if prediction[0] == "spam":
        st.error("SPAM")
    else:
        st.success("NOT SPAM")