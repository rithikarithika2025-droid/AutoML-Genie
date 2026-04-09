import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import matplotlib.pyplot as plt

st.set_page_config(page_title="Pro AutoML App", layout="wide")

st.title("🤖 PRO AutoML App")
st.write("Upload CSV → Train Model → Predict → Analyze")

# 📂 Upload CSV
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    # 🧹 Handle missing values
    df = df.dropna()

    # 🎯 Select target
    target = st.selectbox("Select Target Column", df.columns)

    # 🤖 Select model
    model_name = st.selectbox(
        "Choose Model",
        ["Random Forest", "Logistic Regression", "SVM"]
    )

    if st.button("🚀 Train Model"):

        X = df.drop(target, axis=1)
        y = df[target]

        # 🔄 Encode categorical data
        encoders = {}

        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le

        if y.dtype == "object":
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 🧠 model selection
        if model_name == "Random Forest":
            model = RandomForestClassifier()
        elif model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        else:
            model = SVC()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        st.success(f"🎯 Accuracy: {acc:.2f}")

        # 📊 Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # 🌳 Feature Importance (only for Random Forest)
        if model_name == "Random Forest":
            st.subheader("📌 Feature Importance")
            importance = model.feature_importances_

            fig2, ax2 = plt.subplots()
            ax2.barh(X.columns, importance)
            st.pyplot(fig2)

        # 🔮 Prediction Section
        st.subheader("🔮 Make Prediction")

        input_data = {}

        for col in X.columns:
            input_data[col] = st.number_input(f"{col}", value=0)

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)

            st.success(f"Prediction: {prediction[0]}")