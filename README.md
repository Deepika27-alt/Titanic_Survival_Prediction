 Titanic Survival Prediction

📌 Project Overview

This project predicts whether a passenger survived the Titanic disaster based on various features such as sex, age, gender, passenger class, and fare. It uses machine learning techniques to classify passengers as **Survived (1) or Not Survived (0).**
Supervised Learning - Binary classification problem ( uses Random Forest )

📊 Dataset

The dataset used is the **Titanic dataset** from Kaggle, which contains information about passengers such as:

    - `Pclass` (Passenger class: 1st, 2nd, or 3rd)
    - `Sex` (Male or Female)
    - `Age` (Age in years)
    - `SibSp` (Number of siblings/spouses aboard)
    - `Parch` (Number of parents/children aboard)
    - `Fare` (Ticket price)
    - `Embarked` (Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton)

⚙️ Machine Learning Approach
    1️⃣ **Data Preprocessing**
        - Handled missing values in `Age`, `Embarked`.
        - Converted categorical variables (`Sex`, `Embarked`) using **Label Encoding**.
        - Scaled numerical values using `StandardScaler`.

    2️⃣ **Model Training**
       Trained a Random Forest Classifier, which performed best on the dataset:
        - **Random Forest Classifier** 

    3️⃣ **Evaluation Metrics**
        - **Accuracy Score**
        - **Confusion Matrix**
ALso created a prediction UI using ipywidgets, which enhances usability.

🛠 Installation & Usage
    **1️⃣ Install Dependencies**
    ```bash
        pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```

    **2️⃣ Run the Model**
    ```python
        python TitanicSurvivalPrediction.py
    ```

    **3️⃣ Make Predictions**
You can use the trained model to make predictions on new passenger data:
```python
import joblib
import numpy as np

# Load the trained model
model = joblib.load("titanic_model.pkl")
scaler = joblib.load("scaler.pkl")

# Example passenger input: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
new_passenger = np.array([[3, 1, 28, 0, 0, 7.25, 2]])
new_passenger_scaled = scaler.transform(new_passenger)
prediction = model.predict(new_passenger_scaled)

print("Survived" if prediction[0] == 1 else "Did Not Survive")
```

🚀 Future Improvements
    - Try **deep learning (Neural Networks)** using TensorFlow.
    - Deploy as a **Flask API or Streamlit Web App**.
    - Use **SHAP (SHapley Additive Explanations)** to interpret model predictions.

📜 License
    This project is for educational purposes only. Free to use and modify!
    
-------------------------------------------------------

✨ **Built with Python & Machine Learning** ✨



