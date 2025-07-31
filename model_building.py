import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Step 1: Load your dataset
df = pd.read_csv("Telco-Customer-Churn.csv")  # Replace with your actual file name

# Step 2: Drop unnecessary columns
df.drop(columns=["customerID"], inplace=True)

# Step 3: Encode the target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Step 4: Convert categorical variables to numeric (One-hot encoding)
df_encoded = pd.get_dummies(df)

# Step 5: Split into features and target
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Step 6: Save training column names for use in Streamlit app
with open("training_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# Step 7: Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Step 8: Save trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and training_columns.pkl saved successfully!")
