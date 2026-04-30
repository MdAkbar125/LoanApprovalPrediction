import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
df = pd.read_csv("LoanApprovalPrediction.csv")

print("📊 Dataset Preview:")
print(df.head())

print("\n📌 Columns:", df.columns)

# -------------------------------
# Step 2: Handle Missing Values
# -------------------------------
for col in df.select_dtypes(include=['object', 'str']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].mean())

# -------------------------------
# Step 3: Fix Special Values
# -------------------------------
df['Dependents'] = df['Dependents'].replace('3+', '3')

# -------------------------------
# Step 4: Encode Categorical Data
# -------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'str']).columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Step 5: EDA (Visualization)
# -------------------------------
print("\n📊 Loan Status Distribution:")
df['Loan_Status'].value_counts().plot(kind='bar')
plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Step 6: Features & Target
# -------------------------------
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]

# -------------------------------
# Step 7: Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Step 8: Feature Scaling (IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Step 9: Logistic Regression Model
# -------------------------------
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

# -------------------------------
# Step 10: Decision Tree Model
# -------------------------------
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

# -------------------------------
# Step 11: Evaluation
# -------------------------------
print("\n📈 Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print("\n📊 Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

print("\n🌳 Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print("\n📊 Decision Tree Report:")
print(classification_report(y_test, y_pred_tree))

# -------------------------------
# Step 12: User Input Prediction (SAFE VERSION)
# -------------------------------
print("\n🔍 Enter values for prediction:")

try:
    user_input = []

    for col in X.columns:
        while True:
            value = input(f"{col}: ")
            try:
                value = float(value)
                user_input.append(value)
                break
            except:
                print("❌ Please enter a valid number")

    # Scale input before prediction
    user_input = scaler.transform([user_input])

    result = log_model.predict(user_input)

    if result[0] == 1:
        print("✅ Loan Approved")
    else:
        print("❌ Loan Rejected")

except:
    print("❌ Unexpected error")