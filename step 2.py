import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("cleaned_data.csv")
X = df.drop(columns=["event"])
y = df["event"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Train
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Evaluate 80/20
print("Accuracy (80/20):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')
print("CV Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Std:", scores.std())
