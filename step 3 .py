# Feature Selection for Railway Impact Events
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("cleaned_data.csv")
X = df.drop(columns=["event"])
y = df["event"]

# Method 1: Pearson Correlation Heatmap
corr = X.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Figure 1. Pearson Correlation Heatmap")
plt.tight_layout()
plt.show()


# Method 2: Chi-Square Test
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

chi2_selector = SelectKBest(score_func=chi2, k=10)
chi2_selector.fit(X_scaled, y)

chi2_features = X.columns[chi2_selector.get_support()]
chi2_scores = chi2_selector.scores_

print("Top features by Chi2:", chi2_features.tolist())

# Barplot for Chi2 scores
plt.figure(figsize=(12, 6))
plt.bar(chi2_features, chi2_scores[chi2_selector.get_support()])
plt.title("Figure 2. Chi-Square Feature Scores")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Method 3: Wrapper Method - RFE with RandomForest
rf_for_rfe = RandomForestClassifier(random_state=42)
rfe = RFE(estimator=rf_for_rfe, n_features_to_select=10)
rfe.fit(X, y)

rfe_selected = X.columns[rfe.support_]
rfe_importances = rf_for_rfe.fit(X, y).feature_importances_
rfe_scores = [rfe_importances[list(X.columns).index(f)] for f in rfe_selected]

print("Top features by RFE:", rfe_selected.tolist())

# Barplot for RFE
plt.figure(figsize=(12, 6))
plt.bar(rfe_selected, rfe_scores)
plt.title("Figure 3. Top 10 Features Selected by RFE with Random Forest")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Method 4: Embedded Method - Random Forest Feature Importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
top_indices = indices[:10]
top_features_rf = [X.columns[i] for i in top_indices]
top_importances = importances[top_indices]

print("Top features by RF importance:", top_features_rf)

# Barplot for Random Forest importances
plt.figure(figsize=(12, 6))
plt.bar(top_features_rf, top_importances)
plt.title("Figure 4. Feature Importance – Random Forest")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
