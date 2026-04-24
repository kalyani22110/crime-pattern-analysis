# app/analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from mlxtend.frequent_patterns import apriori, association_rules

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "crime_dataset.csv")

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

# =========================
# CLEAN COLUMN NAMES
# =========================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# =========================
# COLUMN MAPPING
# =========================
date_col = "date_of_occurrence"
time_col = "time_of_occurrence"
location_col = "city"
crime_col = "crime_domain"

# =========================
# PREPROCESSING
# =========================
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

df['year'] = df[date_col].dt.year
df['month'] = df[date_col].dt.month

df['hour'] = pd.to_datetime(df[time_col], errors='coerce').dt.hour

df = df.dropna(subset=[date_col, location_col, crime_col])
df['hour'] = df['hour'].fillna(df['hour'].median())

print("After cleaning:", df.shape)

# =========================
# GRAPHS
# =========================
sns.countplot(x='year', data=df)
plt.title("Crimes per Year")
plt.show()

sns.countplot(x='month', data=df)
plt.title("Crimes per Month")
plt.show()

df[crime_col].value_counts().head(10).plot(kind='bar')
plt.title("Top Crimes")
plt.show()

# =========================
# HOTSPOTS (K-MEANS)
# =========================
le_loc = LabelEncoder()
df['loc_encoded'] = le_loc.fit_transform(df[location_col])

kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['loc_encoded']])

plt.scatter(df['loc_encoded'], df['hour'], c=df['cluster'])
plt.title("Crime Clusters (K-Means)")
plt.show()

# =========================
# MACHINE LEARNING
# =========================
le_crime = LabelEncoder()
df[crime_col] = le_crime.fit_transform(df[crime_col])

X = df[['year', 'month', 'hour', 'loc_encoded']]
y = df[crime_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

# =========================
# APRIORI (PATTERN MINING)
# =========================
df_ap = df[[location_col, crime_col]].copy()

df_ap[crime_col] = le_crime.inverse_transform(df_ap[crime_col])

# One-hot encoding
df_ap = pd.get_dummies(df_ap)

frequent = apriori(df_ap, min_support=0.05, use_colnames=True)

rules = association_rules(frequent, metric="confidence", min_threshold=0.5)

print("\nTop Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']].head())

# =========================
# PREDICTION
# =========================
sample = pd.DataFrame([[2024, 5, 20, 1]],
                      columns=['year', 'month', 'hour', 'loc_encoded'])

pred = rf.predict(sample)
print("\nPredicted Crime:", le_crime.inverse_transform(pred)[0])

# =========================
# INSIGHTS
# =========================
print("\n==== INSIGHTS ====")

print("Most Common Crime:",
      le_crime.inverse_transform([df[crime_col].mode()[0]])[0])

print("Most Dangerous Location:",
      le_loc.inverse_transform([df['loc_encoded'].mode()[0]])[0])

print("Peak Crime Hour:",
      int(df['hour'].mode()[0]))