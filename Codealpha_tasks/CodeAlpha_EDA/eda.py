import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("student_performance_eda.csv")

print(df.head())
# Questions for EDA:
# 1. What is the average final score?
# 2. Does study time affect final score?
# 3. Does attendance affect final score?
# 4. What is the average score for each gender?
# 5. Are there unusual scores or study-hour values?
# 6. Is there a relationship between previous score and final score?
print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns)

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- DATA INFORMATION ---")
df.info()

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after cleaning:", df.duplicated().sum())
print("\n--- MISSING VALUES AFTER CLEANING ---")
print(df.isnull().sum())
print("\n--- BASIC STATISTICS ---")

print("Average Final Score:",
      df['Final_Score'].mean())

print("Highest Final Score:",
      df['Final_Score'].max())

print("Lowest Final Score:",
      df['Final_Score'].min())

print("Average Study Hours:",
      df['Study_Hours'].mean())

print("Average Attendance:",
    df['Attendance'].mean())
study_correlation = df['Study_Hours'].corr(
    df['Final_Score']
)

print("\nStudy Hours vs Final Score Correlation:",study_correlation)
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x='Study_Hours',
    y='Final_Score',
    data=df
)

plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")

plt.tight_layout()
plt.savefig("study_hours_vs_score.png")
plt.show()
attendance_correlation = df['Attendance'].corr(
    df['Final_Score']
)

print("\nAttendance vs Final Score Correlation:",
      attendance_correlation)
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x='Attendance',
    y='Final_Score',
    data=df
)

plt.title("Attendance vs Final Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")

plt.tight_layout()
plt.savefig("attendance_vs_score.png")
plt.show()
gender_scores = df.groupby('Gender')['Final_Score'].mean()

print("\n--- AVERAGE SCORE BY GENDER ---")
print(gender_scores)
plt.figure(figsize=(7, 5))

gender_scores.plot(kind='bar')

plt.title("Average Final Score by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Final Score")

plt.tight_layout()
plt.savefig("gender_average_score.png")
plt.show()
previous_correlation = df['Previous_Score'].corr(
    df['Final_Score']
)

print("\nPrevious Score vs Final Score Correlation:",
      previous_correlation)
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x='Previous_Score',
    y='Final_Score',
    data=df
)

plt.title("Previous Score vs Final Score")
plt.xlabel("Previous Score")
plt.ylabel("Final Score")

plt.tight_layout()
plt.savefig("previous_vs_final_score.png")
plt.show()
plt.figure(figsize=(8, 5))

sns.boxplot(x=df['Study_Hours'])

plt.title("Study Hours - Outlier Detection")
plt.xlabel("Study Hours")

plt.tight_layout()
plt.savefig("study_hours_outliers.png")
plt.show()
plt.figure(figsize=(8, 6))

correlation_matrix = df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()
print("\n--- FINAL DATA QUALITY CHECK ---")

print("Missing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)