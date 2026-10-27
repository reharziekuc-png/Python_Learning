
import pandas as pd
import numpy as np

# BAI6273 - WEEK 2 PRACTICAL A
# DATA CLEANING WITH PANDAS

# Step 1: Create our fictional student dataset
df = pd.DataFrame({
    "hours": [2, 4, None, 7, 5],
    "attendance": [60, 75, 82, 105, 80],
    "mode": ["Campus", "Online", "Campus", "Online", "campus"],
    "score": [48, 60, 70, 85, 72]
})

# Step 2: Inspect the original dataset
print("===== ORIGINAL DATASET =====")
print(df)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== MISSING VALUES =====")
print(df.isna().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)

# Step 3: Identify invalid attendance values
invalid_attendance = ~df["attendance"].between(0, 100)

print("\n===== INVALID ATTENDANCE =====")
print(df.loc[invalid_attendance])

# Treat unverifiable invalid values as missing
df.loc[invalid_attendance, "attendance"] = np.nan

# Step 4: Standardize category labels
df["mode"] = df["mode"].str.strip().str.lower()

print("\n===== CLEANED LABELS =====")
print(df["mode"].unique())

# Step 5: Inspect the updated dataset
print("\n===== DATA AFTER INITIAL CLEANING =====")
print(df)

print("\n===== REMAINING MISSING VALUES =====")
print(df.isna().sum())
