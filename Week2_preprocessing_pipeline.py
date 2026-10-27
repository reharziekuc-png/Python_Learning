
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==================================================
# BAI6273 MACHINE LEARNING - WEEK 2
# PRACTICAL A - STAGE 2
# IMPUTATION, ENCODING AND SCALING
# ==================================================

# STEP 1: Create the original student dataset
df = pd.DataFrame({
    "hours": [2, 4, None, 7, 5],
    "attendance": [60, 75, 82, 105, 80],
    "mode": [
        "Campus", "Online", "Campus",
        "Online", "campus"
    ],
    "score": [48, 60, 70, 85, 72]
})

print("===== ORIGINAL DATASET =====")
print(df)

# STEP 2: Identify invalid attendance
invalid = ~df["attendance"].between(0, 100)

# Flag invalid values as missing
df.loc[invalid, "attendance"] = np.nan

# STEP 3: Normalize text categories
df["mode"] = df["mode"].str.strip().str.lower()

print("\n===== AFTER INITIAL CLEANING =====")
print(df)

# STEP 4: Separate features and target
X = df[["hours", "attendance", "mode"]]
y = df["score"]

# STEP 5: Split before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== TRAIN / TEST SPLIT =====")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 6: Define numerical preprocessing
numeric_features = ["hours", "attendance"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# STEP 7: Define categorical preprocessing
categorical_features = ["mode"]

categorical_pipeline = Pipeline([
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

# STEP 8: Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# STEP 9: Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# STEP 10: Transform testing data
# IMPORTANT: Do NOT fit again on test data
X_test_processed = preprocessor.transform(X_test)

# STEP 11: Display processed results
feature_names = preprocessor.get_feature_names_out()

print("\n===== PROCESSED FEATURE NAMES =====")
print(feature_names)

print("\n===== PROCESSED TRAINING DATA =====")
print(
    pd.DataFrame(
        X_train_processed,
        columns=feature_names
    ).round(3)
)

print("\n===== PROCESSED TESTING DATA =====")
print(
    pd.DataFrame(
        X_test_processed,
        columns=feature_names
    ).round(3)
)

# STEP 12: Display the training medians
fitted_imputer = (
    preprocessor
    .named_transformers_["numeric"]
    .named_steps["imputer"]
)

print("\n===== TRAINING MEDIANS =====")

for name, median in zip(
    numeric_features,
    fitted_imputer.statistics_
):
    print(f"{name}: {median:.2f}")

print("\nPreprocessing completed successfully!")
