
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# PRACTICAL 4: MODEL COMPARISON
# Linear vs Polynomial Regression
# ==========================================

# STEP 1: Generate sample student data
rng = np.random.default_rng(42)

study_hours = rng.uniform(1, 10, 200)

# Curved relationship: improvement slows down
exam_scores = (
    15 + 16 * study_hours
    - 0.8 * study_hours**2
    + rng.normal(0, 5, 200)
)

# Keep scores between 0 and 100
exam_scores = np.clip(exam_scores, 0, 100)

X = study_hours.reshape(-1, 1)
y = exam_scores

# STEP 2: Split data into three subsets
# 70% training, 15% validation, 15% testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

print("===== DATASET INFORMATION =====")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

# STEP 3: Create the models

linear_model = LinearRegression()

polynomial_model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

# STEP 4: Train both models
linear_model.fit(X_train, y_train)
polynomial_model.fit(X_train, y_train)

# STEP 5: Evaluate using validation data

models = {
    "Linear Regression": linear_model,
    "Polynomial Regression": polynomial_model
}

validation_scores = {}

print("\n===== VALIDATION RESULTS =====")

for name, model in models.items():

    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)

    validation_scores[name] = mae

    print(f"\n{name}")
    print("Validation MAE:", round(mae, 2))
    print("Validation R2:", round(r2, 3))

# STEP 6: Select the best model
# Lower validation MAE is better

best_name = min(
    validation_scores,
    key=validation_scores.get
)

best_model = models[best_name]

print("\n===== MODEL SELECTION =====")
print("Selected Model:", best_name)

# STEP 7: Final testing
# Test data is used only after model selection

test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(
    y_test, test_predictions
)

test_r2 = r2_score(
    y_test, test_predictions
)

print("\n===== FINAL TESTING RESULTS =====")
print("Testing MAE:", round(test_mae, 2))
print("Testing R2:", round(test_r2, 3))

# STEP 8: Create a custom graph window

plt.figure(
    "Practical 4 - Linear vs Polynomial Regression",
    figsize=(10, 7)
)

# Plot training data
plt.scatter(
    X_train, y_train,
    color="darkred",
    alpha=0.4,
    label="Training Data (70%)"
)

# Plot validation data
plt.scatter(
    X_val, y_val,
    color="orange",
    alpha=0.8,
    label="Validation Data (15%)"
)

# Plot testing data
plt.scatter(
    X_test, y_test,
    color="green",
    alpha=0.8,
    label="Testing Data (15%)"
)

# STEP 9: Draw both regression lines

x_line = np.linspace(1, 10, 200).reshape(-1, 1)

plt.plot(
    x_line,
    linear_model.predict(x_line),
    color="blue",
    linewidth=2,
    label="Linear Regression"
)

plt.plot(
    x_line,
    polynomial_model.predict(x_line),
    color="purple",
    linewidth=2,
    label="Polynomial Regression"
)

# STEP 10: Customize graph

plt.title(
    "Linear vs Polynomial Regression",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Study Hours", fontsize=12)
plt.ylabel("Exam Score (%)", fontsize=12)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# STEP 11: Save graph for our PDF handbook

plt.savefig(
    "Practical_4_Model_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nGraph saved successfully!")
print("Filename: Practical_4_Model_Comparison.png")

# STEP 12: Display graph

plt.show()
