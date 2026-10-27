
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================
# PRACTICAL 5: UNDERFITTING VS OVERFITTING
# ============================================

# STEP 1: Generate sample student data
rng = np.random.default_rng(42)

study_hours = rng.uniform(1, 10, 200)

exam_scores = (
    15 + 16 * study_hours
    - 0.8 * study_hours**2
    + rng.normal(0, 5, 200)
)

exam_scores = np.clip(exam_scores, 0, 100)

X = study_hours.reshape(-1, 1)
y = exam_scores

# STEP 2: Split 70% / 15% / 15%
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("===== DATASET INFORMATION =====")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# STEP 3: Compare polynomial degrees
degrees = [1, 2, 5, 15]

models = {}
training_errors = []
validation_errors = []

print("\n===== MODEL COMPARISON =====")

for degree in degrees:

    # Scaling helps numerical stability
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )

    # Train using training data only
    model.fit(X_train, y_train)

    # Predict training and validation scores
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_mae = mean_absolute_error(
        y_train, train_pred
    )

    val_mae = mean_absolute_error(
        y_val, val_pred
    )

    val_r2 = r2_score(y_val, val_pred)

    models[degree] = model
    training_errors.append(train_mae)
    validation_errors.append(val_mae)

    print(f"\nPolynomial Degree {degree}")
    print("Training MAE:", round(train_mae, 2))
    print("Validation MAE:", round(val_mae, 2))
    print("Validation R2:", round(val_r2, 3))

# STEP 4: Select the best model
best_index = int(np.argmin(validation_errors))
best_degree = degrees[best_index]
best_model = models[best_degree]

print("\n===== SELECTED MODEL =====")
print("Best Polynomial Degree:", best_degree)

# STEP 5: Final testing
test_pred = best_model.predict(X_test)

print("\n===== FINAL TEST RESULTS =====")
print(
    "Testing MAE:",
    round(mean_absolute_error(y_test, test_pred), 2)
)
print(
    "Testing R2:",
    round(r2_score(y_test, test_pred), 3)
)

# STEP 6: Graph 1 - Compare model curves
plt.figure(
    "Practical 5 - Model Complexity",
    figsize=(11, 7)
)

plt.scatter(
    X_train, y_train,
    color="darkred",
    alpha=0.4,
    label="Training Data"
)

plt.scatter(
    X_val, y_val,
    color="orange",
    alpha=0.7,
    label="Validation Data"
)

x_line = np.linspace(1, 10, 300).reshape(-1, 1)

colors = ["blue", "purple", "green", "black"]

for degree, color in zip(degrees, colors):

    y_line = models[degree].predict(x_line)

    plt.plot(
        x_line,
        y_line,
        color=color,
        linewidth=2,
        label=f"Degree {degree}"
    )

plt.title(
    "Underfitting vs Overfitting",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Score (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "Practical_5_Model_Complexity.png",
    dpi=300,
    bbox_inches="tight"
)

# STEP 7: Graph 2 - Compare prediction errors
plt.figure(
    "Practical 5 - Training vs Validation Error",
    figsize=(10, 6)
)

plt.plot(
    degrees,
    training_errors,
    marker="o",
    color="darkred",
    linewidth=2,
    label="Training MAE"
)

plt.plot(
    degrees,
    validation_errors,
    marker="s",
    color="blue",
    linewidth=2,
    label="Validation MAE"
)

plt.title(
    "Training vs Validation Error",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Absolute Error (MAE)")
plt.xticks(degrees)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "Practical_5_Error_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nBoth graphs saved successfully!")
print("1. Practical_5_Model_Complexity.png")
print("2. Practical_5_Error_Comparison.png")

plt.show()
