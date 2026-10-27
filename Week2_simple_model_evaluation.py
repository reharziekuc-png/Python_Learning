
# BAI6273 MACHINE LEARNING
# WEEK 2 - SIMPLE MODEL EVALUATION

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# STEP 1: Create fictional student data
hours = np.array([
    1, 2, 3, 4, 5, 6, 7, 8,
    2, 4, 6, 8, 3, 5, 7, 9,
    1, 4, 6, 9
]).reshape(-1, 1)

scores = np.array([
    35, 40, 50, 55, 65, 70, 80, 90,
    42, 58, 68, 85, 48, 62, 78, 92,
    38, 53, 73, 95
])

# STEP 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    hours,
    scores,
    test_size=0.20,
    random_state=42
)

print("===== TRAIN / TEST SPLIT =====")
print("Training students:", len(X_train))
print("Testing students:", len(X_test))

# STEP 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 4: Predict testing scores
predictions = model.predict(X_test)

# STEP 5: Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n===== MODEL EVALUATION =====")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# STEP 6: Compare actual and predicted scores
print("\n===== ACTUAL VS PREDICTED =====")

for actual, predicted in zip(y_test, predictions):
    print(
        f"Actual: {actual:.0f} | "
        f"Predicted: {predicted:.2f}"
    )

# STEP 7: Draw one simple graph
plt.figure(
    num="Week 2 - Simple Model Evaluation",
    figsize=(8, 5)
)

plt.scatter(
    X_train,
    y_train,
    color="darkred",
    label="Training Data"
)

plt.scatter(
    X_test,
    y_test,
    color="darkgreen",
    label="Testing Data"
)

line_hours = np.linspace(1, 9, 100).reshape(-1, 1)

plt.plot(
    line_hours,
    model.predict(line_hours),
    color="navy",
    label="Regression Line"
)

plt.title("Linear Regression - Training and Testing")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    "Week2_Simple_Model_Evaluation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
