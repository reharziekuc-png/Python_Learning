
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Create 200 sample student records
rng = np.random.default_rng(42)

study_hours = rng.uniform(1, 10, 200)
exam_scores = 25 + 7 * study_hours + rng.normal(0, 6, 200)
exam_scores = np.clip(exam_scores, 0, 100)

X = study_hours.reshape(-1, 1)
y = exam_scores

# 2. Split: 70% training, 30% remaining
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 3. Split remaining 30% equally
# 15% validation, 15% testing
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

# 4. Train model using training data only
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict validation and testing scores
val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

# 6. Evaluate model performance
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

print("\nValidation Results")
print("MAE:", round(mean_absolute_error(y_val, val_predictions), 2))
print("R2:", round(r2_score(y_val, val_predictions), 3))

print("\nTesting Results")
print("MAE:", round(mean_absolute_error(y_test, test_predictions), 2))
print("R2:", round(r2_score(y_test, test_predictions), 3))

# 7. Plot all three datasets
plt.scatter(X_train, y_train, color="darkred",
            label="Training (70%)", alpha=0.6)

plt.scatter(X_val, y_val, color="orange",
            label="Validation (15%)", alpha=0.8)

plt.scatter(X_test, y_test, color="green",
            label="Testing (15%)", alpha=0.8)

# 8. Draw regression line
x_line = np.linspace(1, 10, 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.plot(x_line, y_line, color="blue",
         linewidth=2, label="Regression Line")

plt.title("Machine Learning: Train / Validation / Test")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
