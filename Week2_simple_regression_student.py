
# BAI6273 MACHINE LEARNING
# WEEK 2 - SIMPLE LINEAR REGRESSION
# STUDENT PRACTICAL

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# STEP 1: Prepare student data
# X = hours studied
# y = examination scores

X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
y = np.array([35, 40, 50, 55, 65, 70, 80, 90])

# STEP 2: Create the machine learning model
model = LinearRegression()

# STEP 3: Train the model
model.fit(X, y)

# STEP 4: Predict the score for 9 study hours
prediction = model.predict([[9]])

print("===== STUDENT SCORE PREDICTION =====")
print(f"Predicted score for 9 hours: {prediction[0]:.2f}")

# STEP 5: Visualize the results
plt.figure(
    num="Week 2 - Simple Linear Regression",
    figsize=(8, 5)
)
plt.scatter(X.ravel(), y, color="darkred", label="Actual Scores")
plt.plot(X.ravel(), model.predict(X),
         color="navy", label="Regression Line")

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Scores")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

