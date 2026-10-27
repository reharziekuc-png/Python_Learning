
# BAI6273 MACHINE LEARNING
# WEEK 2 - LINEAR VS POLYNOMIAL REGRESSION

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# STEP 1: Create fictional student data
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
scores = np.array([30, 42, 55, 65, 73, 78, 81, 83])

# STEP 2: Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(hours, scores)

# STEP 3: Create polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
hours_poly = poly.fit_transform(hours)

# STEP 4: Train a Polynomial Regression model
poly_model = LinearRegression()
poly_model.fit(hours_poly, scores)

# STEP 5: Create smooth lines for the graph
line_hours = np.linspace(1, 8, 100).reshape(-1, 1)

linear_predictions = linear_model.predict(line_hours)

poly_predictions = poly_model.predict(
    poly.transform(line_hours)
)

# STEP 6: Draw the graph
plt.figure(
    num="Week 2 - Linear vs Polynomial Regression",
    figsize=(8, 5)
)

plt.scatter(
    hours, scores,
    color="darkred",
    label="Actual Scores"
)

plt.plot(
    line_hours, linear_predictions,
    color="navy",
    label="Linear Regression"
)

plt.plot(
    line_hours, poly_predictions,
    color="darkgreen",
    label="Polynomial Regression"
)

plt.title("Linear vs Polynomial Regression")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    "Week2_Simple_Polynomial_Regression.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Practical completed successfully!")
