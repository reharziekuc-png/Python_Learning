
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Student data
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
exam_scores = np.array([35, 40, 50, 55, 65, 70, 80, 90])

# Train the Machine Learning model
model = LinearRegression()
model.fit(study_hours, exam_scores)

# Predict exam scores
predictions = model.predict(study_hours)

# Predict score for 9 hours of studying
predicted_score = model.predict([[9]])
print("Predicted score for 9 hours:", round(predicted_score[0], 2))

# Plot original student data
plt.scatter(study_hours.flatten(), exam_scores,
            color="darkred", label="Actual Scores")

# Plot Machine Learning prediction line
plt.plot(study_hours.flatten(), predictions,
         color="blue", label="Prediction Line")

plt.title("Machine Learning: Study Hours vs Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores (%)")
plt.legend()
plt.grid(True)
plt.show()
