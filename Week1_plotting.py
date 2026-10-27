import matplotlib.pyplot as plt

#Sample student data
study_hours = [1,2,3,4,5,6,7,8]
exam_scores = [35, 40, 50, 55, 65, 70, 80, 90]

#Create a scatter plot
plt.scatter(study_hours, exam_scores, color='darkred', s=100)

#Add graph labels
plt.title("Study Hours vs Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores (%)")

#Show grid
plt.grid(True)

#Display graph
plt.show()  
