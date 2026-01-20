# 🧍‍♂️ MediaPipe Pose Detection – Posture Corrector

A client-side JavaScript application that uses MediaPipe Pose and ml5.js (KNN) to recognize a user’s posture through a webcam.
The user trains the model with their own pose data and then receives real-time feedback on their posture.



# 🎯 Project Goal

The goal of this project is to demonstrate that I am able to:

	•	apply computer vision in a JavaScript application
	•	collect pose data independently using a webcam
	•	preprocess and normalize this data
	•	train a machine learning model
	•	evaluate model performance using a train/test split, accuracy, and a confusion matrix



## Technologies Used

	•	MediaPipe Pose – real-time pose detection via webcam
	•	ml5.js – KNN classifier for machine learning
	•	JavaScript (client-side)
	•	HTML5 Canvas – visualization of pose landmarks
	•	CSS – minimalistic, professional user interface

Everything runs entirely in the browser (no backend).


## Machine Learning Workflow

	1.	Collect pose data (good and bad posture)
	2.	Normalize the pose data
	3.	Split the dataset:
	•	80% training
	•	20% testing
	4.	Train the model using only the training data
	5.	Calculate test accuracy and generate a confusion matrix
	6.	Predict posture live on new webcam input

## Known Limitations

	•	The model is user-specific (data is stored locally)
	•	Results depend on:
	•	lighting conditions
	•	camera angle
	•	consistency of posture
	•	KNN is sensitive to noise, which is a deliberate choice for educational purposes


This project demonstrates how computer vision and machine learning can be integrated into an accessible web application.
By collecting custom data, applying normalization, and critically evaluating performance using test metrics, the project provides insight into both the strengths and limitations of pose-based classification.
