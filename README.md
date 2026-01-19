# 🧍‍♂️ MediaPipe Pose Detection – Houding Corrector

Een client-side JavaScript applicatie die met behulp van MediaPipe Pose en ml5.js (KNN) de houding van een gebruiker herkent via de webcam.
De gebruiker traint zelf een model met eigen pose-data en krijgt daarna real-time feedback op zijn of haar houding.




# 🎯 Doel van dit project

Het doel van dit project is om te laten zien dat ik:

	•	beeldherkenning kan toepassen in een JavaScript applicatie
	•	zelf pose-data kan verzamelen via een webcam
	•	deze data kan voorbewerken (normaliseren)
	•	een machine-learning model kan trainen
	•	prestaties van het model kan evalueren met train/test split, accuracy en een confusion matrix



## Gebruikte technologieën

	•	MediaPipe Pose – real-time pose detection via webcam
	•	ml5.js – KNN classifier voor machine learning
	•	JavaScript (client-side)
	•	HTML5 Canvas – visualisatie van pose landmarks
	•	CSS – minimalistische, professionele UI

Alles draait volledig in de browser (geen backend).


## Machine Learning workflow

	1.	Verzamel pose-data (goede & slechte houding)
	2.	Data wordt genormaliseerd
	3.	Dataset wordt gesplitst:
	  •	80% training
	  •	20% test
	4.	Model traint alleen op trainingsdata
	5.	Test accuracy + confusion matrix worden berekend
	6.	Model voorspelt live op nieuwe webcam input

## Bekende beperkingen

	•	Het model is per gebruiker (data wordt lokaal opgeslagen)
	•	Resultaten zijn afhankelijk van:
	  •	licht
	  •	camera-hoek
	  • consistentie van houding
	•	KNN is gevoelig voor ruis, maar dit is bewust gekozen voor educatieve doeleinden


Dit project laat zien hoe beeldherkenning en machine learning op een toegankelijke manier in een webapplicatie kunnen worden geïntegreerd.
Door eigen data te verzamelen, te normaliseren en kritisch te evalueren met test metrics, ontstaat inzicht in zowel de mogelijkheden als beperkingen van pose-based classificatie.
