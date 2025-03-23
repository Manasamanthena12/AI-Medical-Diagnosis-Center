# AI Medical Diagnosis Center 🩺

Welcome to the AI Medical Diagnosis Center! This is a web app that helps people check if they might have certain health problems early. It uses smart technology to predict seven diseases by looking at health details you enter, like your age or blood pressure. The app is easy to use and was created as part of the AICTE Internship on AI: Transformative Learning with TechSaksham, a joint CSR initiative of Microsoft & SAP.

## What Does This App Do? 🌟

The AI Medical Diagnosis Center lets you check for health issues in a simple way. You pick a disease, enter your health information, and the app tells you if you might have that disease. It’s like having a quick health check at home! The app has two pages:
- A **Home Page** where you choose a disease from seven options shown as blue cards.
- A **Details Page** where you type in your health details and see the result, like “Diabetes Detected,” with fun emojis (🎉) to make it friendly.

The app is online, so you can use it anywhere with the internet.

## Cool Things About the App ✨

- Checks for seven diseases: Diabetes, Heart Disease, Parkinson’s Disease, Lung Cancer, Breast Cancer, Brain Disease (Alzheimer’s), and Kidney Disease.
- Easy to use with a clear design, background pictures (like a stethoscope), and celebration emojis.
- Makes sure your inputs are correct (e.g., no zero for things like weight unless it’s a yes/no question).
- Available online for everyone to try without downloading anything.

## Tools We Used 🛠️

Here’s what we used to build the app:
- **Python**: The main language to write the app.
- **Streamlit**: To create the web pages and make them look nice.
- **Pandas**: To organize the health details you enter.
- **Scikit-learn**: To load the smart models that predict diseases.
- **Pickle**: To save and load the models (it’s part of Python, so we didn’t need to install it).

## What’s in the Project Folder 📂

- `app.py`: The main file that runs the app.
- `diabetes_model.pkl`, `heart_disease_model.pkl`, `parkinsons_model.pkl`, `lung_cancer_model.pkl`, `breast_cancer_model.pkl`, `brain_disease_model.pkl`, `kidney_disease_model.pkl`: The smart models for each disease.
- `stethoscope.jpg`, `doctor_patient.jpg`: Pictures used in the app’s background.
- `requirements.txt`: A list of tools the app needs to work (like Streamlit and Pandas).

## Try the App Online 🌐

You can use the app right now on Streamlit Cloud:  
https://ai-medical-diagnosis-center-a3gebhu8wrzbswmfy4i7d3.streamlit.app/ 

## Set Up the App on Your Computer 💻

Want to run the app on your own computer? Follow these steps:

### Things You Need
- Python 3.8 or higher.
- Git to download the project.

### Steps to Run
1. **Download the Project**  
   Open a terminal and type:  
