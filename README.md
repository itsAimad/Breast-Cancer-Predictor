# Breast Cancer Prediction App 🩺🎗️

This is a **Streamlit-based web application** that predicts breast cancer using **Logistic Regression**. The app allows users to adjust feature values using **sidebar sliders** and provides a real-time graphical representation of the selected inputs. It also displays a **probability score** and a **final prediction** indicating whether the tumor is **benign** or **malignant**. 

## Features 🚀
- **Interactive Sliders**: Users can adjust feature values dynamically.
- **Real-time Graphs**: Displays selected values in a graphical format using **Plotly Express**.
- **Machine Learning Model**: Utilizes **Logistic Regression** for prediction.
- **Prediction Output**:
  - **Probability Score** (`predict_proba`): The likelihood of the tumor being benign or malignant.
  - **Final Classification**: Displays whether the tumor is **Benign** or **Malignant**.

## Technologies Used 🛠️
- **Streamlit**: Web application framework for interactive data apps.
- **Scikit-learn**: Used for **Logistic Regression** model.
- **Pickle**: Saves and loads the trained model.
- **Plotly Express**: Visualizes user inputs in an interactive graph.
- **Pandas**: Process our dataset .
- **Numpy**: Manipulations.

## Installation & Usage 📌
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/breast-cancer-predictor.git
cd breast-cancer-predictor
<<<<<<< HEAD
python -m streamlit run app&
=======
python -m streamlit run app/main.py
>>>>>>> 736fb5608c391515071454a2ceb2dc0ad38d6784
