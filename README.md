# 🫁 BreathCare - Respiratory Disease Prediction System

## 📌 Overview
BreathCare is an AI-based system designed to predict respiratory diseases using audio signals. The system analyzes breathing or voice recordings and identifies patterns associated with various respiratory conditions such as Normal, Asthma, Bronchitis, and Pneumonia. It uses audio processing and machine learning techniques to provide fast and reliable predictions.

---

## 🎯 Features
- Upload or record respiratory audio
- Audio preprocessing (noise removal, normalization)
- Feature extraction using:
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - RMS (Root Mean Square Energy)
  - ZCR (Zero Crossing Rate)
- Machine learning-based classification
- Real-time prediction results
- User-friendly interface using Streamlit

---

## 🛠️ Technologies Used
- Python (3.10+)
- Librosa (Audio processing)
- NumPy (Numerical computation)
- Scikit-learn (Machine learning)
- Matplotlib (Visualization)
- Streamlit (Web interface)

---

## 🧠 Machine Learning Model
- Model Used: Random Forest Classifier
- Input Features: MFCC, RMS, ZCR
- Output Classes:
  - Normal
  - Asthma
  - Bronchitis
  - Pneumonia

---

## 📂 Project Structure
BreathCare/
│── app.py
│── model.pkl
│── requirements.txt
│── README.md
│── dataset/ (optional)
│── images/

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository
git clone https://github.com/YOUR_USERNAME/BreathCare.git
cd BreathCare

### Step 2: Install Dependencies
pip install -r requirements.txt

### Step 3: Run the Application
streamlit run app.py

---

## 📊 How It Works
1. User uploads or records respiratory audio  
2. Audio is preprocessed to remove noise  
3. Features such as MFCC, RMS, and ZCR are extracted  
4. Machine learning model analyzes the features  
5. System predicts the respiratory condition  
6. Result is displayed in the user interface  

---

## 📈 Results
- The system successfully classifies respiratory conditions  
- Different diseases show distinct feature patterns  
- Provides quick and real-time predictions  

---

## 🔮 Future Enhancements
- Improve accuracy using deep learning models (CNN, RNN)  
- Train the system with larger and more diverse datasets  
- Develop a mobile application for real-time monitoring  
- Integrate with wearable devices  
- Deploy on cloud for scalability and accessibility  

---

## 📚 Dataset
- ICBHI Respiratory Sound Dataset  
- Coswara Dataset  

---

## 👨‍💻 Author
Your Name  

---

## 📌 License
This project is developed for academic and research purposes.
