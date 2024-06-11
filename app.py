from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from extract_features import extract_features
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
import logging

app = Flask(__name__)

UPLOAD_FOLDER = 'C:/Users/Adi Sahrul R/OneDrive/Documents/Penambangan Data/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logging.basicConfig(level=logging.DEBUG)

classifier3 = None
classifier5 = None
meta_classifier = None
label_encoder = None
X_train, X_test, y_train, y_test = None, None, None, None
XTrain1, XTrain2, XTest1, XTest2 = None, None, None, None

def load_data():
    # Mengambil dataset Raisin
    raisin = fetch_ucirepo(id=850)
    
    # Ekstrak fitur dan target sebagai dataframe pandas
    raisin_features = raisin.data.features
    raisin_class = raisin.data.targets
    
    # Gabungkan fitur dan target menjadi satu dataframe
    df_raisin = raisin_features.join(raisin_class)
    
    # Pisahkan fitur dan label
    feature_columns = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", 
                    "ConvexArea", "Extent", "Perimeter"]
    X = df_raisin[feature_columns].values
    y = df_raisin['Class'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, label_encoder

def train_classifiers():
    global classifier3, classifier5, label_encoder
    global X_train, X_test, y_train, y_test
    global XTrain1, XTrain2, XTest1, XTest2
    
    X, y_encoded, label_encoder = load_data()
    
    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
    
    # Inisialisasi classifier KNN dengan k=3
    classifier3 = KNeighborsClassifier(n_neighbors=3)
    classifier3.fit(X_train, y_train)
    
    # Evaluasi model dengan k=3
    XTrain1 = classifier3.predict(X_train)
    train_accuracy3 = accuracy_score(y_train, XTrain1)
    XTest1 = classifier3.predict(X_test)
    test_accuracy3 = accuracy_score(y_test, XTest1)
    
    # Inisialisasi classifier KNN dengan k=5
    classifier5 = KNeighborsClassifier(n_neighbors=5)
    classifier5.fit(X_train, y_train)
    
    # Evaluasi model dengan k=5
    XTrain2 = classifier5.predict(X_train)
    train_accuracy5 = accuracy_score(y_train, XTrain2)
    XTest2 = classifier5.predict(X_test)
    test_accuracy5 = accuracy_score(y_test, XTest2)
    
    # Gabungkan hasil prediksi dan simpan ke file CSV
    combined_train_df = pd.DataFrame({
        'P1': XTrain1,
        'P2': XTrain2,
        'Y': y_train
    })
    combined_train_df1 = pd.DataFrame({
        'P1': label_encoder.inverse_transform(XTrain1),
        'P2': label_encoder.inverse_transform(XTrain2),
        'Y': label_encoder.inverse_transform(y_train)
    })
    combined_train_df.to_csv('combined_train.csv', index=False)
    combined_train_df1.to_csv('combined_train1.csv', index=False)
    
    combined_test_df = pd.DataFrame({
        'P1': XTest1,
        'P2': XTest2,
        'Y': y_test
    })
    combined_test_df1 = pd.DataFrame({
        'P1': label_encoder.inverse_transform(XTest1),
        'P2': label_encoder.inverse_transform(XTest2),
        'Y': label_encoder.inverse_transform(y_test)
    })
    combined_test_df.to_csv('combined_test.csv', index=False)
    combined_test_df1.to_csv('combined_test1.csv', index=False)
    
    return {
        'train_accuracy3': train_accuracy3,
        'test_accuracy3': test_accuracy3,
        'train_accuracy5': train_accuracy5,
        'test_accuracy5': test_accuracy5
    }

def train_meta_classifier():
    global meta_classifier, XTrain1, XTrain2, y_train
    # Gabungkan prediksi dari kedua model KNN sebagai fitur baru
    f_meta = np.column_stack((XTrain1, XTrain2))
    # Inisialisasi dan pelatihan meta-classifier Naive Bayes
    meta_classifier = GaussianNB()
    meta_classifier.fit(f_meta, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        features = extract_features(file_path)
        return render_template('index.html', features=features, image_name=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/train', methods=['POST'])
def train_model():
    accuracies = train_classifiers()
    train_meta_classifier()
    return jsonify(accuracies)

@app.route('/predict', methods=['POST'])
def predict():
    global classifier3, classifier5, meta_classifier, label_encoder
    try:
        if classifier3 is None or classifier5 is None or meta_classifier is None or label_encoder is None:
            raise Exception("Models not trained yet")
        
        # Ekstrak data inputan dan model yang dipilih
        input_data = request.json['data']
        model_choice = request.json['model']
        logging.debug(f"Received input data: {input_data}, model choice: {model_choice}")
        
        # Prediksi untuk instance baru
        new_data = [input_data]
        
        if model_choice == 'k3':
            prediction = classifier3.predict(new_data)
        elif model_choice == 'k5':
            prediction = classifier5.predict(new_data)
        elif model_choice == 'meta':
            # Gunakan model KNN untuk mendapatkan prediksi fitur
            knn_pred1 = classifier3.predict(new_data)
            knn_pred2 = classifier5.predict(new_data)
            combined_features = np.column_stack((knn_pred1, knn_pred2))
            # Prediksi dengan meta-classifier
            prediction = meta_classifier.predict(combined_features)
        else:
            return jsonify({'error': 'Invalid model choice'}), 400
        
        logging.debug(f"Prediction : {prediction}")
        
        return jsonify({
            'prediction': label_encoder.inverse_transform(prediction)[0]
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
