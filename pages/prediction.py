import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title("Customer Category Prediction (Case: Turkey Customer)")
st.write("Prediction Customer in Turkey with Probability Using Ensemble Technique Based")

# Load CSS style
with open('static/styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model pickle
pickle_model = open('model/best_model_rf.pkl', 'rb')
classifier_model = pickle.load(pickle_model)

# Load Dataset
retail = pd.read_csv('data/customer_shopping_data.csv')

X = retail.loc[:, ['age', 'gender', 'price', 'payment_method', 'shopping_mall']]
y = retail[['category']]

# Encode categorical variables
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])
X['payment_method'] = le.fit_transform(X['payment_method'])
X['shopping_mall'] = le.fit_transform(X['shopping_mall'])
y = le.fit_transform(y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# Preprocessing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Outlier detection using Z-Score
z_scores = np.abs(zscore(X_train))
threshold = 5
outliers = np.where(z_scores > threshold)

X_train = X_train[(z_scores < threshold).all(axis=1)]
y_train = y_train[(z_scores < threshold).all(axis=1)]

# Modelling with Random Forest
classifier = RandomForestClassifier(n_estimators=300, random_state=44)
# RFE feature selection
rfe = RFE(classifier, n_features_to_select=5)
X_train = rfe.fit_transform(X_train, y_train)
X_test = rfe.transform(X_test)
# Fitting the model
classifier.fit(X_train, y_train)
# Predict the model
y_pred = classifier.predict(X_test)

# Evaluasi Metrics
## Accuracy
accuracy = accuracy_score(y_test, y_pred)
## Precision
precision = precision_score(y_test, y_pred, average='weighted')
## Recall
recall = recall_score(y_test, y_pred, average='weighted')
## F1-Score
f1_score = f1_score(y_test, y_pred, average='weighted')

# Evaluation Metrics 
with st.expander("EVALUATION METRICS"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ACCURACY", value = f'{accuracy:.4f}', delta = 'Accuracy Score')
    col2.metric("PRECISION", value = f'{precision:.4f}', delta = 'Precision Score With Weighted Average')
    col3.metric("RECALL", value = f'{recall:.4f}', delta = 'Recall Score With Weighted Average')
    col4.metric("F1 SCORE", value = f'{f1_score:.4f}', delta = 'F1 Score with Weighted Average')
    style_metric_cards(background_color= '#FFFFFF', border_left_color='#9900AD', border_color='#1F66BD', box_shadow='#F71938')
    st.write(f"<b>NOTES</b>: Hasil evaluasi metriks yang diterapkan sangat baik dan sudah sesuai dengan hasil pelatihan model algoritma Random Forest.", unsafe_allow_html=True)

# Prediction Table 
with st.expander("PREDICTION TABLE"):
    # Create dataframe for feature data and prediction
    prediction_table = pd.DataFrame({'age': X_test[:, 0].ravel(), 
                                     'gender': X_test[:, 1].ravel(),
                                     'price': X_test[:, 2].ravel(),
                                     'payment_method': X_test[:, 3].ravel(),
                                     'shopping_mall': X_test[:, 4].ravel(),
                                     'Category | Actual Y': y_test.ravel(), 
                                     'Y_Predicted': y_pred.ravel()})
    # Add evaluation metrics 
    if len(y_test) == len(y_pred) == len([accuracy] * len(y_test)) == len([precision] * len(y_test)) == len([recall] * len(y_test)) == len([f1_score] * len(y_test)):
        # Add evaluation metrics
        prediction_table['Accuracy'] = [accuracy] * len(y_test)
        prediction_table['Precision'] = [precision] * len(y_test)
        prediction_table['Recall'] = [recall] * len(y_test)
        prediction_table['F1 Score'] = [f1_score] * len(y_test)
    else:
        st.error("Error: All arrays must be of the same length!")
    
    st.dataframe(prediction_table, use_container_width=True)
    st.write(f'<b>NOTES</b>: Pada bagian tabel prediksi ini menggunakan data yang telah diolah sebelumnya sehingga sangat berbeda dengan data asli.', unsafe_allow_html=True)

# Download Predicted Table in CSV
df_predict = prediction_table.to_csv(index = False).encode('utf-8')
st.download_button(label = "DOWNLOAD PREDICTED DATA",
                   data = df_predict,
                   key = "download_predict.csv",
                   file_name = 'data_predict.csv') 

# Confusion Matrix and Feature Importance
with st.expander("CONFUSION MATRIX & FEATURE IMPORTANCE"):
    col1, col2 = st.columns(2)
    with col1:
        # Create class target
        target_names = ['Books', 'Clothing', 
                        'Cosmetics', 'Food & Beverage', 
                        'Shoes', 'Souvenir', 
                        'Technology', 'Toys']
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(15, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels = target_names, yticklabels = target_names)
        plt.title('Confusion Matrix Customer Category Prediction')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        st.pyplot(fig=plt, use_container_width=True)

    with col2:
        # Feature Importance Model
        feature_importance = classifier.feature_importances_
        feature_names = ['age', 'gender', 'price', 'payment_method', 'shopping_mall']
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
        importance_df = importance_df.sort_values("Importance", ascending=True)
        bar = px.bar(importance_df, x = 'Importance', y = 'Feature')
        title = bar.update_layout(title = {'text': 'Feature Importance Model Random Forest', 
                                           'xanchor': 'center', 
                                           'yanchor': 'top', 
                                           'x': 0.5, 
                                           'y': 0.95})
        
        st.plotly_chart(bar, use_container_width=True)
    st.write(f'<b>NOTES</b>: Hasil feature importance menunjukkan data fitur Price lebih dominan dibandingkan fitur lainnya dan evaluasi dengan Confusion Matrix terlihat sudah sangat cukup baik dalam hal identifikasi tiap kategori.', unsafe_allow_html=True)

#------------ PREDICT NEW DATA ---------
with st.expander("PREDICT NEW DATA"):
    # Input form for new data
    with st.form("input_form", clear_on_submit = True):
        # Input Age
        x1 = st.number_input("Age", min_value = 0, max_value = 100)
        # Input Gender
        x2 = st.selectbox("Gender", ["Male", "Female"])
        # Input Price
        x3 = st.number_input("Price", min_value = 0.0, max_value = 10000.0, step = 0.1)
        # Input Payment Method
        x4 = st.selectbox("Payment Method", ["Cash", "Credit Card", "Debit Card"])
        # Input Shopping Mall
        x5 = st.selectbox("Shopping Mall", ["Mall of Istanbul", "Kanyon", 
                                            "Metrocity", "Metropol AVM", 
                                            "Istinye Park", "Zorlu Center", 
                                            "Cevahir AVM", "Forum Istanbul", 
                                            "Viaport Outlet", "Emaar Square Mall"])
        # Submit button for Predict
        submitted = st.form_submit_button(label = "PREDICT")

if submitted:
    # Create a pandas DataFrame for new data
    new_data = pd.DataFrame({'age': [x1], 'gender': [x2], 'price': [x3], 'payment_method': [x4], 'shopping_mall': [x5]})
    # Encode categorical data
    le_gender = LabelEncoder()
    le_payment_method = LabelEncoder()
    le_shopping_mall = LabelEncoder()
    new_data['gender'] = le_gender.fit_transform(new_data['gender'])
    new_data['payment_method'] = le_payment_method.fit_transform(new_data['payment_method'])
    new_data['shopping_mall'] = le_shopping_mall.fit_transform(new_data['shopping_mall'])

    # Scale data
    new_data = scaler.transform(new_data)

    # Apply RFE
    new_data_rfe = rfe.transform(new_data.reshape(1, -1))  # reshape to (1, -1) because RFE expects 2D array
    
    # Make prediction
    predict_category = classifier.predict(new_data_rfe)

    # Inverse transform prediction
    prediction = le.inverse_transform(predict_category)
    
    st.write(f"<span style='font-size:34px; color:green;'>Predicted Category: </span> <span style='font-size:34px;'>{prediction[0]}</span>", unsafe_allow_html=True)
    

