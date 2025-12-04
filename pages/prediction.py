import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import onnxruntime as ort
import plotly.express as px
from scipy.stats import zscore
import matplotlib.pyplot as plt
from skl2onnx import convert_sklearn
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title("Customer Category Prediction (Case: Turkey Customer)")
st.write("Prediction Customer in Turkey with Probability Using Ensemble Technique Based")

# Load CSS style
with open('static/styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Dataset
retail = pd.read_csv('data/customer_shopping_data.csv')

X = retail.loc[:, ['age', 'gender', 'price', 'payment_method', 'shopping_mall']]
y = retail[['category']]

# Encode categorical variables
le = LabelEncoder()
X['gender'] = le.fit_transform(X['gender'])
X['payment_method'] = le.fit_transform(X['payment_method'])
X['shopping_mall'] = le.fit_transform(X['shopping_mall'])
y_encoded = le.fit_transform(y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=44)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Outlier detection using Z-Score
z_scores = np.abs(zscore(X_train_scaled))
threshold = 5
outliers = np.where(z_scores > threshold)

X_train_clean = X_train_scaled[(z_scores < threshold).all(axis=1)]
y_train_clean = y_train[(z_scores < threshold).all(axis=1)]

#------------ MODEL TRAINING SECTION ---------
with st.expander("🔄 MODEL TRAINING & MANAGEMENT"):
    st.subheader("Train or Load Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Training Parameters")
        n_estimators = st.slider("Number of Trees (n_estimators)", 
                                 min_value=50, max_value=500, value=300, step=50)
        test_size = st.slider("Test Size", 
                             min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        random_state = st.number_input("Random State", 
                                      min_value=0, max_value=100, value=44)
        n_features = st.slider("Number of Features to Select (RFE)", 
                              min_value=1, max_value=5, value=5)
        
        train_button = st.button("🚀 TRAIN NEW MODEL", type="primary")
    
    with col2:
        st.write("### Model Management")
        model_format = st.radio("Choose Model Format:", 
                               ["ONNX Model (.onnx)", "Pickle Model (.pkl)"])
        
        load_option = st.radio("Choose Model Source:", 
                              ["Load Existing Model", "Use Newly Trained Model"])
        
        if load_option == "Load Existing Model":
            if model_format == "ONNX Model (.onnx)":
                model_path = 'model/best_model_rf.onnx'
                metadata_path = 'model/model_metadata.pkl'
                if os.path.exists(model_path) and os.path.exists(metadata_path):
                    st.success("✅ ONNX model found!")
                    model_loaded = True
                    use_onnx = True
                else:
                    st.error("❌ ONNX model not found. Please train a new model first.")
                    model_loaded = False
                    use_onnx = False
            else:
                model_path = 'model/best_model_rf.pkl'
                if os.path.exists(model_path):
                    st.success("✅ Pickle model found!")
                    model_loaded = True
                    use_onnx = False
                else:
                    st.error("❌ Pickle model not found. Please train a new model first.")
                    model_loaded = False
                    use_onnx = False
        else:
            model_loaded = False
            use_onnx = False

# Initialize session state for model
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.trained_rfe = None
    st.session_state.trained_scaler = None
    st.session_state.trained_le = None
    st.session_state.model_metrics = None
    st.session_state.onnx_session = None

# Train new model
if train_button:
    with st.spinner("Training model... Please wait..."):
        # Re-split data with new test_size
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state
        )
        
        # Preprocessing
        scaler_new = StandardScaler()
        X_train_scaled_new = scaler_new.fit_transform(X_train_new)
        X_test_scaled_new = scaler_new.transform(X_test_new)
        
        # Outlier removal
        z_scores_new = np.abs(zscore(X_train_scaled_new))
        X_train_clean_new = X_train_scaled_new[(z_scores_new < threshold).all(axis=1)]
        y_train_clean_new = y_train_new[(z_scores_new < threshold).all(axis=1)]
        
        # Model training with RFE
        classifier_new = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        rfe_new = RFE(classifier_new, n_features_to_select=n_features)
        X_train_rfe = rfe_new.fit_transform(X_train_clean_new, y_train_clean_new)
        X_test_rfe = rfe_new.transform(X_test_scaled_new)
        
        # Fit the model
        classifier_new.fit(X_train_rfe, y_train_clean_new)
        
        # Predictions
        y_pred_new = classifier_new.predict(X_test_rfe)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test_new, y_pred_new),
            'precision': precision_score(y_test_new, y_pred_new, average='weighted'),
            'recall': recall_score(y_test_new, y_pred_new, average='weighted'),
            'f1_score': f1_score(y_test_new, y_pred_new, average='weighted')
        }
        
        # Save to session state
        st.session_state.trained_model = classifier_new
        st.session_state.trained_rfe = rfe_new
        st.session_state.trained_scaler = scaler_new
        st.session_state.trained_le = le
        st.session_state.model_metrics = metrics
        st.session_state.X_test = X_test_rfe
        st.session_state.y_test = y_test_new
        st.session_state.y_pred = y_pred_new
        
        # Save as Pickle
        model_package = {
            'classifier': classifier_new,
            'rfe': rfe_new,
            'scaler': scaler_new,
            'label_encoder': le,
            'metrics': metrics,
            'n_features': n_features
        }
        
        with open('model/best_model_rf.pkl', 'wb') as f:
            pickle.dump(model_package, f)
        
        # Convert and Save as ONNX
        try:
            # Define initial type for ONNX conversion
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            
            # Convert model to ONNX
            onnx_model = convert_sklearn(classifier_new, initial_types=initial_type, 
                                        target_opset=12)
            
            # Save ONNX model
            with open('model/best_model_rf.onnx', 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            # Save metadata (scaler, rfe, label_encoder) separately
            metadata = {
                'scaler': scaler_new,
                'rfe': rfe_new,
                'label_encoder': le,
                'metrics': metrics,
                'n_features': n_features,
                'feature_names': ['age', 'gender', 'price', 'payment_method', 'shopping_mall']
            }
            
            with open('model/model_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)
            
            st.success(f"✅ Model trained and saved successfully!")
            st.success(f"📊 Accuracy: {metrics['accuracy']:.4f}")
            st.success(f"💾 Saved as: Pickle (.pkl) and ONNX (.onnx)")
            
        except Exception as e:
            st.warning(f"⚠️ Model saved as Pickle only. ONNX conversion failed: {str(e)}")
        
        st.balloons()

# Determine which model to use
if load_option == "Use Newly Trained Model" and st.session_state.trained_model is not None:
    classifier = st.session_state.trained_model
    rfe = st.session_state.trained_rfe
    scaler = st.session_state.trained_scaler
    le_model = st.session_state.trained_le
    X_test_final = st.session_state.X_test
    y_test_final = st.session_state.y_test
    y_pred_final = st.session_state.y_pred
    
    accuracy = st.session_state.model_metrics['accuracy']
    precision = st.session_state.model_metrics['precision']
    recall = st.session_state.model_metrics['recall']
    f1 = st.session_state.model_metrics['f1_score']
    
    onnx_session = None
    st.info("🔵 Using newly trained model from this session")
    
elif model_loaded and use_onnx:
    # Load ONNX Model
    try:
        onnx_session = ort.InferenceSession('model/best_model_rf.onnx')
        
        # Load metadata
        with open('model/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        scaler = metadata['scaler']
        rfe = metadata['rfe']
        le_model = metadata['label_encoder']
        metrics = metadata.get('metrics', {})
        
        # Apply transformations
        X_train_rfe = rfe.fit_transform(X_train_clean, y_train_clean)
        X_test_final = rfe.transform(X_test_scaled)
        
        # Predict using ONNX
        input_name = onnx_session.get_inputs()[0].name
        label_name = onnx_session.get_outputs()[0].name
        
        y_pred_final = onnx_session.run([label_name], {input_name: X_test_final.astype(np.float32)})[0]
        y_test_final = y_test
        
        # Calculate metrics
        accuracy = metrics.get('accuracy', accuracy_score(y_test_final, y_pred_final))
        precision = metrics.get('precision', precision_score(y_test_final, y_pred_final, average='weighted'))
        recall = metrics.get('recall', recall_score(y_test_final, y_pred_final, average='weighted'))
        f1 = metrics.get('f1_score', f1_score(y_test_final, y_pred_final, average='weighted'))
        
        classifier = None  # ONNX doesn't need sklearn classifier
        
        st.info("🟢 Using ONNX model from file")
        
    except Exception as e:
        st.error(f"Failed to load ONNX model: {str(e)}")
        st.warning("Falling back to default model...")
        model_loaded = False
        use_onnx = False
        onnx_session = None

elif model_loaded and not use_onnx:
    # Load Pickle Model
    with open('model/best_model_rf.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict):
        classifier = model_data['classifier']
        rfe = model_data.get('rfe', None)
        scaler = model_data.get('scaler', scaler)
        le_model = model_data.get('label_encoder', le)
        
        if rfe is None:
            rfe = RFE(classifier, n_features_to_select=5)
        
        # Apply transformations
        X_train_rfe = rfe.fit_transform(X_train_clean, y_train_clean)
        X_test_final = rfe.transform(X_test_scaled)
        classifier.fit(X_train_rfe, y_train_clean)
        y_pred_final = classifier.predict(X_test_final)
        y_test_final = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_final, y_pred_final)
        precision = precision_score(y_test_final, y_pred_final, average='weighted')
        recall = recall_score(y_test_final, y_pred_final, average='weighted')
        f1 = f1_score(y_test_final, y_pred_final, average='weighted')
    else:
        classifier = model_data
        le_model = le
        
        if hasattr(classifier, 'named_steps') or hasattr(classifier, 'steps'):
            y_pred_final = classifier.predict(X_test)
            y_test_final = y_test
            X_test_final = X_test_scaled
            rfe = None
        else:
            rfe = RFE(classifier, n_features_to_select=5)
            X_train_rfe = rfe.fit_transform(X_train_clean, y_train_clean)
            X_test_final = rfe.transform(X_test_scaled)
            classifier.fit(X_train_rfe, y_train_clean)
            y_pred_final = classifier.predict(X_test_final)
            y_test_final = y_test
        
        accuracy = accuracy_score(y_test_final, y_pred_final)
        precision = precision_score(y_test_final, y_pred_final, average='weighted')
        recall = recall_score(y_test_final, y_pred_final, average='weighted')
        f1 = f1_score(y_test_final, y_pred_final, average='weighted')
    
    onnx_session = None
    st.info("🟢 Using Pickle model from file")

else:
    # Default: train on the fly
    classifier = RandomForestClassifier(n_estimators=300, random_state=44)
    rfe = RFE(classifier, n_features_to_select=5)
    X_train_rfe = rfe.fit_transform(X_train_clean, y_train_clean)
    X_test_final = rfe.transform(X_test_scaled)
    classifier.fit(X_train_rfe, y_train_clean)
    y_pred_final = classifier.predict(X_test_final)
    y_test_final = y_test
    le_model = le
    
    accuracy = accuracy_score(y_test_final, y_pred_final)
    precision = precision_score(y_test_final, y_pred_final, average='weighted')
    recall = recall_score(y_test_final, y_pred_final, average='weighted')
    f1 = f1_score(y_test_final, y_pred_final, average='weighted')
    
    onnx_session = None
    st.warning("⚠️ Using default model (trained on-the-fly)")

# Evaluation Metrics 
with st.expander("📊 EVALUATION METRICS"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ACCURACY", value=f'{accuracy:.4f}', delta='Accuracy Score')
    col2.metric("PRECISION", value=f'{precision:.4f}', delta='Precision Score With Weighted Average')
    col3.metric("RECALL", value=f'{recall:.4f}', delta='Recall Score With Weighted Average')
    col4.metric("F1 SCORE", value=f'{f1:.4f}', delta='F1 Score with Weighted Average')
    style_metric_cards(background_color='#FFFFFF', border_left_color='#9900AD', border_color='#1F66BD', box_shadow='#F71938')
    st.write(f"<b>NOTES</b>: Hasil evaluasi metriks yang diterapkan sangat baik dan sudah sesuai dengan hasil pelatihan model algoritma Random Forest.", unsafe_allow_html=True)

# Prediction Table 
with st.expander("📋 PREDICTION TABLE"):
    prediction_table = pd.DataFrame({
        'age': X_test_final[:, 0].ravel(), 
        'gender': X_test_final[:, 1].ravel(),
        'price': X_test_final[:, 2].ravel(),
        'payment_method': X_test_final[:, 3].ravel(),
        'shopping_mall': X_test_final[:, 4].ravel(),
        'Category | Actual Y': y_test_final.ravel(), 
        'Y_Predicted': y_pred_final.ravel(),
        'Accuracy': [accuracy] * len(y_test_final),
        'Precision': [precision] * len(y_test_final),
        'Recall': [recall] * len(y_test_final),
        'F1 Score': [f1] * len(y_test_final)
    })
    
    st.dataframe(prediction_table, use_container_width=True)
    st.write(f'<b>NOTES</b>: Pada bagian tabel prediksi ini menggunakan data yang telah diolah sebelumnya sehingga sangat berbeda dengan data asli.', unsafe_allow_html=True)

# Download Predicted Table in CSV
df_predict = prediction_table.to_csv(index=False).encode('utf-8')
st.download_button(label="📥 DOWNLOAD PREDICTED DATA",
                   data=df_predict,
                   key="download_predict.csv",
                   file_name='data_predict.csv') 

# Confusion Matrix and Feature Importance
with st.expander("🔍 CONFUSION MATRIX & FEATURE IMPORTANCE"):
    col1, col2 = st.columns(2)
    with col1:
        target_names = ['Books', 'Clothing', 'Cosmetics', 'Food & Beverage', 
                       'Shoes', 'Souvenir', 'Technology', 'Toys']
        cm = confusion_matrix(y_test_final, y_pred_final)
        plt.figure(figsize=(15, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix Customer Category Prediction')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        st.pyplot(fig=plt, use_container_width=True)

    with col2:
        # Feature Importance only available for sklearn models, not ONNX
        if classifier is not None:
            try:
                if hasattr(classifier, 'named_steps'):
                    actual_classifier = classifier.named_steps['classifier']
                    feature_importance = actual_classifier.feature_importances_
                elif hasattr(classifier, 'steps'):
                    actual_classifier = classifier.steps[-1][1]
                    feature_importance = actual_classifier.feature_importances_
                else:
                    feature_importance = classifier.feature_importances_
            except:
                feature_importance = classifier.feature_importances_
            
            feature_names = ['age', 'gender', 'price', 'payment_method', 'shopping_mall']
            importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
            importance_df = importance_df.sort_values("Importance", ascending=True)
            bar = px.bar(importance_df, x='Importance', y='Feature')
            bar.update_layout(title={'text': 'Feature Importance Model Random Forest', 
                                    'xanchor': 'center', 'yanchor': 'top', 
                                    'x': 0.5, 'y': 0.95})
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.info("📊 Feature importance is not available for ONNX models.\nPlease use Pickle model to view feature importance.")
    
    st.write(f'<b>NOTES</b>: Hasil feature importance menunjukkan data fitur Price lebih dominan dibandingkan fitur lainnya dan evaluasi dengan Confusion Matrix terlihat sudah sangat cukup baik dalam hal identifikasi tiap kategori.', unsafe_allow_html=True)

#------------ PREDICT NEW DATA ---------
with st.expander("🎯 PREDICT NEW DATA"):
    with st.form("input_form", clear_on_submit=True):
        x1 = st.number_input("Age", min_value=0, max_value=100)
        x2 = st.selectbox("Gender", ["Male", "Female"])
        x3 = st.number_input("Price", min_value=0.0, max_value=10000.0, step=0.1)
        x4 = st.selectbox("Payment Method", ["Cash", "Credit Card", "Debit Card"])
        x5 = st.selectbox("Shopping Mall", ["Mall of Istanbul", "Kanyon", 
                                           "Metrocity", "Metropol AVM", 
                                           "Istinye Park", "Zorlu Center", 
                                           "Cevahir AVM", "Forum Istanbul", 
                                           "Viaport Outlet", "Emaar Square Mall"])
        submitted = st.form_submit_button(label="🔮 PREDICT")

if submitted:
    new_data = pd.DataFrame({'age': [x1], 'gender': [x2], 'price': [x3], 
                            'payment_method': [x4], 'shopping_mall': [x5]})
    
    le_gender = LabelEncoder()
    le_payment_method = LabelEncoder()
    le_shopping_mall = LabelEncoder()
    
    # Fit with original data to ensure consistent encoding
    le_gender.fit(retail['gender'])
    le_payment_method.fit(retail['payment_method'])
    le_shopping_mall.fit(retail['shopping_mall'])
    
    new_data['gender'] = le_gender.transform(new_data['gender'])
    new_data['payment_method'] = le_payment_method.transform(new_data['payment_method'])
    new_data['shopping_mall'] = le_shopping_mall.transform(new_data['shopping_mall'])
    
    # Apply transformations
    new_data_scaled = scaler.transform(new_data)
    if rfe is not None:
        new_data_rfe = rfe.transform(new_data_scaled.reshape(1, -1))
    else:
        new_data_rfe = new_data_scaled.reshape(1, -1)
    
    # Make prediction based on model type
    if onnx_session is not None:
        # ONNX Prediction
        input_name = onnx_session.get_inputs()[0].name
        label_name = onnx_session.get_outputs()[0].name
        prob_name = onnx_session.get_outputs()[1].name
        
        pred_result = onnx_session.run([label_name, prob_name], 
                                       {input_name: new_data_rfe.astype(np.float32)})
        predict_category = pred_result[0]
        predict_proba = pred_result[1]
    else:
        # Sklearn Prediction
        if hasattr(classifier, 'named_steps') or hasattr(classifier, 'steps'):
            predict_category = classifier.predict(new_data)
            predict_proba = classifier.predict_proba(new_data)
        else:
            predict_category = classifier.predict(new_data_rfe)
            predict_proba = classifier.predict_proba(new_data_rfe)
    
    prediction = le_model.inverse_transform(predict_category)
    
    st.write(f"<span style='font-size:34px; color:green;'>Predicted Category: </span> <span style='font-size:34px;'>{prediction[0]}</span>", unsafe_allow_html=True)
    
    # Show probability
    st.write("### Prediction Probability:")
    target_names = ['Books', 'Clothing', 'Cosmetics', 'Food & Beverage', 
                   'Shoes', 'Souvenir', 'Technology', 'Toys']
    prob_df = pd.DataFrame({'Category': target_names, 'Probability': predict_proba[0]})
    prob_df = prob_df.sort_values('Probability', ascending=False)
    
    fig = px.bar(prob_df, x='Probability', y='Category', orientation='h',
                 title='Prediction Probability for Each Category')
    st.plotly_chart(fig, use_container_width=True)