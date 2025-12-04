import onnx
import pickle
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
import missingno as msno
import plotly.express as px
from scipy.stats import zscore
import matplotlib.pyplot as plt
from skl2onnx import convert_sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from skl2onnx.common.data_types import FloatTensorType
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Load Dataset
data_ritel = pd.read_csv('data/customer_shopping_data.csv')
data_ritel.sample(25)

data_ritel.info()

data_ritel.shape

"""## Exploratory Data Analysis"""

data_ritel.isnull().sum()

# Visualisasi pada kolom data apabila terjadi Missing Value
msno.matrix(data_ritel, sparkline=False)

category = data_ritel['category'].value_counts()
print(category)

gender = data_ritel['gender'].value_counts()
print(gender)

payment_counts = data_ritel['payment_method'].value_counts()
print(payment_counts)

# Shopping Mall Insights berdasarkan jumlah transaksional
mall = data_ritel['shopping_mall'].value_counts()
print(mall)

# Visualize data on payment method and age
# Set the style for the plot
sns.set_style("whitegrid")
# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))
# Plotting the bar chart
sns.barplot(x=payment_counts.index, y=payment_counts.values, palette="viridis", ax=ax)
# Set plot title and labels
ax.set_title("Distribution of Payment Methods", fontsize=16)
ax.set_xlabel("Payment Method", fontsize=14)
ax.set_ylabel("Number of Transactions", fontsize=14)
# Set x-axis tick labels
ax.set_xticks(payment_counts.index)
ax.set_xticklabels(payment_counts.index.unique(), fontsize=12)
# Display the plot
plt.show()

# Calculate the average prices for each product category
average_prices = data_ritel.groupby('category')['price'].mean()
average_prices

# Visualisasi data pada kolom Category dan Harga
# Create a bar chart for the average prices of each product category
fig = px.bar(average_prices,
             x=average_prices.index,
             y=average_prices.values,
             labels={'x': 'Kategori Produk', 'y': 'Rata-rata Harga'},
             title='Rata-rata harga dalam Kategori Produk')

# Show the plot
fig.show()

# Mengelompokkan data berdasarkan kategori dan menjumlahkan quantity
category_quantity = data_ritel.groupby('category')['quantity'].sum()
# Plot pie chart
plt.figure(figsize=(10, 8))
plt.pie(category_quantity, labels=category_quantity.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
# Set judul
plt.title('Distribusi Kategori Produk Berdasarkan Jumlah Quantity', fontsize=16)
# Tampilkan plot
plt.show()

# Visualisasi total pendapatan disetiap pusat perbelanjaan
total_revenue = data_ritel.groupby('shopping_mall')['price'].sum()
fig = px.bar(total_revenue,
             x = total_revenue.index,
             y = total_revenue.values,
             labels = {'x': 'Shopping Mall', 'y': 'Total Revenue'},
             title = 'Total Pendapatan Setiap Pusat Perbelanjaan')

# Show the plot
fig.show()

# Top penjualan pada kategori
category_quantity = data_ritel.groupby('category')['quantity'].sum().sort_values(ascending=False)
# Create a bar chart for the top-selling product categories
fig = px.bar(category_quantity,
             x = category_quantity.index,
             y = category_quantity.values,
             labels = {'x': 'Kategori Produk', 'y': 'Total Kuantitas Terjual'},
             title = 'Top Penjualan Kuantitas Kategori')

# Show the plot
fig.show()

# Visualisasi data pada kolom umur
# Plot bar chart untuk distribusi umur
plt.figure(figsize=(10, 6))
sns.histplot(data_ritel['age'], bins=20, kde=False, color='skyblue')

# Set judul dan label
plt.title('Distribusi Umur Pelanggan', fontsize=16)
plt.xlabel('Umur', fontsize=14)
plt.ylabel('Jumlah Pelanggan', fontsize=14)
# Tampilkan plot
plt.show()
# Demografi Pelanggan berdasarkan jenis kelamin dan umur
demographics_summary = data_ritel[['gender', 'age']].describe(include='all')
demographics_summary
# Visualisasi hasil transaksi terbanyak pada pusat perbelanjaan atau mall
fig = px.bar(data_ritel['shopping_mall'].value_counts(),
             x = data_ritel['shopping_mall'].value_counts().index,
             y = data_ritel['shopping_mall'].value_counts().values,
             labels = {'x': 'Shopping Mall', 'y': 'Nominal Transaksi'},
             title = 'Jumlah Nominal Transaksi Pada Pusat Perbelanjaan')
fig.show()


### **Data Preprocessing
# Encoding data kolom dengan feature mapping
# Encoding pada kolom metode pembayaran
data_ritel['payment_method'] = data_ritel['payment_method'].map({'Cash': 0, 'Credit Card': 1, 'Debit Card': 2})
# Encoding pada kolom jenis kelamin
data_ritel['gender'] = data_ritel['gender'].map({'Female': 0, 'Male': 1})
# Encoding pada kolom pusat perbelanjaan
data_ritel['shopping_mall'] = data_ritel['shopping_mall'].map({'Mall of Istanbul': 0,
                                                               'Kanyon': 1,
                                                               'Metrocity': 2,
                                                               'Metropol AVM': 3,
                                                               'Istinye Park': 4,
                                                               'Zorlu Center': 5,
                                                               'Cevahir AVM': 6,
                                                               'Forum Istanbul': 7,
                                                               'Viaport Outlet': 8,
                                                               'Emaar Square Mall': 9})

# Encoding data kolom kategori
le = LabelEncoder()
data_ritel['category'] =le.fit_transform(data_ritel['category'])
data_ritel.sample(10)
# Analisa statistik deskriptif
data_ritel.describe().T
# Hapus kolom data yang tidak diperlukan
data_ritel = data_ritel.drop(columns = ['invoice_no', 'customer_id', 'invoice_date'])
data_ritel.sample(10)
# Korelasi antara kolom data
plt.figure(figsize = (12, 8))
sns.heatmap(data_ritel.corr(), annot = True)
plt.show()

# Pemilihan data fitur dan label
features = ['age', 'gender', 'price', 'payment_method', 'shopping_mall']

X = data_ritel[features].values
y = data_ritel['category'].values

"""### **Splitting Data**"""

# Pisahkan data Train dengan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 44)
print('Data training : ', X_train.shape, y_train.shape)
print('Data Testing : ', X_test.shape, y_test.shape)

data_ritel.category.value_counts()

# Filling Missing Data
imputer = SimpleImputer(strategy = 'mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Outlier detection using Z-Score
z_scores = np.abs(zscore(X_train_scaled))
threshold = 5
outliers = np.where(z_scores > threshold)

X_train_no_outliers = X_train_scaled[(z_scores < threshold).all(axis=1)]
y_train_no_outliers = y_train[(z_scores < threshold).all(axis=1)]

# Modelling
# Decision Tree Classifier
model_dt = DecisionTreeClassifier(random_state = 44)
rfe = RFE(model_dt, n_features_to_select=5)

X_train_rfe = rfe.fit_transform(X_train_no_outliers, y_train_no_outliers)
X_test_rfe = rfe.transform(X_test_scaled)

selected_features = np.array(features)[rfe.support_]
print(selected_features)

"""### **Decision Tree**"""

# Training with Pipeline
pipeline = Pipeline([
    ('Classifier', DecisionTreeClassifier(random_state = 44)),

])

param_grid = {
    'Classifier__max_depth': list(range(2, 10)),
    'Classifier__max_leaf_nodes': list(range(2, 10))
}

gridsearch = GridSearchCV(pipeline, param_grid, cv=20)
gridsearch.fit(X_train_rfe, y_train_no_outliers)

best_model = gridsearch.best_estimator_
print(gridsearch.best_params_)

# Evaluation Model Decision Tree
# Predict the model
y_pred = best_model.predict(X_test_rfe)
print("Accuracy:", accuracy_score(y_test, y_pred))

scores = cross_val_score(best_model, X_train_no_outliers, y_train_no_outliers, cv=20, scoring='accuracy')
print("Cross-Validation Accuracy Scores:", scores)

# Classification Report
target_names = ['Books',
                'Clothing',
                'Cosmetics',
                'Food & Beverage',
                'Shoes',
                'Souvenir',
                'Technology',
                'Toys']

print('Classification Report in Hyperparameter Tuning Decision Tree:')
print(classification_report(y_test, y_pred, target_names = le.classes_))

# Plot Decision Tree
# Assuming 'best_model' is your Pipeline object
decision_tree = best_model.named_steps['Classifier']  # Replace 'Classifier' with your step name
plt.figure(figsize = (25, 20))
tree.plot_tree(decision_tree,
               feature_names = features,
               class_names = target_names,
               filled=True)

plt.show()

# Random Forest
model_rf = RandomForestClassifier(random_state = 44)
rfe = RFE(model_rf, n_features_to_select = 5)

X_train_rfe = rfe.fit_transform(X_train_no_outliers, y_train_no_outliers)
X_test_rfe = rfe.transform(X_test_scaled)

selected_features = np.array(features)[rfe.support_]
print(selected_features)

# Training with Pipeline
pipeline = Pipeline([
    ('Classifier', RandomForestClassifier(random_state = 44)),
])

param_grid = {
    'Classifier__n_estimators': [100, 200, 300],
    'Classifier__max_depth': [None, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv = 5)
grid_search.fit(X_train_rfe, y_train_no_outliers)

best_model_rf = grid_search.best_estimator_
print(grid_search.best_params_)

## **Evaluation Model Random Forest
y_pred_rf = best_model_rf.predict(X_test_rfe)
print("Accuracy Score :", accuracy_score(y_test, y_pred_rf))
scores = cross_val_score(best_model_rf, X_train_no_outliers, y_train_no_outliers, cv = 5, scoring = 'accuracy')
print("Cross-Validation Accuracy Scores: ", scores)
print('Classification Report in Hyperparameter Tuning Random Forest:')
print(classification_report(y_test, y_pred_rf, target_names = target_names))

# Assuming 'y_test' and 'y_pred_rf' are your true and predicted labels respectively
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = target_names)
disp.plot(cmap = 'Blues', xticks_rotation = 'vertical')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Assuming 'best_model_rf' is your best performing model (Random Forest in this case)
# Input features for new data (replace with actual values)
new_data = np.array([[30, 0, 50, 1, 0]])  # Example: age, gender, price, payment_method, shopping_mall
# Preprocess the new data (scaling)
new_data_scaled = scaler.transform(new_data)
# Feature selection using RFE (if used during training)
new_data_rfe = rfe.transform(new_data_scaled)
# Make prediction
predicted_category = best_model.predict(new_data_rfe)
# Decode the predicted category (if label encoding was used)
predicted_category_name = le.inverse_transform(predicted_category)
print("Predicted Category (Numerical):", predicted_category)
print("Predicted Category (Name):", predicted_category_name)

# Save model using Pickle
with open('model/best_model_rf.pkl', 'wb') as file:
  pickle.dump(best_model_rf, file)

# Save model using Joblib
joblib.dump(best_model_rf, 'model/best_model_rf.joblib')

# Save model using ONNX
initial_type = [('float_input', FloatTensorType([None, X_train_rfe.shape[1]]))]
# Convert the scikit-learn Random Forest model to ONNX format
onnx_model = convert_sklearn(best_model_rf, initial_types=initial_type)
# Define the path to save the ONNX model
onnx_filename = 'model/best_model_rf.onnx'
# Save the ONNX model to a file
with open(onnx_filename, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"Model saved to {onnx_filename} in ONNX format.")

# Optional: Verify the ONNX model
onnx_model_loaded = onnx.load(onnx_filename)
onnx.checker.check_model(onnx_model_loaded)
print("ONNX model check successful!")

# Load Model Pickle
filename = 'model/best_model_rf.pkl'
model = pickle.load(open(filename, 'rb'))
model.score(X_test_rfe, y_test)
