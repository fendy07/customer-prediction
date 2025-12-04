import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Dashboard Analysis Customer Retail")

# Load CSS style
with open('static/styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('data/customer_shopping_data.csv')
    return data
data = load_data()

with st.expander("HASIL DATA"):
    data = pd.DataFrame({
        'InvoiceNo': data['invoice_no'],
        'CustomerID': data['customer_id'],
        'Gender': data['gender'],
        'Age': data['age'],
        'Category': data['category'],
        'Quantity': data['quantity'],
        'Price': data['price'],
        'PaymentMethod': data['payment_method'],
        'InvoiceDate': data['invoice_date'],
        'ShoppingMall': data['shopping_mall']
    })
    st.dataframe(data, use_container_width=True)

# Download Dataset
download = data.to_csv(index=False).encode('utf-8')
st.download_button(label = "DOWNLOAD DATASET",
                   data = download, 
                   key = 'download_data.csv', 
                   file_name = 'dataset_retail.csv')

# Visualization 
with st.expander("DISTRIBUSI KATEGORI DAN PEMBAYARAN"):
    col1, col2 = st.columns(2)
    with col1:
        data_quantity = data.groupby('Category')['Quantity'].sum()
        # Plot Pie Chart
        plt.figure(figsize = (10, 8))
        plt.pie(data_quantity.values, labels = data_quantity.index, 
                autopct = '%1.1f%%', colors = sns.color_palette("pastel"))
        # Title
        plt.title('Kuantitas Produk Berdasarkan Kategori', fontsize = 16)
        st.pyplot(plt)
    
    with col2:
        payment_counts = data['PaymentMethod'].value_counts()
        fig = px.bar(x = payment_counts.index, y = payment_counts.values, 
             labels = {'x': 'Metode Pembayaran', 'y': 'Jumlah Transaksi'}, 
             color = payment_counts.index)
        fig.update_layout(font_size = 14)
        title = fig.update_layout(title = {'text': 'Distribusi Metode Pembayaran', 
                                           'xanchor': 'center', 
                                           'yanchor': 'top', 
                                           'x': 0.5,
                                           'y': 0.95})
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"<b>NOTES</b>: Distribusi dalam kategori berdasarkan kuantitas kategori produk yang sering dibeli oleh pelanggan adalah baju, kosmetik dan F&B. Sedangkan, metode pembayaran dengan transaksi terbanyak adalah Cash dan Credit.", unsafe_allow_html=True)

with st.expander("TOTAL PENDAPATAN DAN PENJUALAN"):
    col1, col2 = st.columns(2)
    with col1:
        total_revenue = data.groupby('ShoppingMall')['Price'].sum()
        fig = px.bar(x = total_revenue.index, y = total_revenue.values, 
                     labels = {'x': 'Mall', 'y': 'Total Pendapatan'},
                     color = total_revenue.index)
        title = fig.update_layout(title = {'text': 'Total Pendapatan Setiap Pusat Perbelanjaan', 
                                           'xanchor': 'center', 
                                           'yanchor': 'top', 
                                           'x': 0.5, 
                                           'y': 0.95})
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        total_sales = data.groupby('ShoppingMall')['Quantity'].sum().sort_values(ascending=False)
        fig = px.bar(x = total_sales.index, y = total_sales.values,
                     labels = {'x': 'Mall', 'y': 'Total Penjualan'},
                     color = total_sales.index)
        title = fig.update_layout(title = {'text': 'Total Penjualan Setiap Pusat Perbelanjaan', 
                                           'xanchor': 'center',
                                           'yanchor': 'top',
                                           'x': 0.5,
                                           'y': 0.95})
        
        st.plotly_chart(fig, use_container_width=True)

