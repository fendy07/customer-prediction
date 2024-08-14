import streamlit as st

st.set_page_config(page_title = 'Customer Prediction', page_icon = '📈', layout = 'wide')

# --- PAGE SETUP -----
info_page = st.Page(
    page = 'pages/about.py',
    title = 'About Project',
    icon = ':material/person:',
    default= True
)
#---- PAGE PROJECT ------
dashboard = st.Page(
    page = 'pages/dashboard.py',
    title = 'Dashboard Customer Retail',
    icon = ':material/bar_chart:',
)
#---- PAGE PREDICTION ------
prediction = st.Page(
    page = 'pages/prediction.py',
    title = 'Customer Category Prediction',
    icon = ':material/thumb_up:'
)

page = st.navigation(
    {
        "Info": [info_page],
        "Projects": [dashboard, prediction],
    }
)

st.sidebar.info("Source code, find in My Github:")
st.sidebar.link_button("Github Source", "")
st.sidebar.text(f'Created by Fendy Hendriyanto 👨🏼‍💻')

page.run()