import streamlit as st

st.title('Author Project')

col1, col2 = st.columns(2, gap='small', vertical_alignment='center')
with col1:
    st.image("images/Fendy.png", width=250)
with col2:
    st.title("Fendy Hendriyanto", anchor=False)
    st.write(
        "AI Engineer and Instructor"
    )
    st.write(
        "Assisting and mentoring students to help analyze and supporting data driven with creativity and decision making."
    )

#--- EXPERIENCE & QUALIFICATIONS ------
st.write("\n")
st.subheader("Experience and Qualifications", anchor=False)
st.write(
    """
    - 2 years experience coaching and mentoring about Artificial Intelligence
    - Strong hands-on experience and knowledge in Python and Data Science
    - Proficient in using various libraries and tools such as TensorFlow, Keras, Scikit-learn, OpenCV, Pandas
    - Good understanding and analyzing of statistical principles and their perspective applications
    - Excellent team player and initiative on tasks

    """
)

# ---- SKILLS ----
st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write(
    """
    - Programming : Python (Pandas, Scikit-learn, Scikit-image), R, SQL, JavaScript
    - Data Visualization : Tableau, Spreadsheet, Excel
    - Modelling : Tensorflow, Keras, PyCaret, XGBoost, CometML
    - Databases : MySQL, PostgreSQL, SQLite
    - Deployment : Streamlit, Flask, Gradio, Huggingface, Git
    - Frameworks : OpenCV, NLTK

    """
)