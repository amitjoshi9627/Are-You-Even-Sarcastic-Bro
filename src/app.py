import streamlit as st
from sarcasm_detector import get_result

st.title("Sarcasm Detector")
text = st.text_input("Enter Your Text..")

if text is not None:
    if st.button("Check Result"):
        result = get_result(text)
        st.write("## __{}__".format(result))
