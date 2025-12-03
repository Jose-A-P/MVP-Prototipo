import streamlit as st
from api.llm_gen import generate_chat_response

st.title("Chatbot – ¿En qué te puedo ayudar?")

user_input = st.chat_input("Haz una pregunta o ejecuta un escenario...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        response = generate_chat_response(user_input)
        st.write(response)
