import streamlit as st
from time import sleep
from pages.cookie_manager import cookies

for key in st.session_state.keys():
    del st.session_state[key]

cookies.delete("logged_in")
st.success("Bye Bye")

sleep(0.1)
st.switch_page("login.py")
