import sys
sys.path.append("./pages")

import streamlit as st
from time import sleep
from pages.cookie_manager import cookies
import requests
from urllib.parse import urljoin

BASE_URL = "http://localhost:8000"

user_info = st.query_params.get_all('user_info') 

if user_info:
    response = requests.post(f"{BASE_URL}/login", json={"username": user_info[0], "password": "google"})
    if response.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.jwt_token = response.json().get("access_token")
        cookies.batch_set({"logged_in" : "True", "jwt_token" : st.session_state.jwt_token})
        st.success("로그인 성공!")
        sleep(0.1)
        st.switch_page("diary_list.py")
    else:
        response = requests.post(f"{BASE_URL}/register", json={"username": user_info[0], "password": "google"})
        if response.status_code == 200:
            response = requests.post(f"{BASE_URL}/login", json={"username": user_info[0], "password": "google"})
            if response.status_code == 200:
                st.session_state.logged_in = True
                st.session_state.jwt_token = response.json().get("access_token")
                cookies.batch_set({"logged_in" : "True", "jwt_token" : st.session_state.jwt_token})
                st.success("로그인 성공!")
                sleep(0.1)
                st.switch_page("diary_list.py")

st.title("Login / Register")
username = st.text_input("Username", key="username_input")
password = st.text_input("Password", type="password", key="userpassword_input")

login_button_html = f'<a href="http://1.229.207.166:8000/login_google" target="_blank">Login with Google</a>'
st.markdown(login_button_html, unsafe_allow_html=True)

if st.button("Login", key="login_button"):
    response = requests.post(f"{BASE_URL}/login", json={"username": username, "password": password})
    if response.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.jwt_token = response.json().get("access_token")
        cookies.batch_set({"logged_in" : "True", "jwt_token" : st.session_state.jwt_token})
        st.success("로그인 성공!")
        sleep(0.1)
        st.switch_page("diary_list.py")
    elif response.status_code == 404:
        st.error("존재하지 않는 아이디입니다.")
    elif response.status_code == 401:
        st.error("비밀번호가 틀립니다.")

if st.button("Register", key="register_button"):
    response = requests.post(f"{BASE_URL}/register", json={"username": username, "password": password})
    if response.status_code == 200:
        st.success("회원가입 성공! 로그인하세요.")
        st.rerun()
    else:
        st.error("회원가입 실패. 다시 시도하세요.")
            