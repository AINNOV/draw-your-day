import streamlit as st
import requests
from datetime import date
from time import sleep
from pages.cookie_manager import cookies

BASE_URL = "http://localhost:8000"

if "logged_in" not in st.session_state:
    if cookies.get("logged_in") is None:
        st.session_state.logged_in = False
    else:
        st.session_state.logged_in = cookies.get("logged_in") == "True"
        st.session_state.jwt_token = cookies.get("jwt_token")

if "selected_entry" in st.session_state:
    st.session_state.selected_entry = None

if "jwt_token" in st.session_state:
    st.markdown(
        f"""
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; height: 150px; text-align: center; display: flex; flex-direction: column; justify-content: center;">
            <p style="font-size: 64px; line-height: 1.6; color: #333;">{"Write your diary!"}</p>
        </div>
        """,
        unsafe_allow_html = True,
    )
    title, date_input = st.columns([0.8, 0.2])
    title = st.text_input("Title", key="title")
    date_input = st.date_input("Date", value=date.today())
    content = st.text_area("Diary Content", key="content")

    if st.button("Draw It"):
        jwt_token = st.session_state.get("jwt_token")
        headers = {"Authorization": f"Bearer {jwt_token}"} if jwt_token else {}
        data = {"title": title, "date": date_input.isoformat(), "content": content}
        response = requests.post(f"{BASE_URL}/diary", json=data, headers=headers)
        if response.status_code == 200:
            st.success("Diary entry created!")
            sleep(0.1)
            if "selected_entry" in st.session_state and st.session_state.selected_entry is not None:
                st.session_state.selected_entry = None
            st.switch_page("pages/diary_list.py")
        else:
            st.error(f"Failed to create diary entry: {response.status_code}")
            st.write("Response content:", response.text)
else:
    st.warning("로그인이 필요합니다.")