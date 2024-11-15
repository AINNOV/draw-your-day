import streamlit as st
from pages.cookie_manager import cookies

def main_page():
    st.title("Draw Your Day")

# 로그인 상태 초기화 및 불러오기
if "logged_in" not in st.session_state:
    if cookies.get("logged_in") is None:
        st.session_state.logged_in = False
    else:
        st.session_state.logged_in = cookies.get("logged_in") == "True"
        st.session_state.jwt_token = cookies.get("jwt_token")

if st.session_state.logged_in:
    pages = {
        "Menu": [
            st.Page("pages/diary_list.py", title="List"),
            st.Page("pages/diary_new.py", title="New"),
        ],
        "Account": [
            st.Page("pages/logout.py", title="Logout"),
        ],
    }
else:
    pages = {
        "Menu":[
            st.Page(main_page, title="Main")
        ],
        "Account": [
            st.Page("pages/login.py", title="Login/Register"),
        ]
    }

pg = st.navigation(pages)

pg.run()