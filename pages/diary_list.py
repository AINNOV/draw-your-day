import streamlit as st
import requests
from cookie_manager import cookies

BASE_URL = "http://localhost:8000"

def show_diary_detail(entry):
    st.subheader(entry["title"])
    st.image(entry.get("generated_image_path", None), width=300)
    st.write(f"Date: {entry['date']}")
    st.write(entry["content"])

    if st.button("Back to list"):
        st.session_state.selected_entry = None
        st.rerun()

    if st.button(f"Delete {entry['id']}", key=f'delete{entry["id"]}'):
        jwt_token = st.session_state.get("jwt_token")
        headers = {"Authorization": f"Bearer {jwt_token}"}
        response = requests.delete(f"{BASE_URL}/diary/{entry['id']}", headers=headers)
        if response.status_code == 200:
            st.success("Diary entry deleted successfully!")
            st.session_state.selected_entry = None
            st.rerun()
        else:
            st.error("Failed to delete diary entry.")

if "logged_in" not in st.session_state:
    if cookies.get("logged_in") is None:
        st.session_state.logged_in = False
    else:
        st.session_state.logged_in = cookies.get("logged_in") == "True"
        st.session_state.jwt_token = cookies.get("jwt_token")


if "jwt_token" in st.session_state:
    st.header("My Diary Entries")
    jwt_token = st.session_state.get("jwt_token")
    headers = {"Authorization": f"Bearer {jwt_token}"}

    response = requests.get(f"{BASE_URL}/diaries", headers=headers)

    if response.status_code == 200:
        entries = response.json()
        if entries:
            if "selected_entry" not in st.session_state:
                st.session_state.selected_entry = None
                
            # 선택된 일기가 없을 때만 목록 표시
            if st.session_state.selected_entry is None:
                for entry in entries:
                    st.subheader(entry["title"])
                    st.write(f"Date: {entry['date']}")
                    st.write(entry["content"])

                    # 선택한 일기를 표시하는 버튼
                    if st.button(f"View {entry['id']}", key=f"view_{entry['id']}"):
                        st.session_state.selected_entry = entry
                        st.rerun()
            else:
                # 상세 페이지 표시
                show_diary_detail(st.session_state.selected_entry)

        else:
            st.write("No diary entries found.")

else:
    st.warning("로그인이 필요합니다.")