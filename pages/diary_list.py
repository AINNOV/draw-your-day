import streamlit as st
import requests
from pages.cookie_manager import cookies

st.set_page_config(layout="wide")

BASE_URL = "http://localhost:8000"

if "logged_in" not in st.session_state:
    if cookies.get("logged_in") is None:
        st.session_state.logged_in = False
    else:
        st.session_state.logged_in = cookies.get("logged_in") == "True"
        st.session_state.jwt_token = cookies.get("jwt_token")


st.markdown("""
<style>
.streamlit-expanderHeader {
    pointer-events: none;
}
[data-testid="stHeaderActionElements"] {
    visibility: hidden;
}
[data-testid="stExpanderToggleIcon"] {
    visibility: hidden;
}
[data-testid="StyledFullScreenButton"] {
    visibility: hidden;
}
[data-testid="stHeaderActionElements"] {
    visibility: hidden;
}
.stColumn.st-emotion-cache-fplge5.e1f1d6gn3 div {
    display: flex;
    justify-content: center;
}

</style>
""", unsafe_allow_html=True)

def show_diary_entries(entries):
    entries = sorted(entries, key = lambda x: x['date'], reverse = True)

    for entry in entries:
        with st.expander(label = "", expanded = True):
            col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
            col1.image(entry.get("generated_image_path", None), width = 300)
            col2.subheader(entry["title"])
            col3.write(f"Date: {entry['date']}")

            # 선택한 일기를 표시하는 버튼
            if col3.button(f"View", key=f"view_{entry['id']}"):
                st.session_state.selected_entry = entry
                st.rerun()
    
def show_diary_detail(entry):
    back, empty, delete = st.columns([0.2, 0.6, 0.2])
    if back.button("Back"):
        st.session_state.selected_entry = None
        st.rerun()
    if delete.button(f"Delete", key=f'delete{entry["id"]}'):
        jwt_token = st.session_state.get("jwt_token")
        headers = {"Authorization": f"Bearer {jwt_token}"}
        response = requests.delete(f"{BASE_URL}/diary/{entry['id']}", headers=headers)
        if response.status_code == 200:
            st.success("Diary entry deleted successfully!")
            st.session_state.selected_entry = None
            st.rerun()
        else:
            st.error("Failed to delete diary entry.") 
            
    col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
    col1.markdown(f"Title: {entry['title']}")
    col2.image(entry.get("generated_image_path", None), width=300)
    col3.markdown(f"Date: {entry['date']}")
    
    st.markdown(entry["content"])


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
                show_diary_entries(entries)
                
            else:
                # 상세 페이지 표시
                show_diary_detail(st.session_state.selected_entry)

        else:
            st.write("No diary entries found.")

else:
    st.warning("로그인이 필요합니다.")