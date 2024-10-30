import streamlit as st
import requests
from PIL import Image
import io
from datetime import date

# 기본 화면 구성
st.set_page_config(page_title="Draw Your Day", layout="centered")

# 페이지 상태 초기화 (최초 접근 시 메인 화면을 보여주기 위함)
if "page" not in st.session_state:
    st.session_state.page = "Main"

# 일기 저장 공간 초기화
if "diary_entries" not in st.session_state:
    st.session_state.diary_entries = []

# 메인 제목
if st.session_state.page == "Main":
    st.markdown("<h1 style='text-align: center; color: #4B86B4;'>Draw Your Day</h1>", unsafe_allow_html=True)

# 사이드바 메뉴
st.sidebar.title("Menu")
if st.sidebar.button("List"):
    st.session_state.page = "List"
if st.sidebar.button("New"):
    st.session_state.page = "New"

# 페이지 전환
if st.session_state.page == "List":
    st.write("This is the List page.")

    # 저장된 일기 목록 표시
    if st.session_state.diary_entries:
        for entry in st.session_state.diary_entries:
            st.write(f"### {entry['title']} ({entry['date']})")
            st.image(entry["result"])
            st.write("---")
    else:
        st.write("No diary entries yet.")

elif st.session_state.page == "New":
    st.write("New Diary Entry")  # 화면 제목

    # 입력 필드들
    diary_title = st.text_input("Diary Title")
    diary_date = st.date_input("Date", date.today())
    diary_content = st.text_area("Diary Content")
    uploaded_image = st.file_uploader("Attach a Photo", type=["jpg", "jpeg", "png"])

    # Draw it! 버튼
    if st.button("Draw it!"):
        # FastAPI 서버에 POST 요청
        response = requests.post("http://localhost:8000/generate", json={"text": diary_content})
        
        if response.status_code == 200:
            data = response.json()
            # st.write(f"Generated Prompt: {data['prompt']}")
            
            # 생성된 이미지 출력
            for image_path in data["images"]:
                image = Image.open(image_path)
                st.image(image, caption="")
        else:
            st.write("Failed to generate images.")
        
        # 입력된 정보를 session_state에 추가
        st.session_state.diary_entries.insert(0, {
            "title": diary_title,
            "date": diary_date,
            "content": diary_content,
            "image": uploaded_image,
            "result": image
        })
        
        # List 페이지로 이동
        st.session_state.page = "List"
        st.rerun()  # 페이지를 새로고침하여 List 페이지를 보여줌
