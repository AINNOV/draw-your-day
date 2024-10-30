import streamlit as st
import requests
from PIL import Image
import io

st.title("Diary to Image Generator")

# 일기 입력
diary_text = st.text_area("Enter your diary entry:", "")
if st.button("Generate Image"):
    # FastAPI 서버에 POST 요청
    response = requests.post("http://localhost:8000/generate", json={"text": diary_text})
    
    if response.status_code == 200:
        data = response.json()
        # st.write(f"Generated Prompt: {data['prompt']}")
        
        # 생성된 이미지 출력
        for image_path in data["images"]:
            image = Image.open(image_path)
            st.image(image, caption="")
    else:
        st.write("Failed to generate images.")
