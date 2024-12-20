import streamlit as st
import requests
from time import sleep
from PIL import Image
import base64
from io import BytesIO
import streamlit.components.v1 as components
from pages.cookie_manager import cookies

BASE_URL = "http://localhost:8000"

if "logged_in" not in st.session_state:
    if cookies.get("logged_in") is None:
        st.session_state.logged_in = False
    else:
        st.session_state.logged_in = cookies.get("logged_in") == "True"
        st.session_state.jwt_token = cookies.get("jwt_token")


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

        
def show_diary_entries(entries):
    # html_code = f"""
    # <style>
    #     body {{
    #         font-family: Arial, sans-serif;
    #         background-color: #f9f9f9;
    #         margin: 0;
    #         padding: 0;
    #     }}
    #     .container {{
    #         max-width: 800px;
    #         margin: 20px auto;
    #         background: #fff;
    #         border-radius: 10px;
    #         box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    #         padding: 20px;
    #     }}
    #     .header {{
    #         text-align: center;
    #         font-size: 2.5em;
    #         color: #6a4c93;
    #         margin-bottom: 20px;
    #     }}
    #     .entry-list {{
    #         display: flex;
    #         flex-direction: column;
    #         gap: 20px;
    #     }}
    #     .entry {{
    #         display: flex;
    #         gap: 20px;
    #         align-items: center;
    #         background: #f9f9f9;
    #         padding: 10px;
    #         border-radius: 10px;
    #         box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    #         transition: transform 0.2s ease;
    #         cursor: pointer;
    #     }}
    #     .entry:hover {{
    #         transform: scale(1.02);
    #     }}
    #     .entry img {{
    #         width: 100px;
    #         height: 100px;
    #         border-radius: 10px;
    #         object-fit: cover;
    #         flex-shrink: 0;
    #     }}
    #     .entry-content {{
    #         flex-grow: 1;
    #     }}
    #     .entry-title {{
    #         font-size: 1.2em;
    #         font-weight: bold;
    #         color: #333;
    #         margin-bottom: 5px;
    #     }}
    #     .entry-date {{
    #         font-size: 0.9em;
    #         color: #888;
    #     }}
    # </style>
    # <script>
    #     function sendMessageToParent(id) {{
    #         const message = {{ type: 'query_param', id: id }};
    #         window.parent.postMessage(message, '*');
    #     }}
    #     window.addEventListener('message', function(event) {{
    #         const data = event.data;
    #         if (data.type === 'query_param') {{
    #             const newUrl = new URL(window.location.href);
    #             newUrl.searchParams.set('id', data.id);
    #             window.location.href = newUrl;
    #         }}
    #     }});
    # </script>
    # </head>
    # <body>
    #     <div class="container">
    #         <div class="header">Your Days</div>
    #         <div class="entry-list">   
    # """
    # entries = sorted(entries, key = lambda x: x['date'], reverse = True)
    # for entry in entries:
    #     with open(entry.get("generated_image_path", None), "rb") as img_file:
    #         encoded_image = base64.b64encode(img_file.read()).decode()
    #     entry_code = """
    #     <div class="entry" onclick="sendMessageToParent({})">
    #         <img src="data:image/png;base64,{}" alt="Diary Image"/>
    #         <div class="entry-content">
    #             <div class="entry-title">{}</div>
    #             <div class="entry-date">{}</div>
    #         </div>
    #     </div>
    #     """.format(entry['id'], encoded_image, entry['title'], entry['date'])
    #     html_code += entry_code
        
    # html_code += """
    #     </div>
    #     </div>
    # </body>
    # </html>
    # """

    # st.components.v1.html(html_code, height=1000, scrolling = True)
    
    st.title("Your Days")
    entries = sorted(entries, key=lambda x: x['date'], reverse=True)

    for entry in entries:
        container = st.container(border = True)
        with container:
            col1, col2, col3 = st.columns([1, 5, 1])
            with col1:
                with open(entry.get("generated_image_path", None), "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode()
                st.image(f"data:image/png;base64,{encoded_image}", width=100)

            with col2:
                st.subheader(entry['title'])
                st.caption(entry['date'])

            with col3:
                if st.button("View", key=f"view_{entry['id']}"):
                    st.session_state.selected_entry = entry
                    st.rerun()

    
    
def show_diary_detail(entry):
    back_button = st.button("<- Back")
    if back_button:
        st.session_state.selected_entry = None
        st.rerun()
        
    with open(entry.get("generated_image_path", None), "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode()
    html_code = f"""
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your Days</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f9f9f9;
            }}
            .container {{
                max-width: 1500px;
                margin: 20px auto;
                background: #fff;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                position: relative;
            }}
            .header {{
                text-align: center;
                font-size: 3em;
                color: #6a4c93;
                margin-bottom: 20px;
            }}
            .entry {{
                gap: 20px;
                align-items: flex-start;
            }}
            .image-section {{
            }}
            .image-section img {{
                width: 100%;
                height: auto;
                border-radius: 10px;
                object-fit: cover;
            }}
            .details {{
                display: flex;
                flex-direction: column;
                justify-content: center;
            }}
            .date {{
                font-size: 1.2em;
                color: #555;
                margin-bottom: 10px;
            }}
            .title {{
                font-size: 2em;
                color: #6a4c93;
                margin-bottom: 10px;
                line-height: 1.4;
            }}
            .content {{
                margin-top: 40px;
                font-size: 1.2em;
                line-height: 1.6;
                color: #333;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">{entry["title"]}</div>
            <div class="date">{entry["date"]}</div>
            <div class="entry">
                <div class="image-section">
                    <img src="data:image/png;base64,{encoded_image}" alt="Diary Image">
                </div>
            </div>
            <div class="content">
                {entry["content"]}
            </div>
        </div>
    </body>
    </html>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    
    delete_button = st.button("Delete")
    if delete_button:
        jwt_token = st.session_state.get("jwt_token")
        headers = {"Authorization": f"Bearer {jwt_token}"}
        response = requests.delete(f"{BASE_URL}/diary/{entry['id']}", headers=headers)
        if response.status_code == 200:
            st.success("Diary entry deleted successfully!")
            st.session_state.selected_entry = None
            st.rerun()  
        else:
            st.error("Failed to delete diary entry.")
    
    regenerate_button = st.button("Regenerate")
    if regenerate_button:
        jwt_token = st.session_state.get("jwt_token")
        headers = {"Authorization": f"Bearer {jwt_token}"}
        response = requests.delete(f"{BASE_URL}/diary/{entry['id']}", headers=headers)
        
        if response.status_code == 200:
            data = {"title": entry["title"], "date": entry["date"], "content": entry["content"]}
            response = requests.post(f"{BASE_URL}/diary", json=data, headers=headers)
            
            if response.status_code == 200:
                st.success("Diary entry created!")
                new_entry = response.json()
                st.session_state.selected_entry = new_entry
                st.rerun()
            else:
                st.error(f"Failed to create diary entry: {response.status_code}")
                st.write("Response content:", response.text)
        else:
            st.error("Failed to delete diary entry.")


if "jwt_token" in st.session_state:
    jwt_token = st.session_state.get("jwt_token")
    headers = {"Authorization": f"Bearer {jwt_token}"}
    response = requests.get(f"{BASE_URL}/diaries", headers=headers)
    
    if response.status_code == 200:
        entries = response.json()
        
        if entries:
            if "selected_entry" not in st.session_state:
                st.session_state.selected_entry = None   
            # print(st.session_state.selected_entry)
            if st.session_state.selected_entry is None:
                show_diary_entries(entries)   
            else:
                show_diary_detail(st.session_state.selected_entry)
        else:
            st.write("No diary entries found.")
else:
    st.warning("로그인이 필요합니다.")