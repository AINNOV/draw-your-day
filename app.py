import streamlit as st
import calendar
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
from pages.cookie_manager import cookies

st.set_option('client.showErrorDetails', False)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def main_page():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: "#e93109";
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# "#FFFFFF"
    # today = datetime.today()
    # year, month = today.year, today.month

    # cal = calendar.Calendar()
    # days = cal.itermonthdays(year, month)
    
    # calendar_html = f"""
    # <div class="calendar">
    #     <div class="calendar-header">
    #         <span>&lt;</span>
    #         <span>{calendar.month_name[month]} {year}</span>
    #         <span class = "nav">&gt;</span>
    #     </div>
    #     <div class="calendar-days">
    # """
    
    # # Add weekdays header
    # for weekday in calendar.day_abbr:
    #     calendar_html += f"<div class='day-header'>{weekday}</div>"

    # # Add days of the month
    # for day in days:
    #     if day == 0:  # Empty cells for days outside the current month
    #         calendar_html += "<div class='day empty'></div>"
    #     else:
    #         class_name = "day"
    #         if day == today.day and month == today.month and year == today.year:
    #             class_name += " selected"
    #         calendar_html += f"<div class='{class_name}'>{day}</div>"
    
    # calendar_html += "</div></div>"  # Close the calendar grid and container

    
    # image_path = "main_page.png"  
    # with open(image_path, "rb") as image_file:
    #     encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
    # # HTML + CSS for the calendar
    # html_code = f"""
    # <!DOCTYPE html>
    # <html lang="en">
    # <head>
    #     <meta charset="UTF-8">
    #     <meta name="viewport" content="width=device-width, initial-scale=1.0">
    #     <title>Draw Your Day</title>
    #     <style>
    #         body {{
    #             margin: 0;
    #             padding: 0;
    #             font-family: 'Arial', sans-serif;
    #             display: flex;
    #             justify-content: center;
    #             align-items: center;
    #             height: 100vh;
    #             background: url('data:image/png;base64,{encoded_image});
    #             background-size: cover;
    #         }}
    #         .container {{
    #             text-align: center;
    #             background: rgba(255, 255, 255, 0.8);
    #             padding: 30px;
    #             border-radius: 15px;
    #             box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.2);
    #             max-width: 600px;
    #         }}
    #         h1 {{
    #             font-size: 3em;
    #             color: #333;
    #             margin-bottom: 10px;
    #         }}
    #         p {{
    #             font-size: 1.2em;
    #             color: #555;
    #             margin-bottom: 30px;
    #         }}
    #         .calendar {{
    #             display: inline-block;
    #             width: 100%;
    #             background: white;
    #             border-radius: 10px;
    #             grid-template-columns: repeat(7, 1fr);
    #             padding: 20px;
    #             box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1);
    #         }}
    #         .calendar-header {{
    #             display: flex;
    #             justify-content: space-between;
    #             align-items: center;
    #             margin-bottom: 20px;
    #             font-size: 1.2em;
    #             color: #333;
    #         }}
    #         .calendar-days {{
    #             display: grid;
    #             grid-template-columns: repeat(7, 1fr);
    #             gap: 5px;
    #         }}
    #         .day-header {{
    #             font-size: 0.9em;
    #             font-weight: bold;
    #             color: #666;
    #             text-align: center;
    #         }}
    #         .day {{
    #             width: 40px;
    #             height: 40px;
    #             display: flex;
    #             justify-content: center;
    #             align-items: center;
    #             font-size: 1em;
    #             color: #333;
    #             background: #f9f9f9;
    #             border-radius: 5px;
    #             cursor: pointer;
    #         }}
    #         .day.empty {{
    #             background: none;
    #             cursor: default;
    #         }}
    #         .day:hover {{
    #             background: #d3d3d3;
    #         }}
    #         .day.selected {{
    #             background: #333;
    #             color: white;
    #         }}
    #     </style>
    # </head>
    # <body>
    #     <div class="container">
    #         <h1>Draw Your Day</h1>
    #         <p>Please tell me how was your day. <br>Then I will draw and archive for you</p>
    #         {calendar_html}
    #     </div>
    # </body>
    # </html>
    # """

    # # Streamlit component
    # st.components.v1.html(html_code, height=800, scrolling = False)
    st.image("mainpage.png", width = 800)

if "logged_in" not in st.session_state:
    if cookies.get("logged_in") is None:
        st.session_state.logged_in = False
    else:
        st.session_state.logged_in = cookies.get("logged_in") == "True"
        st.session_state.jwt_token = cookies.get("jwt_token")

if st.session_state.logged_in:
    pages = {
        "Menu": [
            st.Page(main_page, title="Main"),
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