import subprocess
import time

# FastAPI 서버 실행
fastapi_process = subprocess.Popen(
    ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
)

# FastAPI 서버가 시작될 때까지 대기 (약간의 시간 필요)
time.sleep(2)

# Streamlit 서버 실행
streamlit_process = subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8501"]
)

# 두 서버가 동시에 실행 중인 동안 대기
try:
    fastapi_process.wait()
    streamlit_process.wait()
except KeyboardInterrupt:
    print("서버 종료 중...")

# 서버 종료
fastapi_process.terminate()
streamlit_process.terminate()
