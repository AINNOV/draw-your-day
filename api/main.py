from fastapi import FastAPI, Depends, UploadFile, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database.models import User, DiaryEntry
from database.database import SessionLocal, get_db, engine
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline
import torch
import os
from io import BytesIO
from PIL import Image
from datetime import date

app = FastAPI()

# Initialize models and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_4bit, device_map="auto", low_cpu_mem_usage=True)
model = model_4bit.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

sd_model_id = "ogkalu/comic-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)
pipe = pipe.to("cuda")

# Generate prompt from diary content
def generate_prompt(content):
    prompt_template = f"""
    Below is my diary. Using this Diary, write a prompt for text-to-image process.
    
    ### Diary
    {content}
    
    ### Prompt
    """
    inputs = tokenizer(prompt_template, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(inputs)
    prompt = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return prompt

# Generate images from prompt
def generate_images(prompt):
    image = pipe(prompt).images[0]
    image_path = f"./generated_images/{os.urandom(4).hex()}.png"
    image.save(image_path)
    return image_path

class LoginRequest(BaseModel):
    username: str
    password: str
    
class UserCreate(BaseModel):
    username: str
    password: str

# JWT 비밀 키 및 알고리즘 설정
SECRET_KEY = "your_secret_key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="유효하지 않은 인증 정보입니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

@app.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    
    if user is None:
        raise HTTPException(status_code=404, detail="존재하지 않는 아이디입니다.")
    
    # 여기에 비밀번호 확인 로직 추가 (해싱된 비밀번호와 비교)
    if user.hashed_password != request.password:  # 예시로 직접 비교
        raise HTTPException(status_code=401, detail="비밀번호가 틀립니다.")
    
    access_token = create_access_token(data={"user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    hashed_password = user.password  # Here, use hashing for security
    user = User(username= user.username, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

class DiaryCreate(BaseModel):
    title: str
    date: date
    content: str

@app.post("/diary")
async def create_diary(diary: DiaryCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)): 
    try:
        generated_prompt = generate_prompt(diary.content)
        generated_image_path = generate_images(generated_prompt)

        diary_entry = DiaryEntry(
            user_id = current_user.id,
            title=diary.title,
            date=diary.date,
            content=diary.content,
            generated_image_path=generated_image_path
        )
        db.add(diary_entry)
        db.commit()
        db.refresh(diary_entry)
        return diary_entry
    except Exception as e:
        print(f"Error occurred: {e}")  # 콘솔에 에러 메시지 출력
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.delete("/diary/{diary_id}")
async def delete_diary(diary_id: int, db: Session = Depends(get_db), authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    diary_entry = db.query(DiaryEntry).filter(DiaryEntry.id == diary_id).first()
    if diary_entry is None:
        raise HTTPException(status_code=404, detail="Diary entry not found")
    
    db.delete(diary_entry)
    db.commit()
    return {"detail": "Diary entry deleted successfully"}

@app.get("/diaries")
async def read_diaries(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(DiaryEntry).filter(DiaryEntry.user_id == current_user.id).all()
