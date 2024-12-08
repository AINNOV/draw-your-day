import faiss
from fastapi import FastAPI, Depends, Request, HTTPException, status, Header
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import requests
from pydantic import BaseModel
from database.models import User, DiaryEntry
from database.database import get_db
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline
import torch
import os
import json
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from datetime import date
import logging
import httpx
from sentence_transformers import SentenceTransformer

class DiaryRetriever:
    def __init__(self, index_path = None, model_name = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.prompts, self.responses = self.load_prompts_and_responses(index_path)
        else:
            # If index file doesn't exist, create a new one
            self.index = None
            self.prompts = []
            self.responses = []
            if index_path:
                # Generate a new index and save it
                json_data = self.load_json("./data/raw/DYD_newtrain.json")
                self.index, self.prompts = self.create_faiss_index(json_data)
                self.save_faiss_index(self.index, index_path)  # Save it to the provided path

            self.index = faiss.read_index(index_path)
            self.prompts, self.responses = self.load_prompts_and_responses(index_path)

    def load_json(self, file_path):
        with open(file_path, "r", encoding="UTF-8") as f:
            return json.load(f)

    ## create faiss index from your json only if needed ##
    def create_faiss_index(self, json_data):
        prompts = [item["prompt"] for item in json_data]
        embeddings = self.model.encode(prompts, convert_to_numpy = True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension) # L2 distance
        index.add(embeddings)
        return index, prompts

    ## save only if needed ##
    def save_faiss_index(self, index, file_path="./DYD_faiss.bin"):
        faiss.write_index(index, file_path)

    ## load "prompt"-"response" json and save to instance variables ##
    def load_prompts_and_responses(self, index_path):
        json_file_path = "./data/raw/DYD_newtrain.json"
        json_data = self.load_json(json_file_path)
        prompts = [item["prompt"] for item in json_data]
        responses = [item["response"] for item in json_data]
        return prompts, responses

    ## retrieve top k similar docs(=contracts) and corresponding responses(=analyses) ##
    def search_similar_documents(self, query, top_k=  3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [{"prompt": self.prompts[idx], "response": self.responses[idx], "distance": distances[0][i]} for i, idx in enumerate(indices[0])]
        return results

# logging.basicConfig(level=logging.DEBUG)
load_dotenv()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = FastAPI()

model_name = "meta-llama/Llama-2-7b-chat-hf"
quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config = quantization_4bit,
    device_map = "auto",
    low_cpu_mem_usage = True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sd_model_id = "ogkalu/comic-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id)
pipe = pipe.to("cuda")

def prompt_with_template(template_path, prompt, rag):
    with open(template_path, "r", encoding = "UTF-8") as file:
        sys_template = file.read()
    return [
    {
        "role": "system",
        "content": f"{sys_template}" + rag + "\n\nNow promptize the following diary entry:" # retrieval results added in this position
    },
    {
        "role": "user", 
        "content": "\n### Input Diary:\n" + prompt + "\n### Promptized Output:\n"
        },
    ]

def generate_prompt(content):
    retriever = DiaryRetriever(index_path="./DYD_faiss.bin") 

    search_results = retriever.search_similar_documents(content, top_k = 1)[0]
    rag = f"\n\n### Similar diary: {search_results['prompt']}\n\n### Its promptized output: {search_results['response']}\n"
    message = prompt_with_template('./template/template_for_rag.txt', content, rag)
    inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt")
    input_len = len(tokenizer.batch_decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

    generate_ids = model_4bit.generate(inputs.to(device)) 
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    llama2_output = outputs[input_len:]
    # print(llama2_output)
    return llama2_output

def generate_images(prompt):
    image = pipe(prompt = prompt, negative_prompt = "disfigured, deformed, ugly, blurry, low resolution, poorly drawn, unnatural, bad anatomy, blurred faces, low-fidelity").images[0]
    image_path = f"./generated_images/{os.urandom(4).hex()}.png"
    image.save(image_path)
    return image_path

class LoginRequest(BaseModel):
    username: str
    password: str
    
class UserCreate(BaseModel):
    username: str
    password: str

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth = OAuth()
app.add_middleware(SessionMiddleware, secret_key = SECRET_KEY)
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


@app.get("/login_google")
async def login_google():
    authorize_url = "https://accounts.google.com/o/oauth2/auth"
    redirect_uri = "http://localhost:8000/login_google/callback"
    
    google_login_url = f"""
    {authorize_url}?response_type=code&client_id={os.getenv("GOOGLE_CLIENT_ID")}&redirect_uri={redirect_uri}&scope=openid%20profile%20email&access_type=offline
    """
    return RedirectResponse(url=google_login_url)

@app.get("/login_google/callback")
async def auth_google(request: Request):
    code = request.query_params.get("code")
    redirect_uri = "http://localhost:8000/login_google/callback"

    data = {
        "code": code,
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post("https://accounts.google.com/o/oauth2/token", data=data)
        response.raise_for_status()
        token = response.json()
    
    access_token = token.get("access_token")
    id_token = token.get("id_token")
    jwks_url = "https://www.googleapis.com/oauth2/v3/certs"
    jwks = requests.get(jwks_url).json()
    algorithm = ["RS256"]
    
    decoded_token = jwt.decode(
        id_token, 
        jwks, 
        algorithms = algorithm,
        audience = os.getenv("GOOGLE_CLIENT_ID"),
        access_token = access_token
    )
    
    user = decoded_token.get("email")
    return_url = f"http://1.229.207.166:8501/login/?user_info={user}"
    return RedirectResponse(url=return_url)

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
    
    if user.hashed_password != request.password:  
        raise HTTPException(status_code=401, detail="비밀번호가 틀립니다.")
    
    access_token = create_access_token(data={"user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register(user: UserCreate, db: Session = Depends(get_db)):
    print(user)
    hashed_password = user.password  
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
        print(f"Error occurred: {e}") 
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
