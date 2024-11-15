# create_tables.py
from models import Base
from database import engine
from sqlalchemy import Column, Integer, String, Text, Date, ForeignKey
from sqlalchemy.exc import IntegrityError

# 데이터베이스에 테이블 생성
if __name__ == "__main__":
    try:
        # 테이블 생성 (이미 존재하는 경우에는 아무 작업도 하지 않음)
        Base.metadata.create_all(bind=engine)
        print("Tables created successfully (if they did not already exist).")
    except IntegrityError:
        print("Tables already exist, no changes made.")
