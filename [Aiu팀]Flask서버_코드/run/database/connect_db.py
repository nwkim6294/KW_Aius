import mysql.connector
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_database():
    host = os.getenv("MYSQL_HOST")  # IP 주소만 입력 (포트 제외)
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")
    port = os.getenv("MYSQL_PORT", 3306)  # 기본 포트 3306

    # MySQL에 연결
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port  # 포트를 별도로 지정
    )
    
    return connection
