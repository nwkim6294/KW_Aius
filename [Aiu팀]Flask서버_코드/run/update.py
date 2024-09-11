import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from run.database.connect_db import get_database
from datetime import datetime, timedelta

# 전역 변수로 작업 상태 추적
task_status = {
    "start_time": None,
    "end_time": None,
    "status": "Idle",
    "current_step": "",
    "details": []
}
def update_app(app):
    @app.route('/update', methods=['GET'])
    def run_autoencoder():
        result = update_task()
        return jsonify(result), 200

    @app.route('/status', methods=['GET'])
    def status_page():
        sampled_data_html = update_task()['sampled_data_html']
        return render_template('status.html', status=task_status, table=sampled_data_html)

    return app




# 작업 상태 업데이트 함수
def update_task():
    global task_status
    task_status["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_status["status"] = "Running"
    task_status["details"].clear()

    print(f"Task started at {task_status['start_time']}. Current status: {task_status['status']}")

    ############ MySQL 연결 및 데이터 가져오기 ############
    db = get_database()
    cursor = db.cursor(dictionary=True)  # dictionary=True로 결과를 딕셔너리 형태로 반환
    
    # MySQL에서 데이터 가져오기
    task_status["current_step"] = "Fetching data from MySQL"
    print(task_status["current_step"])
    
    # query = "SELECT * FROM ur5_target"
    # cursor.execute(query)
    # mysql_data = cursor.fetchall()  # MySQL에서 데이터를 가져옴
    # print("Fetched data from MySQL.")

    # # MySQL에서 가져온 데이터를 DataFrame으로 변환
    # data = pd.DataFrame(mysql_data)

    # 파일 경로 설정
    file_path = os.path.join("run", "models", "ur5_target.csv")

    # CSV 파일 불러오기
    data = pd.read_csv(file_path)

    ############ 샘플링 (60개) ############
    task_status["current_step"] = "Sampling 60 data points"
    print(task_status["current_step"])
    
    # 데이터 샘플링 (60개)
    sampled_data = data.sample(60)
    print(f"Sampled {len(sampled_data)} data points.")

    ############ time 열 생성 (현재 시간 기준) ############
    # 현재 시간에서 9시간 전의 시간을 last_time으로 설정
    last_time = datetime.now() - timedelta(hours=9, minutes=1)

    # 1초 간격으로 time 열을 생성하여 데이터프레임에 추가 (9시간 전부터 시작)
    sampled_data['time'] = [(last_time + timedelta(seconds=i)) for i in range(len(sampled_data))]
    
    print("Final 'time' columns created successfully.")

    ############ 데이터 처리 - 열이름 변경 ############
    task_status["current_step"] = "Renaming columns"
    print(task_status["current_step"])
    
    # 열 이름에서 ' (J1)', ' (J2)' 같은 형식을 '_J1', '_J2'로 변경
    sampled_data.columns = sampled_data.columns.str.replace(r' \((\w+)\)', r'_\1', regex=True)
    print("Renamed columns in DataFrame.")

    ############ 앞의 60개 행 삭제 ############
    task_status["current_step"] = "Deleting the first 60 rows"
    print(task_status["current_step"])

    # 테이블에서 id 또는 time을 기준으로 가장 오래된 60개 행을 삭제
    delete_query = """
    DELETE FROM sample
    ORDER BY time ASC  -- time 또는 id 열을 기준으로 정렬
    LIMIT 60;
    """
    cursor.execute(delete_query)
    db.commit()  # 변경사항을 적용하기 위해 commit 필요
    print("Deleted the first 60 rows from the 'sample' table.")

    ############ 데이터베이스에 새로운 데이터 삽입 ############
    # _id 컬럼 제거 (MySQL에서 사용할 필요 없으므로 제거)
    if '_id' in sampled_data.columns:
        sampled_data = sampled_data.drop(columns=['_id'])

    # 데이터프레임의 열 이름을 가져와서 쿼리에 동적으로 적용
    columns = ", ".join(sampled_data.columns)  # 열 이름을 쉼표로 구분하여 연결
    placeholders = ", ".join(["%s"] * len(sampled_data.columns))  # 각 열에 대한 플레이스홀더(%s) 생성

    # INSERT 쿼리 생성
    insert_query = f"INSERT INTO sample ({columns}) VALUES ({placeholders})"

    # 데이터를 MySQL에 맞는 포맷으로 변환
    data_to_insert = [tuple(row) for row in sampled_data.values]

    # MySQL에 다중 행 삽입
    cursor.executemany(insert_query, data_to_insert)
    db.commit()  # 변경사항을 적용하기 위해 commit 필요
    task_status["details"].append(f"Inserted {cursor.rowcount} rows into MySQL.")

    task_status["current_step"] = "Inserting new data into MySQL"
    print(task_status["current_step"])

    
    
    ############ 마무리 ############
    task_status["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_status["status"] = "Completed"
    task_status["current_step"] = ""
    print(f"Task completed at {task_status['end_time']}. Current status: {task_status['status']}")

    # DataFrame을 HTML로 변환
    sampled_data_html = sampled_data.head().to_html(classes='dataframe', index=False)

    # 커서 및 연결 닫기
    cursor.close()
    db.close()

    return {"message": "Autoencoder task completed!", "sampled_data_html": sampled_data_html}
