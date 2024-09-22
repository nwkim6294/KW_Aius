import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template
from run.database.connect_db import get_database
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from datetime import datetime, timedelta
import dill as pickle
import shap

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


def update_task():
    global task_status
    task_status["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_status["status"] = "Running"
    task_status["details"].clear()

    print(f"Task started at {task_status['start_time']}. Current status: {task_status['status']}")
    
    # ############ MySQL 연결 및 데이터 가져오기 ############
    db = get_database()
    cursor = db.cursor(dictionary=True)  # dictionary=True로 결과를 딕셔너리 형태로 반환
    
    # MySQL에서 데이터 가져오기
    task_status["current_step"] = "[predict] Fetching data from MySQL"
    print(task_status["current_step"])
    
    # ------------------------------------------------------
    # 데이터 불러오기 & 전처리
    # ------------------------------------------------------

    data_file_path = os.path.join("run", "models", "ur5e_combined.csv")
    data = pd.read_csv(data_file_path)
    
    sensor_columns = data.columns

    # 윈도우로 나누고 2D 이미지처럼 변환 (0.25초마다 슬라이딩, 36개 센서 변수)
    window_size = 31  
    step_size = 31  # 0.25초마다 슬라이딩
    num_sensors = len(sensor_columns)  # 36개의 센서 변수

    def create_sliding_windows(df, window_size, step_size):
        windows = []
        for i in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[i:i + window_size].values  # 각 윈도우의 데이터
            windows.append(window)
        return np.array(windows)

    # ------------------------------------------------------
    # 이상탐지
    # ------------------------------------------------------

    
    # 모델 불러오기
    anomodel_file_path = os.path.join("run", "models", "anomal_model.keras")
    custom_objects = {'mse': MeanSquaredError()}
    autoencoder = load_model(anomodel_file_path, custom_objects=custom_objects)

    # 각 윈도우 간 상대적 변화율 계산
    def calculate_change_rate(window1, window2):
        return np.mean(np.abs(window2 - window1))
    
    # MinMaxScaler 적용
    scaler = MinMaxScaler()
    scaled_sensor_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    
    # 상대적 변화율을 계산하여 이상 탐지
    def anomal_labeling(data) :
        
        # 0.25초 간격으로 슬라이딩 윈도우 생성
        X = create_sliding_windows(data, window_size, step_size)

        # CNN에 사용할 수 있도록 데이터를 2D 이미지 형태로 변환
        X_reshaped = X.reshape(-1, num_sensors, window_size, 1)  # (batch_size, height, width, channel)
            
        # 재구성 오류 계산 및 윈도우 간 변화율 계산
        reconstructed_data = autoencoder.predict(X_reshaped)
        
        change_rates = []
        for i in range(1, len(reconstructed_data)):
            change_rate = calculate_change_rate(reconstructed_data[i-1], reconstructed_data[i])
            change_rates.append(change_rate)

        # 이동 평균과 이동 표준편차 기반 임계값 설정
        moving_avg = pd.Series(change_rates).rolling(window=window_size).mean()
        moving_std = pd.Series(change_rates).rolling(window=window_size).std()

        # 임계값 설정 (동적으로 이동 평균 + 표준편차 기반으로 설정)
        # 이동 평균과 표준편차는 NaN이 발생할 수 있으므로 NaN을 제거
        threshold_1 = (moving_avg + 2 * moving_std).bfill()  # 'bfill'로 대체
        threshold_2 = (moving_avg + 3 * moving_std).bfill()  # 'bfill'로 대체

        # 이상 탐지 레이블링 (0: 정상, 1: 주의, 2: 이상)
        anomalies = np.zeros(len(change_rates))
        anomalies[np.array(change_rates) > threshold_2] = 2  # 이상
        anomalies[(np.array(change_rates) > threshold_1) & (np.array(change_rates) <= threshold_2)] = 1  # 주의

        # 결과 출력
        print(f"Detected {np.sum(anomalies == 2)} anomalies and {np.sum(anomalies == 1)} warnings out of {len(anomalies)} windows.")

        # 윈도우 사이즈에 맞춰 원본 데이터에 TARGET 열 추가
        target_labels = np.zeros(len(data))  # 원본 데이터 길이만큼 배열을 생성
        window_index = 0
        total_windows = len(anomalies)  # 전체 윈도우의 개수

        # 각 윈도우별로 레이블을 원본 데이터에 할당
        for i in range(0, len(data) - window_size + 1, step_size):
            window_end = i + window_size
            if window_index < total_windows:  # anomalies 배열의 인덱스 범위를 넘지 않도록 체크
                target_labels[i:window_end] = anomalies[window_index]  # 각 윈도우에 해당하는 레이블을 설정
            window_index += 1

        # TARGET 열을 원본 데이터에 추가
        data['TARGET'] = target_labels.astype(int)  # 레이블을 int형으로 변환 후 추가
        
        return data

    data = anomal_labeling(scaled_sensor_data)
    
    # ------------------------------------------------------
    # 샘플링
    # ------------------------------------------------------
    
    sampled_data = data.sample(60)
    
    
    # ------------------------------------------------------
    # 해석 모델
    # ------------------------------------------------------

    
    # Autoencoder 예측 함수 정의 (SHAP에서 사용)
    def autoencoder_predict(flat_data):
        reshaped_data = flat_data.reshape(-1, num_sensors, window_size, 1)
        reconstructed = autoencoder.predict(reshaped_data)
        reconstruction_error = np.mean(np.abs(reconstructed - reshaped_data), axis=(1, 2, 3))  # 재구성 오류 계산
        return reconstruction_error

    # SHAP Explainer 불러오기
    shap_model_file_path = os.path.join("run", "models", "shap_explainer_20.pkl")
    with open(shap_model_file_path, 'rb') as explainer_file:
        explainer = pickle.load(explainer_file)  # autoencoder_predict 함수가 정의된 상태에서 불러오기

    print("Loaded SHAP Explainer successfully.")

    def anoSHAP(data) :
        # sensor_columns: 센서 데이터에 대한 열 이름 리스트 (36개 센서)
        shap_data = data.drop(columns=['TARGET'])
        
        # 슬라이딩 윈도우 생성
        shap_data_windows = create_sliding_windows(shap_data, window_size, step_size)

        # 4D로 변환
        shap_data_reshaped = shap_data_windows.reshape(-1, num_sensors, window_size, 1)
        print(f"[anoSHAP] shap_data_reshaped shape: {shap_data_reshaped.shape}")

        # 2D로 평탄화
        shap_data_flat = shap_data_reshaped.reshape(shap_data_reshaped.shape[0], -1)
        print(f"[anoSHAP] shap_data_flat shape: {shap_data_flat.shape}")
        
        # 마지막 샘플을 선택
        explainer = shap.KernelExplainer(autoencoder_predict, shap_data_flat[:5])
        shap_values_last_sample = explainer.shap_values(shap_data_flat[:1])

        # SHAP 값의 첫 번째 항목을 가져옴 (SHAP의 결과는 리스트 형식이므로 첫 번째 요소를 선택)
        shap_values_sample = shap_values_last_sample[0]

        # 상위 중요한 feature 인덱스를 추출 (절댓값 기준으로 상위 feature)
        sorted_indices = np.argsort(-np.abs(shap_values_sample))

        # 센서와 시퀀스에 맞는 feature_names 생성
        feature_names = []
        for sensor in sensor_columns:
            for i in range(window_size):
                feature_names.append(f"{sensor} (timestep {i})")

        # 상위 6개의 중요한 feature를 중복되지 않게 선택
        unique_features = []
        used_sensors = set()

        for idx in sorted_indices:
            # 타임스텝을 제외하고 센서 이름과 조인트 정보만 남기기
            sensor_name = feature_names[idx].rsplit(" (timestep", 1)[0]  # 타임스텝 정보만 제거
            if sensor_name not in used_sensors:
                unique_features.append(sensor_name)  # 타임스텝 제거된 센서 이름 추가
                used_sensors.add(sensor_name)
            if len(unique_features) == 6:
                break

        print("Top 6 unique feature names:", unique_features)
        
        for i in range (6) :
            feature_col = f'Top{i+1}_Features'
            data[feature_col] = unique_features[i]
        
        return data
    
    sampled_data = anoSHAP(sampled_data)
    
    # ------------------------------------------------------
    # 예측
    # ------------------------------------------------------

    # 타임스텝을 고려한 입력 데이터 생성 함수
    def create_time_steps(data, time_steps):
        temp = []
        for i in range(len(data) - time_steps + 1):
            temp.append(data.iloc[i:i + time_steps].values)
        return np.array(temp)
    
    # 데이터 선택
    predict_X = data[sensor_columns].sample(frac=0.3)

    # 검증 및 테스트 데이터에 학습된 스케일러로 transform만 적용
    X_test_raw = pd.DataFrame(scaler.transform(predict_X), columns=sensor_columns)

    # 타임스텝 설정
    time_steps = 125

    # 타임스텝을 고려한 입력 데이터 생성
    X_test = create_time_steps(X_test_raw, time_steps)

    # 저장된 모델 로드
    predict_model_path = os.path.join("run", "models", "predict_model.keras")
    predict_model = load_model(predict_model_path)

    # 테스트 데이터에 대한 예측 수행
    predicted_sensor_values = predict_model.predict(X_test)
    predicted_df = pd.DataFrame(predicted_sensor_values, columns=sensor_columns)
    
    predicted_df = anomal_labeling(predicted_df)
    

    # ------------------------------------------------------
    # [예측] 이상상태 예측
    # ------------------------------------------------------
    
    sampled_predicted_data = predicted_df.sample(60)
    sampled_predicted_data = anoSHAP(sampled_predicted_data)

    # ------------------------------------------------------
    # 대시보드용으로 데이터 처리
    # ------------------------------------------------------

    # 현재 시간에서 9시간 전의 시간을 last_time으로 설정
    last_time1 = datetime.now() - timedelta(hours=9, minutes=4)
    last_time2 = datetime.now() - timedelta(hours=9, seconds=30)
    # 1초 간격으로 time 열을 생성하여 데이터프레임에 추가 (9시간 전부터 시작)
    sampled_data['time'] = [(last_time1 + timedelta(seconds=i)) for i in range(len(sampled_data))]
    sampled_predicted_data['time'] = [(last_time2 + timedelta(seconds=i)) for i in range(len(sampled_predicted_data))]
    
    print("Final 'time' columns created successfully.")

    ############ 데이터 처리 - 열이름 변경 ############
    task_status["current_step"] = "Renaming columns"
    print(task_status["current_step"])
    # 열 이름에서 ' (J1)', ' (J2)' 같은 형식을 '_J1', '_J2'로 변경
    sampled_data.columns = sampled_data.columns.str.replace(r' \((\w+)\)', r'_\1', regex=True)
    sampled_predicted_data.columns = sampled_predicted_data.columns.str.replace(r' \((\w+)\)', r'_\1', regex=True)
   
    print("Renamed columns in DataFrame.")

    def db_in_del(data, table_name) :
        # ------------------------------------------------------
        # DB - 데이터 삭제
        # ------------------------------------------------------
        ############ 앞의 60개 행 삭제 ############
        task_status["current_step"] = "Deleting the first 60 rows"
        print(task_status["current_step"])

        # 테이블에서 id 또는 time을 기준으로 가장 오래된 60개 행을 삭제
        delete_query = f"""
        DELETE FROM {table_name}
        ORDER BY time ASC  -- time 또는 id 열을 기준으로 정렬
        LIMIT 60;
        """
        
        cursor.execute(delete_query)
        db.commit()  # 변경사항을 적용하기 위해 commit 필요
        print(f"Deleted the first 60 rows from the {table_name} table.")
        
        
        # ------------------------------------------------------
        # DB - 데이터 삽입
        # ------------------------------------------------------
        
        # _id 컬럼 제거 (MySQL에서 사용할 필요 없으므로 제거)
        if '_id' in data.columns:
            data = data.drop(columns=['_id'])

        # 데이터프레임의 열 이름을 가져와서 쿼리에 동적으로 적용
        columns = ", ".join(data.columns)  # 열 이름을 쉼표로 구분하여 연결
        placeholders = ", ".join(["%s"] * len(data.columns))  # 각 열에 대한 플레이스홀더(%s) 생성

        # INSERT 쿼리 생성
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        # 데이터를 MySQL에 맞는 포맷으로 변환
        data_to_insert = [tuple(row) for row in data.values]

        # MySQL에 다중 행 삽입
        cursor.executemany(insert_query, data_to_insert)
        db.commit()  # 변경사항을 적용하기 위해 commit 필요
        
        task_status["details"].append(f"Inserted {cursor.rowcount} rows into MySQL.")
        task_status["current_step"] = "Inserting new data into MySQL"
        print(task_status["current_step"])

    db_in_del (sampled_predicted_data, 'sampling_predict')
    db_in_del (sampled_data, 'sampling')
    
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
