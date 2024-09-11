from run.update import update_app, update_task
from run.predict import predict_app, predict_task
import threading
import time
from flask import Flask

# Flask 앱 초기화 및 각 모듈의 라우트 추가
app = Flask(__name__, template_folder='run/templates')  # 템플릿 폴더 경로 지정
app = update_app(app)  # update_app 함수에서 라우트 추가
app = predict_app(app)  # predict_app 함수에서 라우트 추가

# 템플릿 자동 리로드 설정
app.config['TEMPLATES_AUTO_RELOAD'] = True

def run_periodic_update_task():
    while True:
        with app.app_context():  # Flask 애플리케이션 컨텍스트 설정
            # update_task 먼저 실행
            print("Starting update task")
            update_task()
            time.sleep(30)  # 30초 대기 후

            # predict_task 실행
            print("Starting predict task")
            predict_task()
            time.sleep(30)  # 다시 30초 대기

if __name__ == '__main__':
    # 백그라운드에서 주기적으로 작업을 실행할 스레드 시작
    task_thread = threading.Thread(target=run_periodic_update_task)
    task_thread.daemon = True  # 메인 스레드 종료 시, 이 스레드도 함께 종료됨
    task_thread.start()

    # Flask 앱 실행
    app.run(debug=True, host="0.0.0.0")
