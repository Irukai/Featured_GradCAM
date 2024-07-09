main.py : 프로그램 메인 코드

modules
 └ config.py		전역 변수 & 하이퍼파마리터 설정
 └ data_loader.py		load_data, check_data : csv파일 불러오기
 └ dataset.py		데이터셋, 데이터로더, 청크 데이터 생성
 └ gradcam.py		gcam_plot, fgcam_plot, show_attributions : GradCAM, Featured GradCAM, Shapley Value 결과 출력 함수
			generate_cross_mask, generate_gradcam_labels : Featured GradCAM의 Loss 함수를 위해 GradCAM lable 만드는 함수 (일단 main에서 사용하지 않는 상태 7/2)
 └ models.py		각 모델 클래스 선언
 └ train.py		train_model, train_model_ours, evaluate_model : 학습 및 테스트 함수, train_model_ours는 GradCAM Loss 때문에 새로 정의하였음. 그러나 일단 주석처리함.
 └ utils.py		min_max_normalize, feature_selection, plot_graph, show_data, transform_data, evaluate_cams, evaluate_model_on_dataloader
			이 중에서 현재 사용 중인 함수는 show_data 뿐임. 나머지는 사용하지 않음. evaluate 관련 함수는 Precision, Recall, F1-score 계산하는 함수임.

Preprocessed_data2
 └ iphost06_wls1.csv	constant_value_remove -> StandardScaler -> RobustScaler -> 1h rolling -> 1 time length 

Result
 └ data
     └ data_plot_#.jpeg	Anomaly 15개, Normal 15개
 └ featured_gradcam_result.jpeg
 └ gradcam_result.jpeg
 └ shapley_value_result.jpeg


>>> python main.py

