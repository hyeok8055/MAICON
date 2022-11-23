# 국방 AI 경진대회 코드 사용법

- 🌊해군호텔 602호🚢팀, 김태옥, 윤주혁, 박준현, 나강건
- 닉네임 : 육군김태옥, hyeok2, junhyun1273, devgeon

## 핵심 파일 설명

#### 이하 코드들의 root 폴더 : **`/root/workspace/Final_submission`** 

  - 학습 데이터 경로:   
    `1) baseline 코드로 학습시 : ./baseline/data`   
    `2) CDP 코드로 학습시 : ./CDP/data`

  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터:    
    `1) baseline 코드로 학습시 : encoder = timm-regnetx_120, encoder_weight = imagenet`   
    `2) CDP 코드로 학습시 : encoder_name = "efficientnet-b5", encoder_weights = "imagenet" `   

  - 학습 메인 코드:   
    `1) baseline : ./baseline/train.py`  
    `2) CDP : ./CDP/train.py`  

  - 테스트 메인 코드:   
    `1) baseline : ./baseline/predict.py`  
    `2) CDP : ./CDP/inference_visualize.py`  

  - 테스트 이미지 경로:   
    `1) baseline : ./baseline/data/test`  
    `2) CDP : ./CDP/data/test`  

  - 테스트 결과 이미지 경로:   
    `1) baseline : ./baseline/results/pred/legnet120_pred/mask`  
    `2) CDP(30epoch) : ./CDP/res/first_model_submission`  
    `3) CDP(60epoch) : ./CDP/res/second_model_submission`  
    `4) ensemble : ./result/last_submission/en14`   


## 코드 구조 설명

  - 주최측 제공인 `baseline 코드` 및 `change_detection.pytorch` 모델 사용.
    
  - 최종 결과는 **CDP(30epoch) + CDP(60epoch) + baseline(deeplabv3plus+ timm_regnetx_120, 7 epoch)의  Hard voting Ensemble로 도출**    
    - 최종 사용 모델 :  
     `1) baseline : segmentation-models-pytorch에서 제공하는 DeepLabV3Plus 모델 사용 `   
      `2) CDP :  change_detection.pytorch 에서 제공하는 DeepLabV3Plus 모델 사용 `
   
  - **최종 제출 파일 :  ./result/last_submission/en14.zip**

  - **학습된 가중치 파일 :**   
  
       - Baseline        
       기존 제출파일에 대한 train 폴더 환경오류로 인한 삭제로 동일 환경에서 재구현하여 Final_model에 저장   
    `1) baseline : ./Final_model/baseline_reg120.pt`    
          
     - CDP   
      `1) CDP(30epoch) : ./Final_model/CDP_DLV3plus_efb5(try1).pth `    
      `2) CDP(60epoch) : ./Final_model/CDP_DLV3plus_efb5(try2).pth`   
      
  - **Ensemble 및 후처리**    
  `1) 각 모델의 예측 결과를 hard voting 방식으로 ensemble함`    
  `2) ensemble된 결과 이미지에 morphology closing 및 opening 연산을 적용하여 노이즈를 제거함.`   


## 주요 설치 library

#### [1] Baseline 주요 설치 library

- segmentation-models-pytorch==0.3.0
- opencv-python==4.6.0.66

#### [2] CDP 주요 설치 library

  - change-detection-pytorch == 0.1.4 **<< 프로젝트에 맞게 라이브러리 일부 수정**   
  - torchvision>=0.5.0
  - pretrainedmodels==0.7.4
  - efficientnet-pytorch==0.6.3
  - timm==0.4.12
  - albumentations>=1.0.0,<=1.0.3

## 실행 환경 설정

#### [1] baseline 실행 환경 설정

* baseline 기본 실행환경과 동일
  - `./baseline/data/train` 의 학습데이터 12000개 에서 x, y 사이 크기가 다른 **5개의 이미지를 제거**함
  - 제거한 이미지 파일명 :    
  `2019_WSN_2LB_000026.png, 2019_WSN_2LB_000059.png, 2019_WSN_2LB_000073.png, 2019_WSN_2LB_000047.png, 2019_WSN_2LB_000016.png`   
  
* 세부 실행환경은 `./baseline/config/train.yaml` 참고

#### [2] CDP 실행 환경 설정

* **소스 코드 및 환경 설치**
  training 및 prediction을 위한 소스 코드는 `/workspace/Final_Submission/CDP/`내에 저장되어 있음

  * `train.py`: 모델 training을 위한 소스 코드

  * `inference_visualize.py`: 학습된 모델을 이용해 test data의 prediction 수행

  * `make_submission.ipynb`: 모델이 prediction한 결과를 제출 양식에 맞게 변환 및 제출함.

  * `data/`: train, validation, test를 위한 데이터가 정리되어 있음. 원본 데이터에서 중간 세로축을 기준으로 이등분함. train 데이터 중 각 클래스별로 97%는 training, 3%는 validation에 이용함.
  
    * **test/label 폴더 내에 있는 mask 이미지들은 inference_vis의 입력 format을 맞추기 위한 dummy 변수임**   

  * `logs/`: training 과정 중 log를 텍스트 파일 형식으로 저장함. F1-score, Precision, Recall, mIoU가 epoch별로 저장됨.

  * `models/`: 학습한 모델 파일이 저장됨.

  * `res/`: 각 모델이 test데이터를 predict한 결과를 저장함. `_res`로 끝나는 폴더는 모델이 예측한 결과를 눈으로 볼 수 있게 0~255scale로 저장했고, `_submission`으로 끝나는 폴더는 제출 양식에 맞게 변환한 자료임.

* `change-detection-pytorch`라이브러리를 이용함. **기존 라이브러리는 binary class를 기준으로 작성되어 그대로 적용하기 힘들어 library code를 4-class에 맞게 일부 수정함.** 

     * 수정된 라이브러리는 `/root/anaconda3/lib/python3.9/site-packages/change_detection_pytorch/`에 저장되어 있음.
     * `GitHub`: https://github.com/likyoo/change_detection.pytorch


## 학습 실행 방법
#### [1] baseline 학습 실행방법

- 학습 데이터 경로 설정
    ```bash
    ./data/train # 학습용 이미지 데이터 경로
    x : x        # train 이미지가 저장된 경로
    y : y        # mask 이미지가 저장된 경로   
    ```
- config 기반 학습진행 과정:

  * 세부 config 값 `./baseline/config/train.yaml` 참고

  * 실행 후 7 epoch 에서 종료 ( miou 값 0.55 이상)

  * 제출 모델.pt 는 환경오류로 지워져 재구현 하여 서버 업로드함
  
- 학습파일 실행
  ```bash
  python3 ./baseline/train.py
  ```



#### [2] CDP 학습 실행 방법

   - 학습 데이터 경로 설정

      - `./CDP/train.py` 내의 line 23~36에서 학습 데이터 경로 설정.    
        
        ```bash
        img_dir: (prj_dir)/data/train  # 학습용 이미지 데이터 절대경로
        sub_dir_1: A  #  left image가 저장된 상대경로
        sub_dir_2: B  # right image가 저장된 상대경로
        ann_dir: (prj_dir)/data/train/label # 학습용 라벨 데이터 절대경로
        ```

   - 학습 파일 실행
     
     ```bash
     /root/anaconda3/bin/python /workspace/Final_Submission/CDP/train.py
     ```

## 테스트 실행 방법

#### [1] Baseline 테스트 실행 방법

- 실행 과정
  - `python3 /workspace/Final_Submission/baseline/predict.py` 터미널 입력

#### [2] CDP 테스트 실행 방법

- 실행 과정
  - `/root/anaconda3/bin/python /workspace/Final_Submission/CDP/inference_visualize.py` 터미널 입력
  - `./CDP/make_submission.ipynb` 실행하여 `CDP 결과 변환` 부분 실행
  - `res/(model)_submission/`내에 테스트 결과 생성됨.

#### [3] Hard voting ensemble and morphology 실행 방법

- 실행 과정
  - `./ensemble/ensemble_fianl.ipynb` 파일 내 셀 순차실행

