# ASL nnLinear 분류 모델 구현 계획

CSV 파일로 된 ASL 데이터셋을 nn.Linear 모델로 분류하는 PyTorch 딥러닝 코드를 작성합니다.

## 데이터셋 정보

- **훈련 데이터**: `sign_mnist_train.csv` (27,455개 샘플)
- **검증 데이터**: `sign_mnist_valid.csv`
- **이미지 크기**: 28x28 픽셀 (784개 특징)
- **클래스 수**: 24개 (A-Z에서 J와 Z 제외)

## Proposed Changes

### [NEW] [ASL_nnLinear_test.ipynb](file:///H:/내%20드라이브/강의자료/Vibe_Coding/Lec_Vibe_Pytorch/project/ASL_nnLinear_test.ipynb)

완전히 새로운 Jupyter Notebook을 생성하여 다음 내용을 포함:

#### 1. CSV 데이터 로드 및 탐색
- pandas를 사용하여 CSV 파일 로드
- 데이터 구조, 클래스 분포, 샘플 이미지 시각화

#### 2. Custom Dataset 클래스 정의 및 데이터 증강
- `ASLCSVDataset` 클래스 구현
- 픽셀 값 정규화 (0-255 → 0-1)
- 데이터 증강: RandomRotation, RandomAffine 등

#### 3. DataLoader 생성
- Train/Validation split (또는 별도 CSV 사용)
- Batch size, shuffle, num_workers 설정

#### 4. 다층 nn.Linear 모델 정의
- 입력층: 784 (28x28)
- 은닉층: 512 → 256 → 128
- 출력층: 24 (클래스 수)
- 활성화 함수: ReLU
- Dropout 추가로 과적합 방지

#### 5. 학습 루프 (Train/Validation)
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Epoch별 Train/Validation loss 및 accuracy 계산
- Best model 저장 로직

#### 6. 성능 시각화 및 과적합 분석
- Train vs Validation Loss 그래프
- Train vs Validation Accuracy 그래프
- Confusion Matrix
- 한글 폰트 설정 (기존 KOREAN_FONT_GUIDE.md 참고)

#### 7. 모델 저장
- 최고 성능 모델을 `./model/best_nnLinear_model.pth`에 저장

#### 8. 모델 로드 및 테스트 이미지 예측
- 저장된 모델 로드
- `a.png`, `b.png` 이미지 예측 (존재하는 경우)
- 예측 결과 시각화

## Verification Plan

### Automated Tests
노트북 실행 후 다음을 확인:
```bash
# 노트북 실행 (Jupyter에서 수동 실행)
jupyter notebook ASL_nnLinear_test.ipynb
```

### Manual Verification
1. **데이터 로드 확인**: CSV 파일이 정상적으로 로드되고 데이터 탐색 결과가 표시되는지 확인
2. **모델 학습 확인**: 학습이 정상적으로 진행되고 loss가 감소하는지 확인
3. **시각화 확인**: 그래프가 정상적으로 표시되고 한글이 깨지지 않는지 확인
4. **모델 저장 확인**: `./model/best_nnLinear_model.pth` 파일이 생성되는지 확인
5. **예측 확인**: 테스트 이미지에 대한 예측이 정상적으로 수행되는지 확인
