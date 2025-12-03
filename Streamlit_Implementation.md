# ASL Streamlit 웹앱 구현 계획

학습된 모델(`best_nnLinear_model.pth`)을 로드하여 사용자 친화적인 Streamlit 기반 웹앱을 제작합니다.

## 주요 변경 사항

### 모델 구조 업데이트
- **기존**: 32x32 RGB 이미지 입력 (3072 features)
- **신규**: 28x28 Grayscale 이미지 입력 (784 features)
- **모델 아키텍처**: 784 → 512 → 256 → 128 → 24 (BatchNorm + Dropout 포함)

### 모델 경로 변경
- **기존**: `./model/asl_linear_best.pth`
- **신규**: `./model/best_nnLinear_model.pth`

### 이미지 전처리 변경
- **기존**: RGB 이미지, 32x32 리사이즈, 3채널 정규화
- **신규**: Grayscale 이미지, 28x28 리사이즈, 1채널 정규화

## Proposed Changes

### [MODIFY] [app.py](file:///h:/내%20드라이브/강의자료/Vibe_Coding/Lec_Vibe_Pytorch/project/app.py)

#### 1. 모델 클래스 정의 수정
- `ASLLinearNet` 클래스를 노트북과 동일한 구조로 업데이트
- 입력 크기: 784 (28x28)
- 레이어 구조: fc1(512) + bn1 + dropout1(0.3) → fc2(256) + bn2 + dropout2(0.3) → fc3(128) + bn3 + dropout3(0.2) → fc4(24)

#### 2. 모델 경로 및 설정 변경
```python
MODEL_PATH = './model/best_nnLinear_model.pth'
```

#### 3. 이미지 전처리 함수 수정
```python
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),  # Grayscale 변환
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 1채널 정규화
    ])
    return transform(image).unsqueeze(0)
```

#### 4. UI 개선
- 테스트 이미지 선택 옵션 추가 (a.png, b.png)
- 업로드 또는 샘플 이미지 선택 기능
- 예측 결과 시각화 개선

## Verification Plan

### Automated Tests
1. 모델 로드 테스트
   ```bash
   python -c "import torch; import torch.nn as nn; from app import ASLLinearNet; model = ASLLinearNet(); print('Model loaded successfully')"
   ```

2. Streamlit 앱 실행
   ```bash
   streamlit run app.py
   ```

### Manual Verification
1. 웹 브라우저에서 앱 접속 확인
2. 샘플 이미지(a.png, b.png) 테스트
3. 사용자 정의 이미지 업로드 테스트
4. 예측 결과 정확도 확인
