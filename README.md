# ASL (American Sign Language) 분류기

PyTorch nn.Linear 기반 신경망을 사용한 ASL 손 모양 분류 웹앱

## 기능

- ✋ ASL 손 모양 이미지 분류 (A-Y, J와 Z 제외)
- 📤 이미지 업로드 또는 테스트 이미지 사용
- 📊 예측 확률 시각화
- 🎯 Top 5 예측 결과 표시

## 사용 방법

1. Streamlit 앱 실행:
   ```bash
   streamlit run app.py
   ```

2. 브라우저에서 사이드바를 통해:
   - 테스트 이미지 선택 (a.png, b.png)
   - 또는 이미지 업로드

## 모델

- **아키텍처**: ImprovedASLClassifier
  - 입력: 784 (28x28 이미지)
  - 은닉층: 1024 → 512 → 256 → 128
  - 출력: 24 클래스 (ASL 알파벳)
  - Batch Normalization 포함

## 요구사항

필요한 라이브러리는 `requirements.txt`를 참조하세요.

## 배포

이 앱은 Streamlit Cloud에서 배포할 수 있습니다.

