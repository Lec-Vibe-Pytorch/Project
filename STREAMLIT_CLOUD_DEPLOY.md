# Streamlit Cloud 배포 가이드

## 배포 단계

### 1. GitHub 저장소 준비

1. GitHub에 새 저장소를 생성하거나 기존 저장소를 사용합니다.

2. 다음 파일들이 포함되어야 합니다:
   - `app.py` - 메인 앱 파일
   - `requirements.txt` - 의존성 목록
   - `model/asl_linear_best.pth` - 학습된 모델 파일
   - `README.md` - 프로젝트 설명 (선택사항)
   - `.streamlit/config.toml` - Streamlit 설정 (선택사항)

### 2. GitHub에 코드 푸시

```bash
# Git 초기화 (아직 안 했다면)
git init

# 파일 추가
git add app.py requirements.txt README.md .streamlit/
git add model/asl_linear_best.pth

# 커밋
git commit -m "Initial commit: ASL 분류기 Streamlit 앱"

# GitHub 저장소에 푸시
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

**주의**: 모델 파일(`.pth`)은 크기가 클 수 있습니다. GitHub의 파일 크기 제한(100MB)을 확인하세요.

### 3. Streamlit Cloud에 배포

1. [Streamlit Cloud](https://streamlit.io/cloud)에 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 다음 정보 입력:
   - **Repository**: GitHub 저장소 선택
   - **Branch**: `main` (또는 기본 브랜치)
   - **Main file path**: `app.py`
   - **App URL**: 원하는 URL 입력 (예: `asl-classifier`)

5. "Deploy!" 클릭

### 4. 배포 확인

배포가 완료되면 Streamlit Cloud에서 제공하는 URL로 앱에 접속할 수 있습니다.

## 문제 해결

### 모델 파일이 너무 큰 경우

모델 파일이 GitHub의 크기 제한을 초과하는 경우:

1. **Git LFS 사용**:
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add model/asl_linear_best.pth
   git commit -m "Add model file with LFS"
   ```

2. **외부 스토리지 사용**: 
   - Google Drive, Dropbox, AWS S3 등에 모델 파일 업로드
   - 앱에서 다운로드하도록 코드 수정

### 의존성 문제

`requirements.txt`에 모든 필요한 라이브러리가 포함되어 있는지 확인하세요.

### 모델 파일 경로

앱은 다음 경로에서 모델 파일을 찾습니다:
- `./model/asl_linear_best.pth` (우선)
- `./model/nnLinear_model.pth`
- 기타 경로들

## 현재 프로젝트 구조

```
Project/
├── app.py                    # 메인 앱 파일
├── requirements.txt          # 의존성 목록
├── README.md                 # 프로젝트 설명
├── .streamlit/
│   └── config.toml          # Streamlit 설정
├── model/
│   └── asl_linear_best.pth  # 학습된 모델
└── data/
    └── asl_image/           # 테스트 이미지 (선택사항)
```

## 추가 참고사항

- Streamlit Cloud는 무료 플랜에서도 사용 가능합니다
- 배포 후 코드를 업데이트하면 자동으로 재배포됩니다
- 로그는 Streamlit Cloud 대시보드에서 확인할 수 있습니다

