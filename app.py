import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# 페이지 설정
st.set_page_config(
    page_title="ASL 분류기",
    page_icon="✋",
    layout="wide"
)

# 모델 클래스 정의
class ImprovedASLClassifier(nn.Module):
    """개선된 nn.Linear 기반 다중 계층 신경망 모델 (Batch Normalization 포함)"""
    
    def __init__(self, input_size=784, hidden_sizes=[1024, 512, 256, 128], num_classes=24):
        super(ImprovedASLClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# 레이블 매핑 (ASL 알파벳: A-Y, J와 Z 제외)
label_to_letter = {i: chr(65 + i) if i < 9 else chr(66 + i) for i in range(24)}
# 0-8: A-I, 9-23: K-Y (J=9, Z=25 제외)

@st.cache_resource
def load_model():
    """모델 로드 (캐싱)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedASLClassifier(input_size=784, hidden_sizes=[1024, 512, 256, 128], num_classes=24)
    
    # 여러 경로에서 모델 파일 찾기
    possible_paths = [
        './model/asl_linear_best.pth',  # 사용자 지정 경로
        './model/nnLinear_model.pth',
        './project/model/nnLinear_model.pth',
        './data/nnLinear_model.pth'
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        st.error(f"모델 파일을 찾을 수 없습니다. 다음 경로들을 확인했습니다:")
        for path in possible_paths:
            st.error(f"  - {path}")
        st.stop()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.sidebar.info(f"모델 로드: {model_path}")
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.stop()
    
    model.eval()
    model = model.to(device)
    return model, device

def preprocess_image(image):
    """이미지 전처리"""
    transform = transforms.Compose([
        transforms.Grayscale(),  # RGB를 Grayscale로 변환
        transforms.Resize((28, 28)),  # 28x28로 리사이즈
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image)

def predict_image(image, model, device, label_to_letter):
    """이미지 예측"""
    # 전처리
    image_tensor = preprocess_image(image).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = predicted.item()
        confidence_score = confidence.item()
    
    # 레이블을 문자로 변환
    predicted_letter = label_to_letter[predicted_label]
    all_probabilities = probabilities[0].cpu().numpy()
    
    return predicted_letter, confidence_score, all_probabilities

# 메인 앱
def main():
    st.title("✋ ASL (American Sign Language) 분류기")
    st.markdown("---")
    
    # 모델 로드
    try:
        model, device = load_model()
        st.sidebar.success("✅ 모델이 성공적으로 로드되었습니다!")
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.stop()
    
    # 사이드바
    st.sidebar.header("📋 옵션")
    
    # 이미지 선택 방법
    option = st.sidebar.radio(
        "이미지 선택 방법",
        ["테스트 이미지 사용", "이미지 업로드"]
    )
    
    image = None
    
    if option == "테스트 이미지 사용":
        st.sidebar.subheader("테스트 이미지")
        test_images = {
            "a.png": "./data/asl_image/a.png",
            "b.png": "./data/asl_image/b.png"
        }
        
        selected_test = st.sidebar.selectbox(
            "테스트 이미지 선택",
            list(test_images.keys())
        )
        
        if st.sidebar.button("이미지 로드"):
            image_path = test_images[selected_test]
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                st.sidebar.success(f"✅ {selected_test} 로드 완료")
            else:
                st.sidebar.error(f"❌ 파일을 찾을 수 없습니다: {image_path}")
    
    else:  # 이미지 업로드
        st.sidebar.subheader("이미지 업로드")
        uploaded_file = st.sidebar.file_uploader(
            "ASL 손 모양 이미지를 업로드하세요",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.sidebar.success("✅ 이미지 업로드 완료")
    
    # 메인 영역
    if image is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 입력 이미지")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🔮 예측 결과")
            
            # 예측 수행
            predicted_letter, confidence, probabilities = predict_image(
                image, model, device, label_to_letter
            )
            
            # 결과 표시
            st.markdown(f"### 예측된 문자: **{predicted_letter}**")
            st.markdown(f"### 신뢰도: **{confidence*100:.2f}%**")
            
            # 신뢰도 바
            st.progress(confidence)
            
            # Top 5 예측
            st.markdown("#### Top 5 예측:")
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            
            for i, pred_idx in enumerate(top5_indices):
                letter = label_to_letter[pred_idx]
                prob = probabilities[pred_idx] * 100
                st.markdown(f"{i+1}. **{letter}**: {prob:.2f}%")
        
        # 확률 분포 차트
        st.markdown("---")
        st.subheader("📊 전체 확률 분포")
        
        # plotly가 있으면 사용, 없으면 streamlit 내장 차트 사용
        try:
            import pandas as pd
            import plotly.express as px
            
            # 데이터프레임 생성
            df = pd.DataFrame({
                '문자': [label_to_letter[i] for i in range(24)],
                '확률 (%)': probabilities * 100
            })
            df = df.sort_values('확률 (%)', ascending=False)
            
            # 차트 생성
            fig = px.bar(
                df, 
                x='문자', 
                y='확률 (%)',
                title='각 문자에 대한 예측 확률',
                color='확률 (%)',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                xaxis_title="ASL 문자",
                yaxis_title="확률 (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # plotly가 없으면 streamlit 내장 차트 사용
            import pandas as pd
            
            # 데이터프레임 생성
            df = pd.DataFrame({
                '문자': [label_to_letter[i] for i in range(24)],
                '확률 (%)': probabilities * 100
            })
            df = df.sort_values('확률 (%)', ascending=False)
            
            # Streamlit 내장 bar_chart 사용
            st.bar_chart(df.set_index('문자')['확률 (%)'])
        
    else:
        st.info("👈 사이드바에서 테스트 이미지를 선택하거나 이미지를 업로드하세요.")
        
        # 테스트 이미지 미리보기
        st.markdown("---")
        st.subheader("📁 사용 가능한 테스트 이미지")
        
        col1, col2 = st.columns(2)
        
        test_images = {
            "a.png": "./data/asl_image/a.png",
            "b.png": "./data/asl_image/b.png"
        }
        
        for idx, (name, path) in enumerate(test_images.items()):
            with col1 if idx == 0 else col2:
                if os.path.exists(path):
                    st.image(path, caption=name, use_container_width=True)
                else:
                    st.error(f"파일을 찾을 수 없습니다: {path}")
    
    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>ASL 분류기 | PyTorch nn.Linear 기반 신경망 모델</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

