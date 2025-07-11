# 📝 감성일기 (Emotional Diary )

GPT 기반 자연어 생성 모델을 활용하여 사용자의 감정과 하루의 키워드를 바탕으로 일기를 감성분류하여 일기글에 대해 피드백 및 인지치료 해주는 시스템입니다.  
이 프로젝트는 핵심 기능만 구현한 **MVP(Minimum Viable Product)** 버전입니다.

## 🚀 주요 기능

- 감정 키워드 및 일기글 입력
- GPT 기반 AI 모델로 감성분류 후 피드백
- FastAPI + Uvicorn 기반 REST API 서버
- 날짜별 감성일기 저장 기능 (JSON or DB)
- 간단한 웹 프론트엔드 또는 API 호출로 결과 확인 가능

---

## 🛠️ 기술 스택

- **Language**: Python 3.10+
- **Model**: OpenAI GPT-4o / Hugging Face 모델
- **API server**: FastAPI
- **UI**: Streamlit /
- **Data 저장**: JSON 파일 or PostgreSQL
- **기타**: LangChain (선택), Docker (선택)

---

## 💻 설치 및 실행

```bash
# 1. 레포지토리 클론
git clone https://github.com/your-username/emotional-diary-gpt.git
cd emotional-diary

# 2. 가상환경 설정
python -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate

# 3. 패키지 설치
- backend
cd backend
pip install -r requirements.txt
- frontend
cd frontend
pip install -r requirements.txt

# 4. 실행
- API
uvicorn backend.main:app
- UI
streamlit run frontend/app.py
