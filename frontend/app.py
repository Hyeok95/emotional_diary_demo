import streamlit as st
import requests
from config.config import config

st.set_page_config(page_title="감성 일기", page_icon="📓")
st.title("📓 감성 일기")

API_BASE = f"https://{config['backend']['host']}:{config['backend']['port']}"

MODEL_LIST = [
    ("gpt-4o", "GPT-4o"),
    ("gpt-4.1", "GPT-4.1"),
    ("gpt-3.5-turbo", "GPT-3.5 Turbo"),
    ("gpt-4o-mini", "GPT-4o Mini"),
    ("gpt-4.1-nano", "GPT-4.1 Nano"),
    ("a.x", "SKT-A.X"),
    ("midm", "KT-Midm"),
    ("qwen3-4b", "Qwen3"),
]
model_keys = [k for k, v in MODEL_LIST]
model_labels = [v for k, v in MODEL_LIST]
selected_label = st.sidebar.selectbox("사용할 모델을 선택하세요", model_labels, index=0)
selected_model = model_keys[model_labels.index(selected_label)]
st.sidebar.markdown(f"**선택된 모델:** `{selected_model}`")

tab1, tab2 = st.tabs(["텍스트 입력", "마이크로 음성 입력"])

with tab1:
    st.subheader("✍️ 오늘의 일기(텍스트)")
    diary = st.text_area("일기를 입력하세요", height=300)
    if st.button("감정 분석하기", key="text") and diary.strip():
        with st.spinner("분석 중..."):
            resp = requests.post(
                f"{API_BASE}/analyze",
                json={"content": diary, "model": selected_model},
                verify=False
            )
            if resp.status_code == 200:
                data = resp.json()
                st.markdown(f"### 감정: **{data['emotion']}**")
                st.info(data['feedback'])
                st.success(data['therapy'])
            else:
                st.error("분석에 실패했습니다.")

with tab2:
    st.subheader("🎤 오늘의 일기(음성)")
    if "voice_text" not in st.session_state:
        st.session_state.voice_text = ""
    if "voice_analysis" not in st.session_state:
        st.session_state.voice_analysis = None

    audio_bytes = st.audio_input("마이크로 음성을 녹음하세요")
    if audio_bytes is not None:
        st.audio(audio_bytes)
        if st.button("이 음성으로 STT 변환", key="stt_convert"):
            with st.spinner("음성 인식 중..."):
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                resp = requests.post(f"{API_BASE}/stt", files=files, verify=False)
                if resp.status_code == 200:
                    st.session_state.voice_text = resp.json()["text"]
                    st.session_state.voice_analysis = None
                else:
                    st.error("STT 변환에 실패했습니다.")

    if st.session_state.voice_text:
        st.text_area("변환된 텍스트", st.session_state.voice_text, height=200, key="voice_text_area")
        if st.button("이 텍스트로 감정 분석하기", key="analyze_voice_text"):
            with st.spinner("감정 분석 중..."):
                resp = requests.post(
                    f"{API_BASE}/analyze",
                    json={"content": st.session_state.voice_text, "model": selected_model},
                    verify=False
                )
                if resp.status_code == 200:
                    st.session_state.voice_analysis = resp.json()
                else:
                    st.error("감정 분석에 실패했습니다.")

    if st.session_state.voice_analysis is not None:
        data = st.session_state.voice_analysis
        st.markdown(f"### 감정: **{data['emotion']}**")
        st.info(data['feedback'])
        st.success(data['therapy'])
