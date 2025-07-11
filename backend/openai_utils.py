import openai
import re
from config.config import config
from hf_utils import run_slm_model

openai.api_key = config["openai"]["api_key"]

EMOTION_LABELS = ["기쁨", "슬픔", "분노", "불안", "놀람", "혐오", "평온"]
EMOTION_LABELS_30 = [
    "기쁨 (Joy)",
    "행복 (Happiness)",
    "감사 (Gratitude)",
    "자부심 (Pride)",
    "만족 (Contentment)",
    "설렘 (Excitement)",
    "흥분 (Thrill)",
    "호기심 (Curiosity)",
    "희망 (Hope)",
    "편안함 (Ease)",
    "슬픔 (Sadness)",
    "상실감 (Loss)",
    "후회 (Regret)",
    "외로움 (Loneliness)",
    "그리움 (Nostalgia)",
    "분노 (Anger)",
    "짜증 (Irritation)",
    "원망 (Resentment)",
    "긴장 (Tension)",
    "놀람 (Surprise)",
    "당황 (Embarrassment)",
    "혐오 (Disgust)",
    "미움 (Hatred)",
    "죄책감 (Guilt)",
    "사랑 (Love)",
    "애착 (Affection)",
    "연민 (Compassion)",
    "평온 (Calm)",
    # "희망 (Hope)",  # 중복 있음
    "지루함 (Boredom)"
]

def analyze_diary(diary_text: str, model: str, mood: str = None):
    mood_line = f"[오늘의 기분 상태]\n{mood}\n" if mood else ""
    prompt = f"""
너는 감정 전문가 AI야.
아래 두 정보를 반드시 함께 참고해서 감정 분석을 해줘.

- [오늘의 기분 상태]: 사용자가 오늘 선택한 기분 상태(예: 좋음/보통/나쁨 등)
- [일기]: 사용자가 오늘 하루에 대해 작성한 일기

이 두 정보를 모두 충분히 고려하여, 사용자가 실제로 느꼈을 가능성이 가장 높은 감정을  
아래 감정 라벨 중에서 **정확히 하나** 골라 분류하고, 헷갈리면 3개까지 분류해줘.

감정 라벨 목록:
{', '.join(EMOTION_LABELS_30)}

반드시 아래 [포맷]을 그대로 지켜서 답변해줘.  
(특히 [감정]에는 분류한 감정 라벨을 정확히 써주고,  
 [감정 이유]에는 "오늘의 기분 상태"와 "일기" 내용을 모두 고려해서 감정을 추론한 이유를 써줘.)

=======
[오늘의 기분 상태]
{mood}

[일기]
\"\"\"{diary_text}\"\"\"

[감정]
(위 라벨 목록에서 정확히 하나만, 한 단어만!)

[감정 이유]
(왜 이 감정이 선택됐는지 "오늘의 기분"과 "일기"를 모두 참고해서 한두 문장으로 설명)

[피드백]
(공감과 위로가 담긴 짧은 대화체 조언)

[인지치료적 조언]
(감정에 따라 도움이 될 만한 구체적인 인지치료적 조언)
=======
"""
    if model.startswith("gpt"):
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "너는 감정 분석과 인지치료 피드백을 주는 AI야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=512
        )
        answer = response.choices[0].message.content.strip()
    elif model in ("hyperclovax", "midm", "a.x", "qwen3-4b"):
        answer = run_slm_model(prompt, model)
    else:
        raise ValueError(f"지원하지 않는 모델: {model}")

    # 더 robust한 파싱
    emotion, reason, feedback, therapy = parse_answer(answer)
    # 주요 3개만 리턴 (reason이 필요하면 feedback이나 기타에 합쳐서 사용)
    return emotion, feedback, therapy

def parse_answer(answer):
    """
    [감정], [감정 이유], [피드백], [인지치료적 조언] 태그가
    [태그]\n내용 또는 [태그] 내용 어떤 경우도 robust하게 파싱
    """
    emotion, reason, feedback, therapy = "", "", "", ""
    # ====== 블록 안쪽만 추출
    m = re.search(r"========(.*)========", answer, re.DOTALL)
    if m:
        main = m.group(1)
    else:
        main = answer

    def extract(label):
        # [태그] 내용 (한 줄) 혹은 [태그]\n내용
        p = re.compile(rf"\[{label}\]\s*(.*?)(?=\n\[|$)", re.DOTALL)
        m = p.search(main)
        if not m:
            return ""
        v = m.group(1).strip()
        # 한 줄에 있으면 그대로, 여러 줄이면 줄바꿈->스페이스로
        v = re.sub(r'\n+', ' ', v).strip()
        return v

    emotion  = extract("감정")
    reason   = extract("감정 이유")
    feedback = extract("피드백")
    therapy  = extract("인지치료적 조언")

    # 필수값 중 일부라도 비었으면 fallback
    if not emotion or not feedback or not therapy:
        emotion, feedback, therapy = _fallback_parse(answer)
    return emotion, reason, feedback, therapy

def _fallback_parse(answer):
    # 답변 포맷이 예상과 다를 때 임시로 전체 텍스트 반환
    return "", "", answer, ""

def transcribe_audio(file_path: str):
    with open(file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
            language="ko"
        )
    return transcript
