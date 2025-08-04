import json
import re
import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- 초기 설정 ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

router = APIRouter(
    prefix="/api/plans",
    tags=["Goal"]
)

# --- LLM 및 프롬프트 정의 ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

game_objective_prompt_template = PromptTemplate(
    input_variables=["theme", "playerCount", "averageWeight", "ideaText", "mechanics", "storyline", "world_setting", "world_tone"],
    template=(
        "# Mission: 당신은 플레이어의 동기 부여에 통달한 '목표 설계의 대가'입니다. 당신의 임무는 주어진 컨셉의 영혼을 꿰뚫고, 모든 요소가 하나의 목표를 향해 달려가는 몰입감 넘치는 경험의 청사진을 그리는 것입니다.\n\n"
        
        "# Goal Design Framework: The 'Motivational Bridge'\n"
        "최고의 게임 목표는 세계관의 '핵심 갈등'과 게임의 '핵심 행동(메커니즘)'을 잇는 '동기 부여의 다리(Motivational Bridge)'를 놓는 것과 같습니다.\n"
        "1.  **[1단계: 핵심 갈등 추출]** 주어진 컨셉, 스토리, 세계관 정보를 종합하여 이 세계의 '근본적인 문제' 또는 '가장 중요한 드라마'가 무엇인지 한 문장으로 정의하세요. (예: '고대 신들의 유물이 깨어나 세상의 균형을 위협하고 있다', '오염된 대지를 정화할 마지막 남은 희망의 씨앗을 차지해야 한다.') 이것이 목표의 존재 이유가 됩니다.\n"
        "2.  **[2단계: 메커니즘의 행동화]** 주어진 메커니즘이 플레이어의 어떤 '구체적인 행동'으로 나타나는지 분석하세요. (예: '일꾼 놓기' -> '자원 채집', '영향력 확장', '기술 연구' / '셋 컬렉션' -> '고대 유물 조각 수집', '예언의 파편 조합')\n"
        "3.  **[3단계: 동기 부여의 다리 건설]** '핵심 갈등'을 해결하기 위해 '메커니즘의 행동화'를 어떻게 사용해야 하는지 연결하여 최종 목표를 설계하세요. 플레이어의 모든 행동이 이 위대한 목표를 향한 의미 있는 발걸음이 되어야 합니다.\n\n"

        "# Input Data Analysis:\n"
        "---\n"
        "**보드게임 컨셉 및 세계관 정보:**\n"
        "- 테마: {theme}\n"
        "- 플레이 인원수: {playerCount}\n"
        "- 난이도: {averageWeight}\n"
        "- 핵심 아이디어: {ideaText}\n"
        "- 주요 메커니즘: {mechanics}\n"
        "- 기존 스토리라인: {storyline}\n"
        "- 세계관 설정: {world_setting}\n"
        "- 세계관 분위기: {world_tone}\n"
        "---\n\n"

        "# Final Output Instruction:\n"
        "이제 'Goal Design Framework'에 따라 깊이 있게 사고한 결과를 바탕으로, 아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요.\n"
        "**JSON 코드 블록 외에 어떤 인사, 설명, 추가 텍스트도 절대 포함해서는 안 됩니다.**\n"
        "모든 내용은 **풍부하고 자연스러운 한국어**로 작성되어야 합니다.\n\n"
        "```json\n"
        "{{\n"
        '   "mainGoal": "[1단계에서 정의한 ‘핵심 갈등’을 해결하는, 게임의 가장 중요한 최종 승리 조건을 한 문장으로 명확하게 제시하세요.]",\n'
        '   "subGoals": [\n'
        '       "[주요 목표로 가는 과정에서, 2단계에서 분석한 ‘메커니즘의 행동화’를 활용하여 점수를 얻거나 유리한 위치를 차지할 수 있는 구체적인 보조 목표들을 2~3개 제시하세요.]",\n'
        '       "[이 보조 목표들은 플레이어에게 다양한 전략적 선택지를 제공해야 합니다.]"\n'
        '   ],\n'
        '   "winConditionType": "[승리 조건의 핵심 분류를 제시하세요. (예: 최고 점수 획득형, 특정 목표 선점형, 마지막 생존자형, 공동 목표 달성형, 비밀 임무 완수형)]",\n'
        '   "designNote": "[3단계에서 구축한 ‘동기 부여의 다리’에 대해 설명하세요. 즉, 이 게임 목표가 어떻게 세계관의 핵심 갈등을 플레이어의 행동(메커니즘)을 통해 풀어내는지, 그리고 이것이 왜 플레이어에게 강력한 동기를 부여하는지에 대한 설계자의 핵심 의도를 서술하세요.]"\n'
        "}}\n"
        "```"
    )
)
game_objective_chain = LLMChain(llm=llm, prompt=game_objective_prompt_template)


# --- Pydantic 모델 정의 ---
class GoalGenerationRequest(BaseModel):
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    storyline: str
    world_setting: str
    world_tone: str

class GameObjectiveResponse(BaseModel):
    mainGoal: str
    subGoals: list[str]
    winConditionType: str
    designNote: str


# --- API 엔드포인트 ---
@router.post("/generate-goal", response_model=GameObjectiveResponse, summary="게임 목표 생성")
async def generate_objective_api(request: GoalGenerationRequest):
    try:
        response = game_objective_chain.invoke(request.dict())
        
        match = re.search(r"```json\s*(\{.*?\})\s*```", response['text'], re.DOTALL)
        if not match:
            return json.loads(response['text'])
        
        json_str = match.group(1)
        return json.loads(json_str)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 체인 실행 중 오류: {str(e)}")
