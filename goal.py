import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import re
import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="보드게임 기획 AI 서비스",
    description="Spring Boot로부터 컨셉/세계관 데이터를 받아 게임 목표, 구성요소, 규칙 등을 생성합니다.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LLM 및 프롬프트 정의 ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# [요청사항] 제공해주신 프롬프트 템플릿을 그대로 유지합니다.
game_objective_prompt_template = PromptTemplate(
    input_variables=["theme", "playerCount", "averageWeight", "ideaText", "mechanics", "storyline", "world_setting", "world_tone"],
    template="""# Mission: 당신은 플레이어의 몰입도를 극대화하는 게임 목표를 설계하는 데 특화된 '리드 게임 디자이너'입니다. 당신의 임무는 주어진 컨셉과 세계관 정보를 깊이 있게 분석하여, 플레이어에게 강력한 동기를 부여하고 게임의 모든 메커니즘을 유기적으로 활용하게 만드는 핵심 게임 목표를 창조하는 것입니다.

# Core Principles of Objective Design:
1.  **서사적 연결 (Narrative-Driven):** 목표는 세계관의 핵심 갈등을 해결하는 행위여야 합니다. '왜 싸우는가?'에 대한 답을 목표가 제시해야 합니다. 예를 들어, '에테르 크리스탈'을 차지하기 위해 경쟁하는 세계관이라면, 목표는 '가장 많은 크리스탈 조각을 모으거나 중앙 제단을 활성화하는 것'이 될 수 있습니다.
2.  **메커니즘 활용 (Mechanics-Centric):** 설계된 목표는 주어진 메커니즘(예: 지역 점령, 덱 빌딩)을 자연스럽게 사용하도록 유도해야 합니다. 메커니즘이 목표 달성을 위한 '도구'가 되어야 합니다.
3.  **명확성과 긴장감 (Clarity & Tension):** 승리 조건은 명확해야 하지만, 게임이 끝날 때까지 승자를 예측하기 어렵게 만들어 긴장감을 유지해야 합니다. 단 하나의 길이 아닌, 여러 부가 목표를 통해 점수를 획득하는 방식이 좋은 예시입니다.

# Input Data Analysis:
---
**보드게임 컨셉 및 세계관 정보:**
-   테마: {theme}
-   플레이 인원수: {playerCount}
-   난이도: {averageWeight}
-   핵심 아이디어: {ideaText}
-   주요 메커니즘: {mechanics}
-   기존 스토리라인: {storyline}
-   세계관 설정: {world_setting}
-   세계관 분위기: {world_tone}
---

# Final Output Instruction:
이제, 위의 모든 지침과 원칙을 따라 아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요.
**JSON 코드 블록 외에 어떤 인사, 설명, 추가 텍스트도 절대 포함해서는 안 됩니다.**
모든 내용은 **풍부하고 자연스러운 한국어**로 작성되어야 합니다.

```json
{{
  "mainGoal": "[게임의 최종 승리 조건을 한 문장으로 명확하게 정의. 플레이어는 무엇을 달성하면 게임에서 승리하는가? (한국어)]",
  "subGoals": [
    "[주요 목표 달성을 돕거나, 점수를 얻을 수 있는 구체적인 보조 목표들. 플레이어에게 다양한 전략적 선택지를 제공해야 함. (한국어)]",
    "[또 다른 보조 목표 (필요 시 추가). (한국어)]"
  ],
  "winConditionType": "[승리 조건의 핵심 분류. (예: 점수 경쟁형, 목표 달성형, 마지막 생존형, 비밀 임무형)]",
  "designNote": "[이러한 게임 목표가 왜 이 게임에 최적인지에 대한 설계 의도. 어떻게 테마를 강화하고, 플레이어 간의 상호작용을 유도하는지 설명. (한국어)]"
}}
    ```
    """
)
game_objective_chain = LLMChain(llm=llm, prompt=game_objective_prompt_template)


# --- Pydantic 모델 정의 ---
# [수정] Spring Boot에서 모든 데이터를 직접 전달받도록 모델 변경
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
# [수정] 더 이상 내부 데이터를 조회하지 않고, 받은 데이터로 바로 LLM 호출
@app.post("/api/plans/generate-goal", response_model=GameObjectiveResponse, summary="게임 목표 생성")
async def generate_objective_api(request: GoalGenerationRequest):
    try:
        response = game_objective_chain.invoke(request.dict())
        
        match = re.search(r"```json\s*(\{.*?\})\s*```", response['text'], re.DOTALL)
        if not match:
            # LLM이 코드 블록 없이 바로 JSON을 반환하는 경우 대비
            return json.loads(response['text'])
        
        json_str = match.group(1)
        return json.loads(json_str)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 체인 실행 중 오류: {str(e)}")
