import os
import re
import json
import random
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# =======================================================================
# 1. 초기 설정 (환경 변수, FastAPI 앱)
# =======================================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="보드게임 기획 AI 통합 서비스",
    description="컨셉, 목표, 규칙, 구성요소 생성부터 밸런스 테스트 및 기획서 작성까지 모든 AI 기능을 제공합니다.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================================
# 2. LLM 모델 초기화
# =======================================================================
# 용도에 따라 Temperature를 조절하여 다양한 LLM 인스턴스 생성
llm_creative = ChatOpenAI(model_name="gpt-4o", temperature=0.9) # 창의성이 중요한 작업용 (컨셉 생성/재생성, 시뮬레이션)
llm_structured = ChatOpenAI(model_name="gpt-4o", temperature=0.7) # 구조화된 결과가 중요한 작업용 (목표, 규칙, 구성요소, 기획서)
llm_analytical = ChatOpenAI(model_name="gpt-4o", temperature=0.5) # 분석 및 평가용 (밸런스 분석)


# =======================================================================
# 3. RAG (Retrieval-Augmented Generation) 설정
# =======================================================================
retriever = None
try:
    df = pd.read_json("./boardgame_detaildata_1-101.json")
    df_processed = df[['게임ID', '이름', '설명', '최소인원', '최대인원', '난이도', '카테고리', '메커니즘']].copy()
    df_processed.rename(columns={'카테고리': '테마', '최소인원': 'min_players', '최대인원': 'max_players', '난이도': 'difficulty_weight', '메커니즘': 'mechanics_list'}, inplace=True)
    
    embeddings = OpenAIEmbeddings()
    documents = [
        (f"게임 이름: {row['이름']}\n설명: {row['설명']}\n테마: {row['테마']}\n"
         f"플레이 인원: {row['min_players']}~{row['max_players']}명\n난이도: {row['difficulty_weight']:.2f}\n"
         f"메커니즘: {row['mechanics_list']}")
        for index, row in df_processed.iterrows()
    ]
    vectorstore = FAISS.from_texts(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    print("RAG 데이터 및 FAISS 인덱스 로드 완료.")
except Exception as e:
    print(f"Warning: RAG 데이터 파일 또는 FAISS 인덱스 생성 실패. {e}")


# =======================================================================
# 4. Pydantic 모델 정의
# =======================================================================

# --- 컨셉(Concept) 관련 모델 ---
class GenerateConceptRequest(BaseModel):
    theme: str
    playerCount: str
    averageWeight: float

class OriginalConcept(BaseModel):
    conceptId: int
    planId: int
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    storyline: str
    createdAt: str

class RegenerateConceptRequest(BaseModel):
    originalConcept: OriginalConcept
    feedback: str

class ConceptResponse(BaseModel):
    conceptId: int
    planId: int
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    storyline: str
    createdAt: str

# --- 목표(Goal) 관련 모델 ---
class GoalGenerationRequest(BaseModel):
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    storyline: str
    world_setting: str = "{}"
    world_tone: str = ""

class GameObjectiveResponse(BaseModel):
    mainGoal: str
    subGoals: list[str]
    winConditionType: str
    designNote: str

# --- 규칙(Rule) 관련 모델 ---
class GameRuleGenerationRequest(BaseModel):
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    storyline: str
    world_setting: str
    world_tone: str
    mainGoal: str
    subGoals: str
    winConditionType: str
    objective_designNote: str

class GameRuleRegenerationRequest(BaseModel):
    theme: str
    mechanics: str
    mainGoal: str
    original_ruleId: int
    original_turnStructure: str
    original_actionRules: List[str]
    original_victoryCondition: str
    original_penaltyRules: List[str]
    feedback: str

# --- 구성요소(Component) 관련 모델 ---
class ComponentGenerationRequest(BaseModel):
    theme: str
    ideaText: str
    mechanics: str
    mainGoal: str
    turnStructure: str
    actionRules: List[str]

class ComponentItem(BaseModel):
    type: str
    title: str
    quantity: str
    role_and_effect: str = Field(alias="role_and_effect")
    art_concept: str = Field(alias="art_concept")
    interconnection: str

class RegenerateComponentsRequest(BaseModel):
    current_components_json: str
    feedback: str
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    mainGoal: str
    winConditionType: str

class RegenerateComponentsResponse(BaseModel):
    components: List[ComponentItem]

# --- 밸런스(Balance) 관련 모델 ---
class GameRuleDetails(BaseModel):
    ruleId: int
    gameName: str
    turnStructure: str
    actionRules: List[str]
    victoryCondition: str
    penaltyRules: List[str]

class SimulateRequest(BaseModel):
    rules: GameRuleDetails
    playerNames: List[str]
    maxTurns: int
    enablePenalty: bool

class AnalysisRequest(BaseModel):
    rules: GameRuleDetails

class ActionLog(BaseModel):
    player: str
    action: str
    details: str
    rationale: str

class TurnLog(BaseModel):
    turn: int
    actions: List[ActionLog]

class GameSimulationResult(BaseModel):
    gameId: int
    turns: List[TurnLog]
    winner: str
    totalTurns: int
    victoryCondition: str
    durationMinutes: int
    score: Dict[str, int]

class SimulateResponse(BaseModel):
    simulationHistory: List[GameSimulationResult]

class BalanceAnalysis(BaseModel):
    simulationSummary: str
    issuesDetected: List[str]
    recommendations: List[str]
    balanceScore: float

class FeedbackBalanceResponse(BaseModel):
    balanceAnalysis: BalanceAnalysis

# --- 기획서(Summary) 관련 모델 ---
class SummaryConceptInfo(BaseModel):
    theme: Optional[str] = None
    playerCount: Optional[str] = None
    averageWeight: Optional[float] = None
    ideaText: Optional[str] = None
    mechanics: Optional[str] = None
    storyline: Optional[str] = None

class SummaryGoalInfo(BaseModel):
    mainGoal: Optional[str] = None
    subGoals: Optional[List[str]] = []
    winConditionType: Optional[str] = None

class SummaryRuleInfo(BaseModel):
    turnStructure: Optional[str] = None
    actionRules: Optional[List[str]] = []
    penaltyRules: Optional[List[str]] = []

class SummaryComponentInfo(BaseModel):
    type: Optional[str] = None
    title: Optional[str] = None
    quantity: Optional[str] = None
    roleAndEffect: Optional[str] = None
    artConcept: Optional[str] = None

class SummaryRequest(BaseModel):
    conceptInfo: SummaryConceptInfo
    goalInfo: SummaryGoalInfo
    ruleInfo: SummaryRuleInfo
    componentInfo: List[SummaryComponentInfo]


# =======================================================================
# 5. 프롬프트 및 체인 정의
# =======================================================================

# --- 컨셉 생성 프롬프트 ---
generate_concept_prompt = PromptTemplate.from_template(
    """
# Mission: 당신은 세계 최고의 보드게임 크리에이티브 디렉터입니다. 당신의 임무는 단순한 아이디어를 넘어, 플레이어들에게 잊을 수 없는 경험을 선사할 '살아있는 세계'를 창조하는 것입니다.
## Input Data Analysis:
- User's Theme: {theme}
- Player Count: {playerCount}
- Target Difficulty: {averageWeight}
- Inspirations from other games: {retrieved_games}
## Output Language Requirement:
- All generated text for `ideaText`, `mechanics`, `storyline` MUST be in rich, natural KOREAN.
## Final Output Instruction:
아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요. **JSON 코드 블록 외에 다른 설명은 절대 포함하지 마세요.**
```json
{{
    "conceptId": 0, "planId": 0, "theme": "{theme}", "playerCount": "{playerCount}", "averageWeight": {averageWeight},
    "ideaText": "[게임의 핵심 플레이 경험과 승리 목표를 서사 중심으로 생생하게 설명 (한국어)]",
    "mechanics": "[핵심 메커니즘들을 나열하고, 각 메커니즘이 테마와 어떻게 유기적으로 연결되는지 구체적으로 설명 (한국어)]",
    "storyline": "[플레이어가 몰입할 수 있는 매력적인 배경 세계관과 그 안에서 플레이어의 역할을 드라마틱하게 설명 (한국어)]",
    "createdAt": " "
}}
```"""
)
concept_generation_chain = generate_concept_prompt | llm_creative | StrOutputParser()

# --- 목표 생성 프롬프트 ---
game_objective_prompt = PromptTemplate.from_template(
    """# Mission: 당신은 플레이어의 몰입도를 극대화하는 게임 목표를 설계하는 데 특화된 '리드 게임 디자이너'입니다.
# Input Data Analysis:
- 테마: {theme}, 플레이 인원수: {playerCount}, 난이도: {averageWeight}, 핵심 아이디어: {ideaText}, 주요 메커니즘: {mechanics}
# Final Output Instruction:
아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요. **JSON 코드 블록 외에 어떤 인사, 설명, 추가 텍스트도 절대 포함해서는 안 됩니다.**
```json
{{
  "mainGoal": "[게임의 최종 승리 조건을 한 문장으로 명확하게 정의 (한국어)]",
  "subGoals": ["[주요 목표 달성을 돕거나, 점수를 얻을 수 있는 구체적인 보조 목표들 (한국어)]"],
  "winConditionType": "[승리 조건의 핵심 분류. (예: 점수 경쟁형, 목표 달성형)]",
  "designNote": "[이러한 게임 목표가 왜 이 게임에 최적인지에 대한 설계 의도 설명 (한국어)]"
}}
```"""
)
game_objective_chain = game_objective_prompt | llm_structured | StrOutputParser()

# --- 규칙 생성 프롬프트 ---
game_rules_prompt = PromptTemplate.from_template(
    """# Mission: 당신은 복잡한 게임 컨셉을 명확하고 완결성 있는 규칙으로 만드는 '리드 룰(Rule) 디자이너'입니다.
# Input Data Analysis:
- 핵심 컨셉: {ideaText}, 주요 메커니즘: {mechanics}, 핵심 목표: {mainGoal} (승리 조건 유형: {winConditionType})
# Final Output Instruction:
아래 JSON 형식에 맞춰 게임의 준비부터 종료까지 모든 과정을 아우르는 완전한 게임 규칙서를 설계해주세요.
```json
{{
  "ruleId": {random_id},
  "turnStructure": "[한 플레이어의 턴이 어떤 단계(Phase)로 구성되는지 순서대로 설명]",
  "actionRules": ["[플레이어가 턴에 할 수 있는 주요 행동 1에 대한 구체적인 규칙]"],
  "victoryCondition": "[게임의 최종 승리 조건을 명확하게 서술]",
  "penaltyRules": ["[플레이어가 특정 상황에서 받게 되는 페널티 1]"],
  "designNote": "[이 규칙들이 어떻게 게임의 핵심 재미를 만들어내는지에 대한 설계 의도 설명]"
}}
```"""
)
game_rules_chain = game_rules_prompt | llm_structured | StrOutputParser()

# --- 구성요소 생성 프롬프트 ---
component_generation_prompt = PromptTemplate.from_template(
    """# Mission: 당신은 수십 년 경력의 '마스터 보드게임 아키텍트'입니다. 주어진 기획안을 분석하여, 게임의 '재미의 정수(Core Fun)'를 극대화하고 양산까지 고려한 '프로덕션 레벨의 구성요소' 전체 시스템을 설계하는 것입니다.
# Input Data Analysis:
- 테마: {theme}, 핵심 아이디어: {ideaText}, 주요 메커니즘: {mechanics}, 게임 목표: {mainGoal}, 게임 흐름: {turnStructure}, 주요 행동 규칙: {actionRules}
# Final Output Instruction:
아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요.
```json
{{
    "components": [
    {{
      "type": "Game Board", "title": "시간의 균열이 새겨진 아스트랄 지도", "quantity": "1개",
      "role_and_effect": "게임의 주 무대. 총 5개의 대륙과 20개의 지역으로 나뉘며, 각 지역마다 특수 규칙이 존재합니다.",
      "art_concept": "6단으로 접히는 무광 코팅 보드. 고대 양피지 위에 마법 잉크로 그린 듯한 신비로운 스타일.",
      "interconnection": "'플레이어 말'의 이동 경로를 제공하며, '지역 타일'이 놓이는 공간이 됩니다."
    }}
    ]
}}
```"""
)
component_generation_chain = component_generation_prompt | llm_structured | StrOutputParser()

# --- 시뮬레이션 프롬프트 ---
simulation_prompt = PromptTemplate.from_template(
    """# Mission: 당신은 최고의 보드게임 AI 게임 마스터(GM)입니다. 주어진 게임 규칙과 플레이어 정보를 바탕으로, 논리적으로 일관된 가상 플레이 시뮬레이션을 수행하고 그 결과를 JSON 형식으로만 출력하는 것입니다.
# Game Context:
- **Game Rules**: {game_rules_text}
- **Players**: {player_names}
- **Session Conditions**: Maximum Turns: {max_turns}, Penalty Rules Enabled: {penalty_info}
# Task:
최종 결과만을 아래 JSON 스키마에 맞춰 생성해야 합니다. **JSON 코드 블록 외에 어떤 추가 텍스트도 절대 포함해서는 안 됩니다.**
```json
{{
  "winner": "플레이어 A", "totalTurns": 8, "victoryCondition": "유물 부품 3개 수리 완료", "durationMinutes": 42,
  "score": {{ "플레이어 A": 25, "플레이어 B": 20 }},
  "turns": [{{ "turn": 1, "actions": [{{ "player": "플레이어 A", "action": "탐색", "details": "에너지 1 소모", "rationale": "빠른 탐색" }}] }}]
}}
```"""
)
simulation_chain = simulation_prompt | llm_dynamic | StrOutputParser()

# --- 밸런스 분석 프롬프트 ---
balance_prompt = PromptTemplate.from_template(
    """# SYSTEM DIRECTIVE: AI Game Balance Analyst
# GAME_RULES_FOR_ANALYSIS: {game_rules_text}
# CORE_TASK: 위 게임 규칙을 분석하여 다음 스키마에 따라 JSON 객체를 생성하세요.
```json
{{
  "balanceAnalysis": {{
    "simulationSummary": "이 게임은 팀 플레이와 자원 관리가 중요한 협력 게임입니다.",
    "issuesDetected": ["'탐색' 액션의 성공 확률이 낮아 좌절감을 줄 수 있습니다."],
    "recommendations": ["'탐색' 성공 시 최소한의 '에너지'라도 돌려받도록 수정합니다."],
    "balanceScore": 7.5
  }}
}}
```"""
)
balance_analyzer_chain = balance_prompt | llm_analytical | StrOutputParser()

# --- 기획서 생성 프롬프트 ---
summary_prompt = PromptTemplate.from_template(
    """# Mission: 당신은 '수석 게임 기획자'입니다. 주어진 게임의 모든 핵심 데이터를 종합 분석하여, 완성도 높은 '게임 기획서'를 Markdown 형식으로 작성해야 합니다.
# Input Data:
- Concept: {concept_info_str}
- Goal: {goal_info_str}
- Rule: {rule_info_str}
- Components: {components_text}
# Task:
위 모든 정보를 종합하여, 아래 목차 구조에 따라 전문가 수준의 기획서를 작성하세요.
# ==================================================
# 게임 기획서: {theme}
# ==================================================
## 1. 게임 개요
- **테마**: {theme}
- **플레이 인원**: {playerCount}
- **예상 플레이 시간**: {averageWeight}
## 2. 스토리 및 세계관
- {storyline}
## 3. 게임의 목표
- {mainGoal}
## 4. 핵심 게임플레이
- {turnStructure}
## 5. 주요 구성요소
{components_text}
"""
)
summary_chain = summary_prompt | llm_structured | StrOutputParser()


# =======================================================================
# 6. 헬퍼 함수
# =======================================================================
def parse_llm_json_response(response_text: str) -> dict:
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        raise ValueError("LLM 응답에서 JSON 객체를 찾을 수 없습니다.")
    json_string = match.group(0)
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"추출된 문자열의 JSON 파싱에 실패했습니다: {e}")


# =======================================================================
# 7. API 엔드포인트
# =======================================================================

# --- 컨셉(Concept) API ---
@app.post("/api/plans/generate-concept", response_model=ConceptResponse)
async def generate_concept_api(request: GenerateConceptRequest):
    retrieved_games_info = "유사 게임 정보를 찾을 수 없음."
    if retriever:
        try:
            search_query = f"테마: {request.theme}, 플레이 인원: {request.playerCount}, 난이도: {request.averageWeight}"
            docs = retriever.invoke(search_query)
            retrieved_games_info = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Retriever 실행 중 오류: {e}")
    try:
        response_text = concept_generation_chain.invoke({
            "theme": request.theme, "playerCount": request.playerCount,
            "averageWeight": request.averageWeight, "retrieved_games": retrieved_games_info
        })
        concept = parse_llm_json_response(response_text)
        concept["conceptId"] = random.randint(1000, 9999)
        concept["planId"] = random.randint(1000, 9999)
        concept["createdAt"] = datetime.datetime.now().isoformat(timespec='seconds')
        return JSONResponse(content=concept)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 목표(Goal) API ---
@app.post("/api/plans/generate-goal", response_model=GameObjectiveResponse)
async def generate_objective_api(request: GoalGenerationRequest):
    try:
        response_text = game_objective_chain.invoke(request.dict())
        return JSONResponse(content=parse_llm_json_response(response_text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 규칙(Rule) API ---
@app.post("/api/plans/generate-rule")
def generate_rules_api(request: GameRuleGenerationRequest):
    try:
        response_text = game_rules_chain.invoke({
            "ideaText": request.ideaText, "mechanics": request.mechanics,
            "mainGoal": request.mainGoal, "winConditionType": request.winConditionType,
            "random_id": random.randint(10000, 99999)
        })
        return JSONResponse(content=parse_llm_json_response(response_text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"규칙 생성 중 오류 발생: {e}")

# --- 구성요소(Component) API ---
@app.post("/api/plans/generate-components")
def generate_components_api(request: ComponentGenerationRequest):
    try:
        response_text = component_generation_chain.invoke(request.dict())
        return JSONResponse(content=parse_llm_json_response(response_text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"구성요소 생성 중 오류 발생: {e}")

# --- 밸런스(Balance) API ---
@app.post("/api/balance/simulate", response_model=SimulateResponse)
async def simulate_endpoint(request: SimulateRequest):
    rules_text = json.dumps(request.rules.model_dump(), ensure_ascii=False, indent=2)
    player_names_str = ", ".join(request.playerNames)
    penalty_info_str = "적용됨" if request.enablePenalty else "적용되지 않음"
    try:
        response_text = simulation_chain.invoke({
            "game_rules_text": rules_text, "player_names": player_names_str,
            "max_turns": request.maxTurns, "penalty_info": penalty_info_str
        })
        sim_result = parse_llm_json_response(response_text)
        final_response = {
            "simulationHistory": [{
                "gameId": request.rules.ruleId, "turns": sim_result.get("turns", []),
                "winner": sim_result.get("winner", "N/A"), "totalTurns": sim_result.get("totalTurns", 0),
                "victoryCondition": sim_result.get("victoryCondition", "정보 없음"),
                "durationMinutes": sim_result.get("durationMinutes", random.randint(15, 60)),
                "score": sim_result.get("score", {})
            }]
        }
        return JSONResponse(content=final_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {str(e)}")

@app.post("/api/balance/analyze", response_model=FeedbackBalanceResponse)
async def analyze_balance_endpoint(request: AnalysisRequest):
    rules_text = json.dumps(request.rules.model_dump(), ensure_ascii=False, indent=2)
    try:
        response_text = balance_analyzer_chain.invoke({"game_rules_text": rules_text})
        balance_result = parse_llm_json_response(response_text)
        return JSONResponse(content=balance_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 기반 밸런스 분석 중 오류 발생: {str(e)}")

# --- 기획서(Summary) API ---
@app.post("/api/plans/generate-summary")
async def generate_summary_endpoint(request: SummaryRequest):
    try:
        components_text = "\n".join([f"- **{c.title} ({c.type})**: {c.roleAndEffect}" for c in request.componentInfo]) if request.componentInfo else "정의된 구성요소 없음"
        input_data = {
            "concept_info_str": json.dumps(request.conceptInfo.model_dump(), indent=2, ensure_ascii=False),
            "goal_info_str": json.dumps(request.goalInfo.model_dump(), indent=2, ensure_ascii=False),
            "rule_info_str": json.dumps(request.ruleInfo.model_dump(), indent=2, ensure_ascii=False),
            "components_text": components_text,
            "theme": request.conceptInfo.theme or "정의되지 않음",
            "playerCount": request.conceptInfo.playerCount or "정의되지 않음",
            "averageWeight": request.conceptInfo.averageWeight or 0.0,
            "storyline": request.conceptInfo.storyline or "정의되지 않음",
            "mainGoal": request.goalInfo.mainGoal or "정의되지 않음",
            "winConditionType": request.goalInfo.winConditionType or "정의되지 않음",
            "turnStructure": request.ruleInfo.turnStructure or "정의되지 않음",
        }
        result = summary_chain.invoke(input_data)
        return PlainTextResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 문서 생성 중 오류 발생: {e}")

# =======================================================================
# 8. 서버 실행
# =======================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
