import os
import re
import json
import random
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 OpenAI API 키를 설정합니다.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# FastAPI 애플리케이션을 초기화합니다.
app = FastAPI(
    title="게임 규칙 시뮬레이션 및 밸런스 분석 전문 API",
    description="전달받은 규칙 정보로 가상 게임 플레이 로그를 생성하고, LLM 기반 게임 밸런스 피드백을 제공합니다.",
    version="2.1.0",
)

# CORS 설정을 추가합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 1. LLM 설정 및 프롬프트 정의
# -----------------------------------------------------------------------------
llm_simulator = ChatOpenAI(model_name="gpt-4o", temperature=0.9)
llm_analyzer = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

# 게임 시뮬레이션 프롬프트 템플릿
simulation_prompt_template = PromptTemplate(
    input_variables=["game_rules_text", "player_names", "max_turns", "penalty_info"],
    template="""
# SYSTEM DIRECTIVE: AI Game Master (GM) Simulation Protocol
<SYSTEM_ROLE>
당신은 최고의 보드게임 AI 게임 마스터(GM)입니다. 당신의 임무는 주어진 게임 규칙과 플레이어 정보를 바탕으로, 논리적으로 일관되고 흥미로운 가상 플레이 시뮬레이션 로그를 생성하는 것입니다.
</SYSTEM_ROLE>
<GAME_CONTEXT>
### Game Rules:
{game_rules_text}
### Players:
{player_names}
### Session Conditions:
- Maximum Turns: {max_turns}
- Penalty Rules Enabled: {penalty_info}
</GAME_CONTEXT>
<CORE_TASK>
다음 지침에 따라 시뮬레이션을 단계별로 수행하고, 그 결과를 '내러티브 로그'와 '최종 요약 JSON' 형식으로 출력하세요.
**Task 1: Turn-by-Turn Narrative Log Generation**
1. 턴 시작: **[ 1턴 ]** 과 같이 현재 턴 번호를 명시합니다.
2. 상황 분석 및 플레이어 행동 서술: 각 플레이어가 규칙에 따라 어떤 행동을 선택했고 그 결과가 무엇인지 구체적으로 서술합니다.
3. 턴 종료: 모든 플레이어의 행동이 끝나면 턴을 종료합니다. 승리 조건이 충족되거나 최대 턴에 도달하면 시뮬레이션을 종료합니다.
**Task 2: Final Summary Generation**
1. 게임 종료 선언: 시뮬레이션 종료 이유와 승자를 명확하게 설명합니다.
2. 최종 요약 JSON 생성: **내러티브 로그 작성이 모두 끝난 후, 출력의 맨 마지막 부분에** 다음 스키마를 따르는 JSON 객체를 ` ```json ... ``` 코드 블록 안에 생성합니다.
</CORE_TASK>
<OUTPUT_FORMAT_SPECIFICATION>
```json
{{
  "winner": "탐험가 A, 공학자 B",
  "totalTurns": 8,
  "victoryCondition": "팀 전체가 '유물 부품' 3개를 모아 비상 신호 장치를 수리 완료",
  "durationMinutes": 42,
  "score": {{ "탐험가 A": 25, "공학자 B": 20 }},
  "turns": [
    {{
      "turn": 1,
      "actions": [
        {{ "player": "탐험가 A", "action": "탐색", "details": "'에너지' 1개 소모, 새로운 '에너지' 자원 발견", "rationale": "빠른 유물 부품 확보를 위한 선제적 탐색" }},
        {{ "player": "공학자 B", "action": "자원 수집", "details": "'부품' 1개 획득", "rationale": "미래의 기지 건설을 위한 자원 축적" }}
      ]
    }}
  ]
}}
```
"""
)
simulation_chain = LLMChain(llm=llm_simulator, prompt=simulation_prompt_template)

# 게임 밸런스 분석 프롬프트 템플릿
balance_prompt_template = PromptTemplate(
    input_variables=["game_rules_text"],
    template="""
# SYSTEM DIRECTIVE: AI Game Balance Analyst
<SYSTEM_ROLE>
당신은 주어진 게임 규칙을 분석하여 잠재적인 밸런스 문제점과 그에 대한 개선 방안을 전문적으로 제시하는 'AI 게임 밸런스 분석가'입니다.
</SYSTEM_ROLE>
<GAME_RULES_FOR_ANALYSIS>
{game_rules_text}
</GAME_RULES_FOR_ANALYSIS>
<CORE_TASK>
위 게임 규칙을 철저히 분석하여 다음 스키마에 따라 JSON 객체를 생성하세요.
</CORE_TASK>
<OUTPUT_FORMAT_SPECIFICATION>
```json
{{
  "balanceAnalysis": {{
    "simulationSummary": "이 게임은 팀 플레이와 자원 관리가 중요한 협력 게임입니다. 예측 불가능한 외계 생명체 이벤트가 주요 변수가 될 것입니다.",
    "issuesDetected": ["'탐색' 액션의 성공 확률이 낮아 플레이어에게 좌절감을 줄 수 있습니다.", "'부품' 자원 확보 난이도에 비해 '방어 포탑' 건설 비용이 너무 높을 수 있습니다."],
    "recommendations": ["'탐색' 성공 시 최소한의 '에너지'라도 돌려받도록 수정하여 리스크를 줄입니다.", "'방어 포탑' 건설 비용을 '부품' 2개로 줄이거나, 건설 시 추가적인 방어 보너스를 부여합니다."],
    "balanceScore": 7.5
  }}
}}
```
"""
)
balance_analyzer_chain = LLMChain(llm=llm_analyzer, prompt=balance_prompt_template)

# -----------------------------------------------------------------------------
# 2. Pydantic 모델 정의
# -----------------------------------------------------------------------------

# --- API 요청(Request) 모델 ---
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

# --- API 응답(Response) 모델 ---
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

# -----------------------------------------------------------------------------
# 3. 로직 및 FastAPI 엔드포인트
# -----------------------------------------------------------------------------

def parse_llm_json_response(response_text: str) -> dict:
    """LLM 응답에서 JSON 코드 블록을 추출하고 파싱합니다."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not json_match:
        raise ValueError("LLM 응답에서 유효한 JSON 요약 블록을 찾을 수 없습니다.")
    
    json_str = json_match.group(1).replace("{{", "{").replace("}}", "}")
    return json.loads(json_str)

@app.post("/api/balance/simulate", response_model=SimulateResponse, summary="규칙 기반 시뮬레이션")
async def simulate_endpoint(request: SimulateRequest):
    rules_text = json.dumps(request.rules.dict(), ensure_ascii=False, indent=2)
    player_names_str = ", ".join(request.playerNames)
    penalty_info_str = "적용됨" if request.enablePenalty else "적용되지 않음"

    try:
        response = simulation_chain.invoke({
            "game_rules_text": rules_text,
            "player_names": player_names_str,
            "max_turns": request.maxTurns,
            "penalty_info": penalty_info_str
        })
        
        sim_result = parse_llm_json_response(response['text'])
        
        final_response = {
            "simulationHistory": [{
                "gameId": request.rules.ruleId,
                "turns": sim_result.get("turns", []),
                "winner": sim_result.get("winner", "N/A"),
                "totalTurns": sim_result.get("totalTurns", 0),
                "victoryCondition": sim_result.get("victoryCondition", "승리 조건 정보 없음"),
                "durationMinutes": sim_result.get("durationMinutes", random.randint(15, 60)),
                "score": sim_result.get("score", {})
            }]
        }
        return final_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {e}")

@app.post("/api/balance/analyze", response_model=FeedbackBalanceResponse, summary="게임 밸런스 분석")
async def analyze_balance_endpoint(request: AnalysisRequest):
    rules_text = json.dumps(request.rules.dict(), ensure_ascii=False, indent=2)

    try:
        response = balance_analyzer_chain.invoke({"game_rules_text": rules_text})
        balance_result = parse_llm_json_response(response['text'])
        return FeedbackBalanceResponse(**balance_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 기반 밸런스 분석 중 오류 발생: {e}")

