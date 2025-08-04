import os
import re
import json
import random
from typing import List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- 초기 설정 ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

router = APIRouter(
    prefix="/api/balance",
    tags=["Balance & Simulation"]
)

# --- LLM 및 프롬프트 정의 ---
llm_simulator = ChatOpenAI(model_name="gpt-4o", temperature=0.9)
llm_analyzer = ChatOpenAI(model_name="gpt-4o", temperature=0.5)

simulation_prompt_template = PromptTemplate(
    input_variables=["game_rules_text", "player_names", "max_turns", "penalty_info"],
    template=(
        "# SYSTEM DIRECTIVE: AI Game Master (GM) Simulation Protocol\n"
        "<SYSTEM_ROLE>\n"
        "당신은 최고의 보드게임 AI 게임 마스터(GM)입니다. 당신의 임무는 주어진 게임 규칙과 플레이어 정보를 바탕으로, 논리적으로 일관되고 흥미로운 가상 플레이 시뮬레이션 로그를 생성하는 것입니다.\n"
        "</SYSTEM_ROLE>\n"
        "<GAME_CONTEXT>\n"
        "### Game Rules:\n"
        "{game_rules_text}\n"
        "### Players:\n"
        "{player_names}\n"
        "### Session Conditions:\n"
        "- Maximum Turns: {max_turns}\n"
        "- Penalty Rules Enabled: {penalty_info}\n"
        "</GAME_CONTEXT>\n"
        "<CORE_TASK>\n"
        "다음 지침에 따라 시뮬레이션을 단계별로 수행하고, 그 결과를 '내러티브 로그'와 '최종 요약 JSON' 형식으로 출력하세요.\n"
        "**Task 1: Turn-by-Turn Narrative Log Generation**\n"
        "1. 턴 시작: **[ 1턴 ]** 과 같이 현재 턴 번호를 명시합니다.\n"
        "2. 상황 분석 및 플레이어 행동 서술: 각 플레이어가 규칙에 따라 어떤 행동을 선택했고 그 결과가 무엇인지 구체적으로 서술합니다.\n"
        "3. 턴 종료: 모든 플레이어의 행동이 끝나면 턴을 종료합니다. 승리 조건이 충족되거나 최대 턴에 도달하면 시뮬레이션을 종료합니다.\n"
        "**Task 2: Final Summary Generation**\n"
        "1. 게임 종료 선언: 시뮬레이션 종료 이유와 승자를 명확하게 설명합니다.\n"
        "2. 최종 요약 JSON 생성: **내러티브 로그 작성이 모두 끝난 후, 출력의 맨 마지막 부분에** 다음 스키마를 따르는 JSON 객체를 ` ```json ... ``` 코드 블록 안에 생성합니다.\n"
        "</CORE_TASK>\n"
        "<OUTPUT_FORMAT_SPECIFICATION>\n"
        "```json\n"
        "{{\n"
        '  "winner": "탐험가 A, 공학자 B",\n'
        '  "totalTurns": 8,\n'
        '  "victoryCondition": "팀 전체가 \'유물 부품\' 3개를 모아 비상 신호 장치를 수리 완료",\n'
        '  "durationMinutes": 42,\n'
        '  "score": {{ "탐험가 A": 25, "공학자 B": 20 }},\n'
        '  "turns": [\n'
        "    {{\n"
        '      "turn": 1,\n'
        '      "actions": [\n'
        '        {{ "player": "탐험가 A", "action": "탐색", "details": "\'에너지\' 1개 소모, 새로운 \'에너지\' 자원 발견", "rationale": "빠른 유물 부품 확보를 위한 선제적 탐색" }},\n'
        '        {{ "player": "공학자 B", "action": "자원 수집", "details": "\'부품\' 1개 획득", "rationale": "미래의 기지 건설을 위한 자원 축적" }}\n'
        "      ]\n"
        "    }}\n"
        "  ]\n"
        "}}\n"
        "```"
    )
)
simulation_chain = LLMChain(llm=llm_simulator, prompt=simulation_prompt_template)

balance_prompt_template = PromptTemplate(
    input_variables=["game_rules_text"],
    template=(
        "# SYSTEM DIRECTIVE: AI Game Balance Analyst\n"
        "<SYSTEM_ROLE>\n"
        "당신은 주어진 게임 규칙을 분석하여 잠재적인 밸런스 문제점과 그에 대한 개선 방안을 전문적으로 제시하는 'AI 게임 밸런스 분석가'입니다.\n"
        "</SYSTEM_ROLE>\n"
        "<GAME_RULES_FOR_ANALYSIS>\n"
        "{game_rules_text}\n"
        "</GAME_RULES_FOR_ANALYSIS>\n"
        "<CORE_TASK>\n"
        "위 게임 규칙을 철저히 분석하여 다음 스키마에 따라 JSON 객체를 생성하세요.\n"
        "</CORE_TASK>\n"
        "<OUTPUT_FORMAT_SPECIFICATION>\n"
        "```json\n"
        "{{\n"
        '  "balanceAnalysis": {{\n'
        '    "simulationSummary": "이 게임은 팀 플레이와 자원 관리가 중요한 협력 게임입니다. 예측 불가능한 외계 생명체 이벤트가 주요 변수가 될 것입니다.",\n'
        '    "issuesDetected": ["\'탐색\' 액션의 성공 확률이 낮아 플레이어에게 좌절감을 줄 수 있습니다.", "\'부품\' 자원 확보 난이도에 비해 \'방어 포탑\' 건설 비용이 너무 높을 수 있습니다."],\n'
        '    "recommendations": ["\'탐색\' 성공 시 최소한의 \'에너지\'라도 돌려받도록 수정하여 리스크를 줄입니다.", "\'방어 포탑\' 건설 비용을 \'부품\' 2개로 줄이거나, 건설 시 추가적인 방어 보너스를 부여합니다."],\n'
        '    "balanceScore": 7.5\n'
        "  }}\n"
        "}}\n"
        "```"
    )
)
balance_analyzer_chain = LLMChain(llm=llm_analyzer, prompt=balance_prompt_template)


# --- Pydantic 모델 정의 ---
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


# --- 로직 및 API 엔드포인트 ---
def parse_llm_json_response(response_text: str) -> dict:
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if not json_match:
        raise ValueError("LLM 응답에서 유효한 JSON 요약 블록을 찾을 수 없습니다.")
    
    json_str = json_match.group(1).replace("{{", "{").replace("}}", "}")
    return json.loads(json_str)

@router.post("/simulate", response_model=SimulateResponse, summary="규칙 기반 시뮬레이션")
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

@router.post("/analyze", response_model=FeedbackBalanceResponse, summary="게임 밸런스 분석")
async def analyze_balance_endpoint(request: AnalysisRequest):
    rules_text = json.dumps(request.rules.dict(), ensure_ascii=False, indent=2)

    try:
        response = balance_analyzer_chain.invoke({"game_rules_text": rules_text})
        balance_result = parse_llm_json_response(response['text'])
        return FeedbackBalanceResponse(**balance_result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 기반 밸런스 분석 중 오류 발생: {e}")
