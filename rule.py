
import datetime
import json
import re
import os
from dotenv import load_dotenv
from typing import List
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
    tags=["Rule"]
)

# --- LLM 공통 설정 ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# --- 게임 규칙 '최초' 생성 기능 ---
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

game_rules_prompt_template = PromptTemplate(
    # 더 많은 input_variables를 활용하여 풍부한 컨텍스트 제공
    input_variables=[
        "theme", "playerCount", "averageWeight", "ideaText", "mechanics", 
        "storyline", "world_setting", "world_tone", "mainGoal", 
        "subGoals", "winConditionType", "objective_designNote"
    ],
    template=(
        "# Mission: 당신은 플레이어의 몰입감을 최우선으로 생각하는 '마스터 보드게임 디자이너'입니다. 당신의 철학은 '모든 규칙은 테마를 뒷받침해야 한다'입니다. 주어진 컨셉을 바탕으로, 직관적이면서도 전략적인 깊이가 있는 완벽한 게임 규칙서를 창조해주세요.\n\n"
        
        "# Context Analysis: 다음은 게임의 핵심 설계도입니다. 각 요소를 깊이 분석하고 유기적으로 연결하여 규칙을 설계하세요.\n"
        "<Game_Concept>\n"
        "- **Theme & Story**: {theme} ({storyline})\n"
        "- **World**: {world_setting} (Tone: {world_tone})\n"
        "- **Core Idea**: {ideaText}\n"
        "- **Target Players & Weight**: {playerCount}인용, 난이도 {averageWeight}/5.0\n"
        "- **Key Mechanics**: {mechanics}\n"
        "</Game_Concept>\n\n"
        
        "<Game_Objectives>\n"
        "- **Main Goal**: {mainGoal}\n"
        "- **Sub Goals**: {subGoals}\n"
        "- **Victory Condition Type**: {winConditionType}\n"
        "- **Designer's Note on Objectives**: {objective_designNote}\n"
        "</Game_Objectives>\n\n"

        "# Chain of Thought (규칙 설계를 위한 사고 과정): 이 단계를 먼저 머릿속으로 실행한 후, 최종 결과물만 출력하세요.\n"
        "1.  **컨셉 구체화**: 게임의 테마와 스토리를 플레이어가 경험할 핵심 '재미'로 변환한다. 어떤 감정을 느끼게 할 것인가?\n"
        "2.  **메커니즘 연결**: {mechanics} 메커니즘이 {theme} 테마를 어떻게 구현할 수 있을지 구체적인 아이디어를 구상한다. (예: '자원 관리' 메커니즘 -> '마법 세계에서의 마나 관리'로 연결)\n"
        "3.  **게임 흐름 설계**: '게임 준비 -> 플레이어 턴 진행 -> 게임 종료'의 큰 그림을 그린다. 플레이어의 턴은 어떤 단계로 구성되어야 가장 직관적이고 재미있을까?\n"
        "4.  **핵심 액션 정의**: 플레이어가 목표({mainGoal})를 달성하기 위해 할 수 있는 핵심 행동들은 무엇인가? 각 행동의 비용과 결과는 명확한가?\n"
        "5.  **승리 조건 명확화**: {winConditionType}에 맞춰, 게임의 종료를 유발하는 조건과 최종 승리 조건을 구체적이고 명료하게 정의한다.\n\n"

        "# Final Output Instruction: 위의 모든 분석과 사고 과정을 바탕으로, 아래 JSON 형식에 맞춰 '완전한 게임 규칙'만을 생성해주세요. **JSON 코드 블록 외에 어떤 추가 텍스트도 포함해서는 안 됩니다.**\n"
        "```json\n"
        "{{\n"
        '  "ruleId": [10000~99999 사이의 임의의 정수 ID],\n'
        '  "gameOverview": "[게임을 처음 접하는 플레이어를 위한 간략한 테마 소개 및 목표 설명]",\n'
        '  "gameSetup": "[게임 시작을 위해 각 플레이어가 준비해야 할 것들과 게임판(보드)의 초기 상태를 순서대로 설명. 예: 1. 각자 원하는 색상의 말을 고릅니다. 2. 시작 자원으로 5골드를 받습니다.]",\n'
        '  "turnStructure": "[한 플레이어의 턴이 어떤 단계(Phase)로 구성되는지 순서대로 설명. 예: 1.자원 수집 -> 2.액션 수행 -> 3.정리]",\n'
        '  "actionRules": [\n'
        '    "[플레이어가 턴에 할 수 있는 주요 행동 1에 대한 구체적인 규칙 (비용, 효과, 제약 조건 포함)]",\n'
        '    "[플레이어가 턴에 할 수 있는 주요 행동 2에 대한 구체적인 규칙]"\n'
        '  ],\n'
        '  "endOfGameTrigger": "[게임이 즉시 또는 현재 라운드 후에 종료되는 조건을 명확하게 서술. 예: 누군가 승점 10점에 도달하면 그 라운드를 끝으로 게임이 종료됩니다.]",\n'
        '  "victoryCondition": "[게임의 최종 승리 조건을 명확하게 서술. 예: 게임 종료 시, 가장 높은 승점을 가진 플레이어가 승리합니다. 동점일 경우, 남은 자원이 많은 플레이어가 승리합니다.]",\n'
        '  "penaltyRules": [\n'
        '    "[플레이어가 특정 상황에서 받게 되는 페널티 1]"\n'
        '  ],\n'
        '  "designIntent": "[이 규칙들이 어떻게 게임의 핵심 컨셉({ideaText})과 재미({mechanics})를 만들어내는지에 대한 설계 의도 설명]"\n'
        "}}\n"
        "```"
    )
)
game_rules_chain = LLMChain(llm=llm, prompt=game_rules_prompt_template)

@router.post("/generate-rule")
def generate_rules_api(request: GameRuleGenerationRequest):
    try:
        response = game_rules_chain.invoke(request.dict())
        response_text = response.get('text', '')
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)

        if not json_match:
            raise ValueError("LLM 응답에서 유효한 JSON 블록을 찾을 수 없습니다.")

        json_str = json_match.group(1)
        game_rules = json.loads(json_str)

        if not isinstance(game_rules.get("ruleId"), int):
            game_rules["ruleId"] = int(datetime.datetime.now().timestamp())

        return game_rules
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"규칙 생성 중 오류 발생: {e}")


# --- 게임 규칙 '재생성' (개선) 기능 ---
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

regenerate_rules_prompt_template = PromptTemplate(
    input_variables=["game_context", "original_rule_json", "feedback", "rule_id"],
    template=(
        "# Mission: 당신은 플레이어의 피드백을 반영하여 게임의 깊이를 더하는 '리드 게임 밸런서'입니다.\n"
        "# Refinement Process:\n"
        "1.  **피드백 해석:** 사용자의 피드백 '{feedback}'의 근본적인 원인을 파악합니다.\n"
        "2.  **컨셉 연계:** 주어진 '게임 컨텍스트'의 핵심 메커니즘을 어떻게 더 잘 활용하여 피드백을 해결할지 고민합니다.\n"
        "3.  **규칙 재설계:** 위의 분석을 바탕으로 'actionRules'를 중심으로 규칙을 재설계하여 더 다양한 플레이 스타일이 가능하도록 만듭니다.\n\n"
        "# Input Data:\n"
        "---\n"
        "### 1. Game Context:\n"
        "{game_context}\n\n"
        "### 2. Original Rules (To be Improved):\n"
        "```json\n"
        "{original_rule_json}\n"
        "```\n\n"
        "### 3. User Feedback to Reflect:\n"
        "{feedback}\n"
        "---\n\n"
        "# Final Output Instruction:\n"
        "위 모든 지침을 따라 아래 JSON 형식에 맞춰 '재생성된 전체 규칙'만을 생성해주세요.\n"
        "**기존 규칙 ID({rule_id})는 그대로 유지해야 합니다.**\n"
        "**JSON 코드 블록 외에 어떤 추가 텍스트도 포함해서는 안 됩니다.**\n"
        "```json\n"
        "{{\n"
        '  "ruleId": {rule_id},\n'
        '  "turnStructure": "[개선된 게임 흐름에 맞는 새로운 턴 구조]",\n'
        '  "actionRules": [\n'
        '    "[피드백을 반영하여 더 다양하고 전략적인 선택지를 제공하는 행동 규칙 1]",\n'
        '    "[새롭게 추가되거나 흥미롭게 변경된 행동 규칙 2]"\n'
        "  ],\n"
        '  "victoryCondition": "[기존 승리 조건을 유지하되, 더 명확하게 서술]",\n'
        '  "penaltyRules": [\n'
        '    "[게임의 복잡도에 맞게 수정되거나 추가된 페널티 규칙]"\n'
        "  ],\n"
        '  "designNote": "[피드백을 어떻게 반영했고, 새로운 규칙이 어떻게 게임의 전략적 깊이를 더하는지에 대한 구체적인 설명]"\n'
        "}}\n"
        "```"
    )
)
regenerate_rules_chain = LLMChain(llm=llm, prompt=regenerate_rules_prompt_template)

@router.post("/regenerate-rule")
def regenerate_rules_api(request: GameRuleRegenerationRequest):
    try:
        game_context_summary = f"""
        - 테마: {request.theme}
        - 핵심 메커니즘: {request.mechanics}
        - 최종 목표: {request.mainGoal}
        """
        
        original_rule_data = {
            "ruleId": request.original_ruleId,
            "turnStructure": request.original_turnStructure,
            "actionRules": request.original_actionRules,
            "victoryCondition": request.original_victoryCondition,
            "penaltyRules": request.original_penaltyRules,
        }
        original_rule_json_str = json.dumps(original_rule_data, indent=2, ensure_ascii=False)

        response = regenerate_rules_chain.invoke({
            "game_context": game_context_summary.strip(),
            "original_rule_json": original_rule_json_str,
            "feedback": request.feedback,
            "rule_id": request.original_ruleId
        })

        response_text = response.get('text', '')
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("LLM 응답에서 유효한 JSON 블록을 찾을 수 없습니다.")

        json_str = json_match.group(1)
        regenerated_rules = json.loads(json_str)
        
        regenerated_rules["ruleId"] = request.original_ruleId
        
        return regenerated_rules

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"규칙 재생성 중 오류 발생: {e}")
