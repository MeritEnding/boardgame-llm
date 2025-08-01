import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import datetime
import json
import re
import numpy as np
import os
from dotenv import load_dotenv

# FastAPI 관련 라이브러리
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정 (환경 변수에서 가져오기)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="보드게임 규칙 생성 및 개선 API",
    description="게임 컨셉/목표 데이터 기반으로 규칙을 생성하고, 피드백을 반영하여 규칙을 개선하는 기능을 제공합니다.",
    version="2.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 1. LLM 공통 설정
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)


# -----------------------------------------------------------------------------
# 2. 게임 규칙 '최초' 생성 기능
# -----------------------------------------------------------------------------

# Pydantic 모델: Spring Boot로부터 받을 데이터 형식 (최초 생성용)
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

# 최초 생성을 위한 프롬프트 템플릿
game_rules_prompt_template = PromptTemplate(
    input_variables=["theme", "ideaText", "mechanics", "mainGoal", "winConditionType"],
    template="""
# Mission: 당신은 복잡한 게임 컨셉을 명확하고 완결성 있는 규칙으로 만드는 '리드 룰(Rule) 디자이너'입니다.
# Input Data Analysis:
- 핵심 컨셉: {ideaText}
- 주요 메커니즘: {mechanics}
- 핵심 목표: {mainGoal} (승리 조건 유형: {winConditionType})
# Final Output Instruction:
아래 JSON 형식에 맞춰 게임의 준비부터 종료까지 모든 과정을 아우르는 완전한 게임 규칙서를 설계해주세요.
**JSON 코드 블록 외에 어떤 추가 텍스트도 포함해서는 안 됩니다.**
```json
{{
  "ruleId": [10000~99999 사이의 임의의 정수 ID],
  "turnStructure": "[한 플레이어의 턴이 어떤 단계(Phase)로 구성되는지 순서대로 설명. 예: 1.자원 수집 -> 2.액션 수행 -> 3.정리]",
  "actionRules": [
    "[플레이어가 턴에 할 수 있는 주요 행동 1에 대한 구체적인 규칙]",
    "[플레이어가 턴에 할 수 있는 주요 행동 2에 대한 구체적인 규칙]"
  ],
  "victoryCondition": "[게임의 최종 승리 조건을 명확하게 서술]",
  "penaltyRules": [
    "[플레이어가 특정 상황에서 받게 되는 페널티 1]"
  ],
  "designNote": "[이 규칙들이 어떻게 게임의 핵심 재미를 만들어내는지에 대한 설계 의도 설명]"
}}
```
    """
)
game_rules_chain = LLMChain(llm=llm, prompt=game_rules_prompt_template)

@app.post("/api/plans/generate-rule")
def generate_rules_api(request: GameRuleGenerationRequest):
    """
    Spring Boot로부터 게임 기획 상세 데이터를 받아 게임 규칙을 '최초'로 생성합니다.
    """
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


# -----------------------------------------------------------------------------
# 3. 게임 규칙 '재생성' (개선) 기능
# -----------------------------------------------------------------------------

# Pydantic 모델: Spring Boot로부터 받을 데이터 형식 (재생성용)
class GameRuleRegenerationRequest(BaseModel):
    # 게임 컨셉 정보
    theme: str
    mechanics: str
    mainGoal: str
    # 개선할 원본 규칙 정보
    original_ruleId: int
    original_turnStructure: str
    original_actionRules: List[str]
    original_victoryCondition: str
    original_penaltyRules: List[str]
    # 사용자 피드백
    feedback: str

# 재생성을 위한 프롬프트 템플릿
regenerate_rules_prompt_template = PromptTemplate(
    input_variables=["game_context", "original_rule_json", "feedback", "rule_id"],
    template="""
# Mission: 당신은 플레이어의 피드백을 반영하여 게임의 깊이를 더하는 '리드 게임 밸런서'입니다.
# Refinement Process:
1.  **피드백 해석:** 사용자의 피드백 '{feedback}'의 근본적인 원인을 파악합니다.
2.  **컨셉 연계:** 주어진 '게임 컨텍스트'의 핵심 메커니즘을 어떻게 더 잘 활용하여 피드백을 해결할지 고민합니다.
3.  **규칙 재설계:** 위의 분석을 바탕으로 'actionRules'를 중심으로 규칙을 재설계하여 더 다양한 플레이 스타일이 가능하도록 만듭니다.

# Input Data:
---
### 1. Game Context:
{game_context}

### 2. Original Rules (To be Improved):
```json
{original_rule_json}
```

### 3. User Feedback to Reflect:
{feedback}
---

# Final Output Instruction:
위 모든 지침을 따라 아래 JSON 형식에 맞춰 '재생성된 전체 규칙'만을 생성해주세요.
**기존 규칙 ID({rule_id})는 그대로 유지해야 합니다.**
**JSON 코드 블록 외에 어떤 추가 텍스트도 포함해서는 안 됩니다.**
```json
{{
  "ruleId": {rule_id},
  "turnStructure": "[개선된 게임 흐름에 맞는 새로운 턴 구조]",
  "actionRules": [
    "[피드백을 반영하여 더 다양하고 전략적인 선택지를 제공하는 행동 규칙 1]",
    "[새롭게 추가되거나 흥미롭게 변경된 행동 규칙 2]"
  ],
  "victoryCondition": "[기존 승리 조건을 유지하되, 더 명확하게 서술]",
  "penaltyRules": [
    "[게임의 복잡도에 맞게 수정되거나 추가된 페널티 규칙]"
  ],
  "designNote": "[피드백을 어떻게 반영했고, 새로운 규칙이 어떻게 게임의 전략적 깊이를 더하는지에 대한 구체적인 설명]"
}}
```
    """
)
regenerate_rules_chain = LLMChain(llm=llm, prompt=regenerate_rules_prompt_template)

@app.post("/api/plans/regenerate-rule")
def regenerate_rules_api(request: GameRuleRegenerationRequest):
    """
    Spring Boot로부터 컨셉, 원본 규칙, 피드백을 받아 게임 규칙을 '재생성'(개선)합니다.
    """
    try:
        # 1. LLM에 전달할 정보 가공
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

        # 2. LLM Chain 호출
        response = regenerate_rules_chain.invoke({
            "game_context": game_context_summary.strip(),
            "original_rule_json": original_rule_json_str,
            "feedback": request.feedback,
            "rule_id": request.original_ruleId
        })

        # 3. LLM 응답 파싱
        response_text = response.get('text', '')
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if not json_match:
            raise ValueError("LLM 응답에서 유효한 JSON 블록을 찾을 수 없습니다.")

        json_str = json_match.group(1)
        regenerated_rules = json.loads(json_str)
        
        # ID가 원본과 동일한지 최종 확인
        regenerated_rules["ruleId"] = request.original_ruleId
        
        return regenerated_rules

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"규칙 재생성 중 오류 발생: {e}")
