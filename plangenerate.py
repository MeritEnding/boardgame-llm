import os
import re
import json
import random
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse # PlainTextResponse 임포트
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# FastAPI 애플리케이션을 초기화합니다.
app = FastAPI(
    title="AI 게임 기획서 생성 API",
    description="종합된 게임 데이터를 받아 전문적인 형식의 Markdown 문서로 변환합니다.",
    version="1.2.1", # 버전 업데이트
)

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
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

prompt_template = PromptTemplate.from_template(
    """
# Mission: 당신은 '수석 게임 기획자'입니다. 주어진 게임의 핵심 데이터를 분석하여, 투자자나 팀원에게 즉시 발표할 수 있는 수준의 완성도 높은 '게임 기획서'를 Markdown 형식으로 작성해야 합니다.

# Input Data (For Your Reference):
---
### 1. Concept (컨셉)
{concept_info_str}

### 2. Goal (목표)
{goal_info_str}

### 3. Rule (규칙)
{rule_info_str}

### 4. Components (구성요소)
{components_text}
---

# Task:
위 모든 정보를 종합하여, 아래 목차 구조에 따라 전문가 수준의 기획서를 작성하세요. 각 항목은 명확하고 매력적으로 서술해야 합니다. 단순 정보 나열이 아닌, 각 요소가 어떻게 유기적으로 연결되어 '재미'를 만들어내는지에 초점을 맞춰 설명해주세요.

# ==================================================
# 게임 기획서: {theme}
# ==================================================

## 1. 게임 개요 (Game Overview)

### 1.1. 한 줄 요약
> 이 게임의 핵심 경험을 한 문장으로 요약해주세요. (예: "고대 유물을 찾아 미지의 행성을 탐사하는 전략적 협력 생존 게임")

### 1.2. 게임 정보
- **테마**: {theme}
- **플레이 인원**: {playerCount}
- **예상 플레이 시간**: (게임의 복잡도({averageWeight})와 메커니즘을 고려하여 예상 시간을 제시해주세요. 예: 30~60분)
- **타겟 플레이어**: (이 게임을 가장 즐길만한 플레이어 층을 제시해주세요. 예: 전략 게임 초심자, 친구들과 협력하는 것을 즐기는 플레이어)

## 2. 스토리 및 세계관 (Story & World)
- 제공된 스토리 라인({storyline})을 바탕으로, 플레이어가 게임에 깊이 몰입할 수 있도록 배경 이야기를 풍부하게 각색하여 서술해주세요.

## 3. 게임의 목표 (Game Objective)

### 3.1. 최종 목표 (Main Goal)
- 플레이어가 게임에서 승리하기 위해 달성해야 할 최종 목표({mainGoal})를 명확하고 흥미롭게 설명해주세요.

### 3.2. 승리 조건 (Winning Condition)
- 게임이 어떻게 끝나는지, 승리 조건({winConditionType})을 구체적으로 설명해주세요.

## 4. 핵심 게임플레이 (Core Gameplay)

### 4.1. 게임의 흐름
- 게임의 전체적인 흐름을 턴 구조({turnStructure})에 따라 단계별로 설명해주세요. 플레이어의 턴이 어떻게 시작되고, 어떤 과정을 거쳐 종료되는지 알기 쉽게 서술해야 합니다.

### 4.2. 주요 행동 (Player Actions)
- 플레이어가 자신의 턴에 할 수 있는 주요 행동들({actionRules})을 각각 설명해주세요. 각 행동이 게임에 어떤 영향을 미치는지 구체적으로 서술합니다.

### 4.3. 페널티 및 제약사항
- 게임의 재미와 긴장감을 더하는 페널티 규칙({penaltyRules})이나 제약사항이 있다면 설명해주세요.

## 5. 주요 구성요소 (Key Components)
- 아래 구성요소들이 게임 내에서 어떤 역할을 하며, 어떻게 상호작용하여 재미를 만들어내는지 설명해주세요.
{components_text}

## 6. 게임의 재미 요소 및 기대 효과 (Fun Factors & Outlook)
- 이 게임의 가장 큰 매력과 재미 포인트는 무엇인지 3가지 핵심 키워드로 요약해주세요.
- 이 게임이 성공적으로 개발되었을 때 기대되는 시장 반응이나 플레이어 경험에 대해 긍정적으로 서술하며 문서를 마무리해주세요.
"""
)

chain = prompt_template | llm | StrOutputParser()

# -----------------------------------------------------------------------------
# 2. Pydantic 모델 정의
# -----------------------------------------------------------------------------
class ConceptInfo(BaseModel):
    theme: Optional[str] = None
    playerCount: Optional[str] = None
    averageWeight: Optional[float] = None
    ideaText: Optional[str] = None
    mechanics: Optional[str] = None
    storyline: Optional[str] = None

class GoalInfo(BaseModel):
    mainGoal: Optional[str] = None
    subGoals: Optional[List[str]] = []
    winConditionType: Optional[str] = None

class RuleInfo(BaseModel):
    turnStructure: Optional[str] = None
    actionRules: Optional[List[str]] = []
    penaltyRules: Optional[List[str]] = []

class ComponentInfo(BaseModel):
    type: Optional[str] = None
    title: Optional[str] = None
    quantity: Optional[str] = None
    roleAndEffect: Optional[str] = None
    artConcept: Optional[str] = None

class SummaryRequest(BaseModel):
    conceptInfo: ConceptInfo
    goalInfo: GoalInfo
    ruleInfo: RuleInfo
    componentInfo: List[ComponentInfo]

# -----------------------------------------------------------------------------
# 3. FastAPI 엔드포인트
# -----------------------------------------------------------------------------
@app.post("/api/plans/generate-summary")
async def generate_summary_endpoint(request: SummaryRequest):
    """
    종합된 기획안 데이터를 받아 Markdown 형식의 기획서로 생성하여 반환합니다.
    """
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
            "actionRules": "\n".join([f"- {rule}" for rule in request.ruleInfo.actionRules]) if request.ruleInfo.actionRules else "정의된 행동 규칙 없음",
            "penaltyRules": "\n".join([f"- {rule}" for rule in request.ruleInfo.penaltyRules]) if request.ruleInfo.penaltyRules else "정의된 페널티 규칙 없음",
        }
        
        result = chain.invoke(input_data)
        
        # [수정] response_class=str 대신 PlainTextResponse를 사용하여 안정적으로 문자열을 반환합니다.
        return PlainTextResponse(content=result)

    except Exception as e:
        print(f"Error during document generation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI 문서 생성 중 오류 발생: {e}")

