# 파일: main.py

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
from typing import List, Optional, Literal

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# FastAPI 애플리케이션 초기화
app = FastAPI(
    title="보드게임 구성요소 생성 및 재생성 전문 API",
    description="게임의 컨셉, 목표, 규칙 정보를 종합하여 전문가 수준의 구성요소를 생성하고, 피드백에 따라 재생성합니다.",
    version="2.2.2", # 버전 업데이트
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 1. LLM 설정 및 프롬프트 정의
# =============================================================================

# 구성요소 생성을 위한 LLM과 프롬프트
llm_components = ChatOpenAI(model_name="gpt-4o", temperature=0.8)

component_generation_prompt = PromptTemplate(
    input_variables=["theme", "ideaText", "mechanics", "mainGoal", "turnStructure", "actionRules"],
    template="""
# Mission: 당신은 수십 년 경력의 '마스터 보드게임 아키텍트'입니다. 당신의 임무는 주어진 기획안을 분석하여, 게임의 '재미의 정수(Core Fun)'를 극대화하고 양산까지 고려한 '프로덕션 레벨의 구성요소' 전체 시스템을 설계하는 것입니다.

# Architect's Blueprint (설계 청사진):
1.  **Deconstruct the Core Loop (핵심 플레이 분석):** 주어진 모든 게임 정보(테마, 아이디어, 메커니즘, 목표, 규칙)를 종합하여 플레이어의 턴(Turn) 동안 발생하는 핵심 행동 루프(Action Loop)를 파악합니다.
2.  **Materialize the Mechanics (메커니즘의 물질화):** 핵심 행동 루프의 각 단계를 현실 세계의 '구성요소'로 치환합니다.
3.  **Weave the Thematic Narrative (테마 서사 엮기):** 물질화된 구성요소에 게임 테마({theme})를 깊이 불어넣어 고유한 이름과 의미를 부여합니다.
4.  **Engineer Player Engagement (재미 설계):** 각 구성요소가 다른 구성요소와 어떻게 '상호작용'하는지를 명확히 설계하여 플레이어에게 즐거운 '선택의 딜레마'를 안겨줍니다.
5.  **Specify for Production (양산 사양 구체화):** 실제 제작을 고려하여, 각 구성요소의 '전체 수량', '재질', '아트 컨셉'까지 제안합니다.

# Input Data Analysis:
---
### **보드게임 종합 정보:**
-   **테마:** {theme}
-   **핵심 아이디어:** {ideaText}
-   **주요 메커니즘:** {mechanics}
-   **게임 목표:** {mainGoal}
-   **게임 흐름 (턴 구조):** {turnStructure}
-   **주요 행동 규칙:** {actionRules}
---

# Final Output Instruction:
이제, '마스터 보드게임 아키텍트'로서 위의 모든 설계 청사진에 따라, 아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요.
**JSON 코드 블록 외에 어떤 인사, 설명, 추가 텍스트도 절대 포함해서는 안 됩니다.**
**가장 중요한 원칙: '카드 20장'과 같이 묶어서 표현할 경우, 반드시 'examples' 필드에 그 중 가장 개성 강한 카드 3~5개의 이름과 효과를 각각의 개별 JSON 객체로 생성해야 합니다.** 모든 구성요소는 이 원칙을 따릅니다.

```json
{{
    "components": [
    {{
      "type": "Game Board",
      "title": "시간의 균열이 새겨진 아스트랄 지도",
      "quantity": "1개",
      "role_and_effect": "게임의 주 무대. 총 5개의 대륙과 20개의 지역으로 나뉘며, 각 지역마다 특수 규칙(자원 생산, 몬스터 출현 등)이 존재합니다. '시간의 균열' 칸은 '운명의 두루마리' 카드를 뽑는 장소입니다.",
      "art_concept": "6단으로 접히는 무광 코팅 보드. 고대 양피지 위에 마법 잉크로 그린 듯한 신비로운 스타일로, 주요 지점은 홀로그램 코팅 처리.",
      "interconnection": "'플레이어 말'의 이동 경로를 제공하며, '지역 타일'이 놓이는 공간이 됩니다. '시간의 균 '열' 칸은 '운명의 두루마리' 카드 덱과 직접 상호작용합니다.",
      "examples": []
    }},
    {{
      "type": "Player Mat",
      "title": "영웅의 기록서",
      "quantity": "총 4개 (플레이어 인원수)",
      "role_and_effect": "플레이어의 개인 영역. 생명력, 마나, 경험치를 추적하는 트랙이 있으며, 획득한 '자원 토큰'과 '유물 카드'를 보관하는 공간을 제공합니다. 경험치를 사용해 개인 능력을 해금할 수 있습니다.",
      "art_concept": "두꺼운 카드보드 재질의 개인판. 고서처럼 디자인되었으며, 토큰과 카드를 놓는 자리는 홈이 파여 있어 흔들림을 방지합니다.",
      "interconnection": "플레이어의 모든 자산('토큰', '카드')을 관리하는 허브이며, '플레이어 말'의 능력치와 직접 연동됩니다.",
      "examples": []
    }}
    ]
}}
```
    """
)
component_generation_chain = LLMChain(llm=llm_components, prompt=component_generation_prompt)

# 구성요소 재생성을 위한 LLM과 프롬프트
llm_regenerate_components = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# [수정] 프롬프트의 input_variables와 template에서 불필요한 변수 제거
component_regeneration_prompt_template = PromptTemplate(
    input_variables=["current_components_json", "feedback", "theme", "playerCount", "averageWeight", "ideaText", "mechanics", "mainGoal", "winConditionType"],
    template="""
    # Mission: 당신은 보드게임의 '리드 컴포넌트 전략가'로서, 기존에 설계된 게임 구성요소에 대한 피드백을 받아, 이를 반영하여 더욱 완벽한 구성요소 목록을 재생성하는 임무를 맡았습니다. 피드백의 의도를 정확히 파악하고, 기존 구성요소의 장점은 유지하되, 필요한 부분을 추가, 수정 또는 제거하여 최적의 목록을 도출해야 합니다.

    # Component Design Philosophy:
    1.  **피드백 반영 (Feedback Integration):** 주어진 피드백을 최우선으로 고려하여 구성요소 목록을 수정합니다.
    2.  **기능성 (Functionality):** 모든 구성요소는 반드시 게임의 핵심 메커니즘이나 목표 달성과 직접적으로 연결되어야 합니다.
    3.  **테마성 (Thematic Resonance):** 구성요소의 이름과 역할(effect)은 게임의 세계관과 스토리에 깊이 몰입하게 만드는 장치입니다.
    4.  **직관성 (Intuitive UX):** 플레이어가 구성요소를 보고 그 역할과 사용법을 쉽게 이해할 수 있어야 합니다. 'effect' 설명 시, 플레이어의 행동 관점에서 구체적으로 서술해주세요.
    5.  **기존 구성요소 유지/개선:** 기존에 존재하는 구성요소가 여전히 유효하다면 유지하고, 피드백에 따라 개선하거나 새로운 요소를 추가합니다. 불필요하다고 판단되면 제거할 수도 있습니다.

    # Input Data Analysis:
    ---
    **기존 보드게임 구성요소:**
    {current_components_json}

    **새로운 피드백:**
    {feedback}

    **보드게임 종합 정보 (참고용):**
    - 테마: {theme}
    - 컨셉: {ideaText}
    - 메커니즘: {mechanics}
    - 주요 목표: {mainGoal}
    - 승리 조건: {winConditionType}
    ---

    # Final Output Instruction:
    이제, 위의 모든 지침과 철학, 그리고 피드백을 반영하여 아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요.
    최소 5개 이상의 '핵심' 구성요소를 제안하되, 게임에 필요한 다양한 종류(보드, 카드, 토큰 등)를 균형 있게 포함해주세요.
    **JSON 코드 블록 외에 어떤 인사, 설명, 추가 텍스트도 절대 포함해서는 안 됩니다.**

    ```json
    {{
        "components": [
            {{
                "type": "[구성요소의 종류 (예: Game Board, Player Mat, Card Set, Token Set 등)]",
                "title": "[세계관에 몰입감을 더하는 고유한 이름 (한국어)]",
                "quantity": "[구성요소의 전체 수량 (예: 1개, 총 4개, 총 50장)]",
                "role_and_effect": "[이 구성요소의 '게임플레이 기능'을 설명. 플레이어는 이걸로 무엇을 할 수 있고, 게임 목표 달성에 어떤 영향을 미치는지 구체적으로 서술 (한국어)]",
                "art_concept": "[실제 제작을 고려한 시각적 컨셉 (재질, 스타일, 특징 등)]",
                "interconnection": "[다른 구성요소와의 상호작용 방식 설명]"
            }}
        ]
    }}
    ```
    """
)
component_regeneration_chain = LLMChain(llm=llm_regenerate_components, prompt=component_regeneration_prompt_template)


# =============================================================================
# 2. Pydantic 모델 정의
# =============================================================================

# 구성요소 생성 요청 모델
class ComponentGenerationRequest(BaseModel):
    theme: str
    ideaText: str
    mechanics: str
    mainGoal: str
    turnStructure: str
    actionRules: List[str]

# 구성요소 아이템 모델 (재생성 응답에서도 사용)
class ComponentItem(BaseModel):
    type: str
    title: str
    quantity: str
    role_and_effect: str = Field(alias="role_and_effect")
    art_concept: str = Field(alias="art_concept")
    interconnection: str

# [수정] 구성요소 재생성 요청 모델
class RegenerateComponentsRequest(BaseModel):
    current_components_json: str = Field(description="재생성 대상이 되는 현재 구성요소 목록의 JSON 문자열")
    feedback: str = Field(description="구성요소 재생성을 위한 사용자 피드백")
    
    # Spring Boot에서 전달하는 데이터와 일치
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    mainGoal: str
    winConditionType: str

# 구성요소 재생성 응답 모델
class RegenerateComponentsResponse(BaseModel):
    components: List[ComponentItem]


# =============================================================================
# 3. FastAPI 엔드포인트 정의
# =============================================================================

# 3-1. 구성요소 생성 엔드포인트
@app.post("/api/plans/generate-components")
def generate_components_api(request: ComponentGenerationRequest):
    try:
        response = component_generation_chain.invoke(request.dict())
        response_text = response.get('text', '')
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)

        if not json_match:
            raise ValueError("LLM 응답에서 유효한 JSON 블록을 찾을 수 없습니다.")

        json_str = json_match.group(1)
        components_data = json.loads(json_str)

        return components_data

    except Exception as e:
        print(f"구성요소 생성 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {e}")


# 3-2. 구성요소 재생성 핵심 로직 함수
def regenerate_game_components_logic(request: RegenerateComponentsRequest) -> dict:
    try:
        response = component_regeneration_chain.invoke(request.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 체인 실행 중 오류 발생: {e}")

    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response['text'], re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            components_data = json.loads(json_str)
            return components_data
        else:
            raise ValueError("LLM 응답에서 유효한 JSON 블록을 찾을 수 없습니다.")

    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"LLM 응답 텍스트: {response['text']}")
        raise HTTPException(status_code=500, detail=f"LLM 응답을 JSON 형식으로 파싱할 수 없습니다.")
    except (ValueError, KeyError) as e:
        print(f"오류 발생: {e}")
        print(f"LLM 응답 텍스트: {response['text']}")
        raise HTTPException(status_code=500, detail=str(e))

# 3-3. 구성요소 재생성 엔드포인트
@app.post("/api/plans/regenerate-components", response_model=RegenerateComponentsResponse, summary="기존 구성요소 재생성 (피드백 반영)")
async def regenerate_components_api(request: RegenerateComponentsRequest):
    try:
        regenerated_data = regenerate_game_components_logic(request)
        return regenerated_data
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {e}")
