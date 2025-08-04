import os
import json
import re
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- 초기 설정 ---
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()
# 환경 변수에서 OpenAI API 키를 설정합니다.
# 이 코드가 작동하려면 프로젝트 루트에 .env 파일이 있고, OPENAI_API_KEY가 설정되어 있어야 합니다.
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# APIRouter 인스턴스를 생성합니다.
router = APIRouter(
    prefix="/api/plans",
    tags=["Summary"]
)

# --- LLM 및 프롬프트 정의 ---
# 이 파일에서 직접 LLM 인스턴스를 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

summary_prompt_template = PromptTemplate(
    input_variables=["game_data_summary"],
    template=(
        "# Mission: 당신은 모든 기획 단계를 종합하여 최종 보드게임 기획서를 완성하는 '총괄 프로듀서'입니다. 당신의 임무는 흩어져 있는 기획 정보들을 하나의 일관되고 매력적인 문서로 통합하는 것입니다.\n\n"
        "# Core Principles for Final Document:\n"
        "1.  **Clarity and Cohesion (명확성과 통일성):** 모든 섹션이 유기적으로 연결되어야 합니다. 컨셉에서 제시된 테마가 구성요소의 이름과 아트 컨셉에 어떻게 반영되었는지 명확히 보여주세요.\n"
        "2.  **Engaging Narrative (매력적인 서사):** 딱딱한 정보 나열이 아닌, 투자자나 퍼블리셔가 이 게임에 흥미를 느낄 수 있도록 스토리텔링 기법을 사용하세요.\n"
        "3.  **Completeness (완결성):** 게임을 이해하는 데 필요한 모든 핵심 정보(컨셉, 목표, 규칙, 구성요소)를 포함해야 합니다.\n\n"
        "# Input Data:\n"
        "---\n"
        "### **종합 보드게임 기획 데이터:**\n"
        "{game_data_summary}\n"
        "---\n\n"
        "# Final Output Instruction:\n"
        "이제, 위의 모든 원칙에 따라, 최종 보드게임 기획서를 **Markdown 형식의 텍스트**로 생성해주세요.\n"
        "각 섹션은 `#`, `##`, `*` 등을 사용하여 명확하게 구분하고, 전체적으로 전문적인 보고서 형태를 갖춰야 합니다.\n"
        "**JSON 형식이 아닌, 완성된 문서 형태의 텍스트만 출력해주세요.**\n\n"
        "## 예시 출력 형식:\n\n"
        "# [게임 제목] 최종 기획서\n\n"
        "## 1. 게임 개요 (Overview)\n"
        "* **테마:**\n"
        "* **플레이어 수:**\n"
        "* **예상 플레이 시간:**\n"
        "* **핵심 경험 (Core Experience):**\n\n"
        "## 2. 게임 목표 (Goal)\n"
        "* **최종 목표:**\n"
        "* **승리 조건:**\n\n"
        "## 3. 게임 규칙 (Rules)\n"
        "* **게임 흐름:**\n"
        "* **주요 액션:**\n\n"
        "## 4. 핵심 구성요소 (Components)\n"
        "* **[구성요소 1 이름]:** [설명]\n"
        "* **[구성요소 2 이름]:** [설명]\n\n"
        "## 5. 디자인 노트 및 매력 포인트\n"
        "* [이 게임이 다른 게임과 차별화되는 지점과 핵심 재미 요소를 요약]\n"
    )
)

summary_chain = LLMChain(llm=llm, prompt=summary_prompt_template)


# --- Pydantic 모델 정의 ---
class ConceptData(BaseModel):
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    storyline: str

class GoalData(BaseModel):
    mainGoal: str
    subGoals: List[str]
    winConditionType: str

class RuleData(BaseModel):
    turnStructure: str
    actionRules: List[str]
    victoryCondition: str

class ComponentItemSummary(BaseModel):
    title: str
    role_and_effect: str

class ComponentData(BaseModel):
    components: List[ComponentItemSummary]

class SummaryGenerationRequest(BaseModel):
    gameName: str = Field(..., description="게임의 최종 이름")
    concept: ConceptData
    goal: GoalData
    rule: RuleData
    components: List[ComponentItemSummary] # ✅ 이렇게 수정!

class SummaryResponse(BaseModel):
    summaryText: str


# --- API 엔드포인트 ---
@router.post("/generate-summary", response_model=SummaryResponse, summary="전체 기획서 요약 생성")
async def generate_summary_api(request: SummaryGenerationRequest):
    try:
        components_list = [c.dict() for c in request.components]

        game_data_summary = f"""
        # 게임 이름: {request.gameName}

        ## 컨셉 정보
        - 테마: {request.concept.theme}
        - 플레이어 수: {request.concept.playerCount}
        - 난이도: {request.concept.averageWeight}
        - 핵심 아이디어: {request.concept.ideaText}
        - 메커니즘: {request.concept.mechanics}
        - 스토리라인: {request.concept.storyline}

        ## 목표 정보
        - 최종 목표: {request.goal.mainGoal}
        - 승리 조건 유형: {request.goal.winConditionType}

        ## 규칙 정보
        - 턴 구조: {request.rule.turnStructure}
        - 승리 조건 상세: {request.rule.victoryCondition}

        ## 구성요소 정보
        {json.dumps(components_list, ensure_ascii=False, indent=2)}
        """

        response = summary_chain.invoke({"game_data_summary": game_data_summary.strip()})
        
        summary_text = response.get('text', '요약 생성에 실패했습니다.')

        return SummaryResponse(summaryText=summary_text)

    except Exception as e:
        print(f"기획서 요약 생성 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류 발생: {str(e)}")
