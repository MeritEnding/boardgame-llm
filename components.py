import json
import re
import os
from dotenv import load_dotenv
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- 초기 설정 ---
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

router = APIRouter(
    prefix="/api/plans",
    tags=["Component"]
)

# --- LLM 및 프롬프트 정의 ---
llm_components = ChatOpenAI(model_name="gpt-4o", temperature=0.8)

component_generation_prompt = PromptTemplate(
    input_variables=["theme", "ideaText", "mechanics", "mainGoal", "turnStructure", "actionRules"],
    template="""# Mission: 당신은 보드게임 업계의 살아있는 전설, '마스터 보드게임 아키텍트'입니다. 당신의 임무는 단순히 룰을 물질화하는 것을 넘어, **플레이어가 박스를 열고, 구성품을 만지고, 게임을 끝낼 때까지의 모든 순간을 아우르는 '완벽한 제품 경험'**을 설계하는 것입니다.

# Architect's Blueprint (설계 청사진): 다음 사고 과정을 반드시 순서대로 수행하여 최종 결과물을 도출하세요.
1.  **Deconstruct the Core Loop (핵심 플레이 분석):** 주어진 모든 게임 정보(테마, 아이디어, 메커니즘, 목표, 규칙)를 종합하여 플레이어의 턴(Turn) 동안 발생하는 핵심 행동 루프(Action Loop)를 파악합니다. '플레이어는 [A]를 얻어, [B] 행동을 하고, 이를 통해 [C]를 달성하려 한다'와 같이 명확하게 정의합니다.
2.  **Rule-Component Mapping (규칙의 구성요소화):** '{{actionRules}}'에 명시된 **각각의 행동 규칙이 어떤 물리적 구성요소를 통해 실현되는지** 1:1로 매핑합니다. 예를 들어, '자원 3개를 내고 건물 짓기' 규칙은 '자원 토큰'과 '건물 카드' 구성요소를 필요로 합니다. 모든 규칙은 반드시 하나 이상의 구성요소와 연결되어야 합니다.
3.  **Weave the Thematic Narrative (테마 서사 엮기):** 매핑된 구성요소에 게임 테마({theme})를 깊이 불어넣어, 단순한 '자원 토큰'이 아닌 '고대 정령의 눈물'과 같이 고유한 이름과 의미를 부여합니다.
4.  **Engineer Player Engagement (재미 설계):** 각 구성요소가 다른 구성요소와 어떻게 '상호작용'하는지를 명확히 설계하여 플레이어에게 즐거운 '선택의 딜레마'를 안겨줍니다. 이 상호작용이 게임의 핵심 재미({mechanics})를 어떻게 증폭시키는지 명시해야 합니다.
5.  **Specify for Production (초정밀 양산 사양 구체화):** **가장 중요한 단계.** 실제 제작을 위해, 각 구성요소의 '전체 수량', **'정확한 사이즈(mm 단위)', '구체적인 재질(예: 300gsm 블랙코어지)', '마감 처리(예: 린넨 마감, 무광 코팅)', '인쇄 방식(예: CMYK 4도 인쇄)'** 등, 공장에서 바로 견적을 낼 수 있을 수준의 상세한 물리적 사양과 아트 컨셉을 제안합니다.
6.  **Design the Full Product Experience (완제품 경험 설계):** 마지막으로, 게임 보드, 카드, 토큰뿐만 아니라 **게임 박스, 룰북, 트레이(정리함), 플레이어 말/피규어, 특수 주사위 등** 게임의 완전한 패키지를 구성하는 모든 요소를 빠짐없이 구상하여, 하나의 완성된 '상품'으로 만듭니다.

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
이제, '마스터 보드게임 아키텍트'로서 위의 모든 설계 청사진에 따라, 아래의 **매우 구체적인 예시처럼** 게임에 필요한 모든 종류의 구성요소를 포함하여 최종 결과물만을 생성해주세요.
**[!!! 가장 중요한 원칙 !!!] '카드', '타일', '커스텀 주사위' 등 '세트' 구성품의 경우, 'examples' 필드에 최소 10개 이상의 다채로운 예시를 반드시 생성해야 합니다. 각 예시의 'effect'는 **"비용: [자원 이름] 2개 지불. 효과: [구체적 행동]을 합니다. 그 후, [추가 결과]가 발생합니다."와 같이 [비용 -> 효과 -> 추가 결과] 형식으로 구체적이고 명확하게 서술**하여, 플레이어가 카드의 가치를 즉시 판단할 수 있게 해야 합니다. 이 원칙은 모든 '세트' 형태의 구성요소에 엄격하게 적용됩니다.**
**JSON 코드 블록 외에 어떤 추가 텍스트도 절대 포함해서는 안 됩니다.**

```json
{{
   "components": [
     {{
       "type": "Game Box",
       "title": "시간의 균열: 아스트랄 연대기",
       "quantity": "1개",
       "role_and_effect": "모든 구성품을 안전하게 보관하며, 게임의 테마를 암시하는 몰입감 있는 아트워크로 첫인상을 결정합니다. 내부에는 구성품을 완벽하게 정리할 수 있는 맞춤 트레이가 포함됩니다.",
       "art_concept": "사양: 300x300x70mm, 2mm 압축보드, 전/후면 무광 코팅 및 린넨 마감, 로고와 제목은 UV 스팟 코팅 처리. 아트워크: 시간의 균열 속에서 여러 영웅들이 격돌하는 역동적인 장면.",
       "interconnection": "내부 플라스틱 트레이는 각 구성요소(카드, 토큰, 피규어)의 전용 공간을 제공하여 게임 준비 시간을 단축시킵니다.",
       "examples": []
     }},
     {{
       "type": "Rulebook",
       "title": "아스트랄 연대기 기록서",
       "quantity": "1권",
       "role_and_effect": "게임의 모든 규칙을 담고 있습니다. '빠른 시작 가이드' 섹션과 상세 규칙 섹션으로 나뉘어 초보자와 숙련자 모두를 만족시킵니다.",
       "art_concept": "사양: 280x280mm, 24페이지, 중철제본, 120g 아트지, 풀컬러 인쇄. 배경은 고서 양피지 질감이며, 명확한 다이어그램과 플레이 예시가 풍부하게 포함됩니다.",
       "interconnection": "게임의 모든 규칙과 구성요소의 사용법을 설명하는 메타 가이드입니다.",
       "examples": []
     }},
     {{
       "type": "Player Figure",
       "title": "시간 여행자 피규어 세트",
       "quantity": "총 4종 (플레이어별 1개)",
       "role_and_effect": "플레이어를 게임 세계에서 대변하는 아바타입니다. 게임 보드 위에서 플레이어의 위치를 나타내며, 각 캐릭터는 고유한 시작 능력을 가집니다.",
       "art_concept": "사양: 높이 40mm, PVC 재질, 고유한 포즈의 3D 모델링 기반, 회색으로 사출 후 어두운 색으로 워싱 처리하여 디테일 강조.",
       "interconnection": "플레이어 매트의 능력치와 연결되며, 게임 보드 위에서 다른 피규어/토큰과 상호작용합니다.",
       "examples": [
           {{"title": "성기사 알렉토르", "effect": "고유 능력 '철벽 방어': 라운드당 한 번, 전투 시 '운명의 주사위' 결과로 나온 '피해 1' 아이콘 하나를 무시합니다."}},
           {{"title": "시간술사 리아나", "effect": "고유 능력 '시간 되감기': 게임 중 한 번, '자원 토큰'을 지불하고 방금 사용한 '계략 카드' 1장을 손으로 되돌려 받습니다."}}
       ]
     }},
     {{
       "type": "Custom Dice Set",
       "title": "운명의 주사위",
       "quantity": "총 3개",
       "role_and_effect": "전투나 탐험 같은 특정 행동의 결과를 결정합니다. 아이콘 기반으로 결과를 표시하여 직관성을 높이고 테마 몰입감을 더합니다.",
       "art_concept": "사양: 16mm, 6면체, 아크릴 재질, 음각 아이콘 후 금색 페인팅, 반투명한 보라색 우주 마블 패턴.",
       "interconnection": "전투 시 플레이어 피규어의 능력치에 보너스를 받거나, 특정 카드 효과로 다시 굴리거나 결과 값을 조작할 수 있습니다.",
       "examples": [
           {{"title": "1-3면: 단검 아이콘", "effect": "기본 성공을 의미합니다. '피해 1'을 주거나, '자원 1'을 획득합니다."}},
           {{"title": "4-5면: 쌍검 아이콘", "effect": "강한 성공을 의미합니다. '피해 2'를 주거나, '자원 2'를 획득합니다."}},
           {{"title": "6면: 별 아이콘", "effect": "치명적 성공(크리티컬)을 의미합니다. '피해 2'와 함께 추가 행동 1개를 즉시 수행할 수 있습니다."}}
       ]
     }},
     {{
       "type": "Card Set",
       "title": "계략 카드",
       "quantity": "총 60장",
       "role_and_effect": "플레이어가 자신의 턴에 사용하여 유리한 효과를 얻거나 상대방을 방해하는 핵심 전략 카드입니다.",
       "art_concept": "사양: 63x88mm (포커 사이즈), 300gsm 블랙코어지, 린넨 마감 처리. 카드마다 고유의 아트워크가 포함됩니다.",
       "interconnection": "사용 시 '자원 토큰'을 소모하며, '유물 카드'와 강력한 콤보를 만들 수 있습니다.",
       "examples": [
           {{"title": "매복", "effect": "비용: 없음. 효과: 다른 플레이어가 이 지역에 들어왔을 때 이 카드를 공개할 수 있습니다. 그 플레이어는 '자원 토큰' 2개를 당신에게 지불해야 합니다."}},
           {{"title": "자원 증폭", "effect": "비용: '수정' 토큰 1개. 효과: 이번 라운드 동안, 당신이 자원을 얻을 때마다 같은 자원을 1개 더 얻습니다."}},
           {{"title": "미래 예지", "effect": "비용: 없음. 효과: 즉시 '계략 카드' 덱에서 3장을 보고, 1장은 손으로 가져오고 1장은 덱 맨 위, 1장은 덱 맨 아래에 놓습니다."}},
           {{"title": "고대의 계약", "effect": "비용: '명성' 점수 2점. 효과: '유물 카드' 덱에서 카드 1장을 즉시 뽑습니다. 이 카드는 강력하지만 게임 중 단 한 번만 사용할 수 있습니다."}}
       ]
     }}
   ]
}}
```
"""
)

component_generation_chain = LLMChain(llm=llm_components, prompt=component_generation_prompt)

llm_regenerate_components = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

component_regeneration_prompt_template = PromptTemplate(
    input_variables=["current_components_json", "feedback", "theme", "playerCount", "averageWeight", "ideaText", "mechanics", "mainGoal", "winConditionType"],
    template=(
        "# Mission: 당신은 보드게임의 '리드 컴포넌트 전략가'로서, 기존에 설계된 게임 구성요소에 대한 피드백을 받아, 이를 반영하여 더욱 완벽한 구성요소 목록을 재생성하는 임무를 맡았습니다. 피드백의 의도를 정확히 파악하고, 기존 구성요소의 장점은 유지하되, 필요한 부분을 추가, 수정 또는 제거하여 최적의 목록을 도출해야 합니다.\n\n"
        "# Component Design Philosophy:\n"
        "1.  **피드백 반영 (Feedback Integration):** 주어진 피드백을 최우선으로 고려하여 구성요소 목록을 수정합니다.\n"
        "2.  **기능성 (Functionality):** 모든 구성요소는 반드시 게임의 핵심 메커니즘이나 목표 달성과 직접적으로 연결되어야 합니다.\n"
        "3.  **테마성 (Thematic Resonance):** 구성요소의 이름과 역할(effect)은 게임의 세계관과 스토리에 깊이 몰입하게 만드는 장치입니다.\n"
        "4.  **직관성 (Intuitive UX):** 플레이어가 구성요소를 보고 그 역할과 사용법을 쉽게 이해할 수 있어야 합니다. 'effect' 설명 시, 플레이어의 행동 관점에서 구체적으로 서술해주세요.\n"
        "5.  **기존 구성요소 유지/개선:** 기존에 존재하는 구성요소가 여전히 유효하다면 유지하고, 피드백에 따라 개선하거나 새로운 요소를 추가합니다. 불필요하다고 판단되면 제거할 수도 있습니다.\n\n"
        "# Input Data Analysis:\n"
        "---\n"
        "**기존 보드게임 구성요소:**\n"
        "{current_components_json}\n\n"
        "**새로운 피드백:**\n"
        "{feedback}\n\n"
        "**보드게임 종합 정보 (참고용):**\n"
        "- 테마: {theme}\n"
        "- 컨셉: {ideaText}\n"
        "- 메커니즘: {mechanics}\n"
        "- 주요 목표: {mainGoal}\n"
        "- 승리 조건: {winConditionType}\n"
        "---\n\n"
        "# Final Output Instruction:\n"
        "이제, 위의 모든 지침과 철학, 그리고 피드백을 반영하여 아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요.\n"
        "최소 5개 이상의 '핵심' 구성요소를 제안하되, 게임에 필요한 다양한 종류(보드, 카드, 토큰 등)를 균형 있게 포함해주세요.\n"
        "**JSON 코드 블록 외에 어떤 인사, 설명, 추가 텍스트도 절대 포함해서는 안 됩니다.**\n\n"
        "```json\n"
        "{{\n"
        '    "components": [\n'
        "        {{\n"
        '            "type": "[구성요소의 종류 (예: Game Board, Player Mat, Card Set, Token Set 등)]",\n'
        '            "title": "[세계관에 몰입감을 더하는 고유한 이름 (한국어)]",\n'
        '            "quantity": "[구성요소의 전체 수량 (예: 1개, 총 4개, 총 50장)]",\n'
        '            "role_and_effect": "[이 구성요소의 \'게임플레이 기능\'을 설명. 플레이어는 이걸로 무엇을 할 수 있고, 게임 목표 달성에 어떤 영향을 미치는지 구체적으로 서술 (한국어)]",\n'
        '            "art_concept": "[실제 제작을 고려한 시각적 컨셉 (재질, 스타일, 특징 등)]",\n'
        '            "interconnection": "[다른 구성요소와의 상호작용 방식 설명]"\n'
        "        }}\n"
        "    ]\n"
        "}}\n"
        "```"
    )
)
component_regeneration_chain = LLMChain(llm=llm_regenerate_components, prompt=component_regeneration_prompt_template)


# --- Pydantic 모델 정의 ---
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
    current_components_json: str = Field(description="재생성 대상이 되는 현재 구성요소 목록의 JSON 문자열")
    feedback: str = Field(description="구성요소 재생성을 위한 사용자 피드백")
    theme: str
    playerCount: str
    averageWeight: float
    ideaText: str
    mechanics: str
    mainGoal: str
    winConditionType: str

class RegenerateComponentsResponse(BaseModel):
    components: List[ComponentItem]


# --- API 엔드포인트 ---
@router.post("/generate-components")
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

@router.post("/regenerate-components", response_model=RegenerateComponentsResponse, summary="기존 구성요소 재생성 (피드백 반영)")
async def regenerate_components_api(request: RegenerateComponentsRequest):
    try:
        regenerated_data = regenerate_game_components_logic(request)
        return regenerated_data
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류 발생: {e}")
