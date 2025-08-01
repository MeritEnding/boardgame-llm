import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import numpy as np
import datetime
import json
import re
import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="보드게임 기획 AI 서비스",
    description="Spring Boot 메인 서버의 요청에 따라 AI 기능을 제공합니다.",
    version="1.3.0", # 버전 업데이트
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG용 데이터 및 FAISS 인덱스 설정 (이전과 동일)
retriever = None
try:
    df = pd.read_json("./boardgame_detaildata_1-101.json")
    if not df.empty:
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


# --- [수정] 성능 개선 및 한글 출력 강화를 위한 프롬프트 ---

llm_generate = ChatOpenAI(model_name="gpt-4o", temperature=0.8) # 모델 성능 업그레이드 및 창의성 조절
generate_concept_prompt_template = PromptTemplate(
    input_variables=["theme", "playerCount", "averageWeight", "retrieved_games"],
    template="""
# Mission: 당신은 세계 최고의 보드게임 크리에이티브 디렉터입니다. 당신의 임무는 단순한 아이디어를 넘어, 플레이어들에게 잊을 수 없는 경험을 선사할 '살아있는 세계'를 창조하는 것입니다.

## Core Principles for World-Class Game Concepts:
1.  **Theme-Mechanic Harmony (테마와 메커니즘의 조화):** 메커니즘은 테마를 뒷받침해야 합니다. 왜 이 테마에 이 메커니즘이 필수적인지 플레이어가 직관적으로 느껴야 합니다. '우주 탐험' 테마라면 '자원 관리' 메커니즘이 우주선의 한정된 산소나 연료를 표현하는 것처럼요.
2.  **Narrative-Driven Experience (서사 중심의 경험):** 플레이어가 단순한 경쟁자가 아닌, 게임 세계관 속 주인공이 되도록 만드세요. `storyline`은 플레이어의 행동에 동기를 부여하고, `ideaText`는 게임의 핵심적인 드라마와 갈등을 담아야 합니다.
3.  **Vivid & Specific Language (생생하고 구체적인 언어):** 추상적인 표현을 피하세요. '재미있는 상호작용' 대신 '상대방의 유물 카드를 복제하는 '모방' 마법 카드'와 같이 구체적으로 묘사하여 상상력을 자극해야 합니다.

## Input Data Analysis:
- User's Theme: {theme}
- Player Count: {playerCount}
- Target Difficulty: {averageWeight}
- Inspirations from other games: {retrieved_games}

## Output Language Requirement:
- **All generated text for `ideaText`, `mechanics`, `storyline` MUST be in rich, natural KOREAN.** (결과물은 반드시 풍부하고 자연스러운 한국어로 작성되어야 합니다.)

## Final Output Instruction:
이제, 위의 모든 원칙과 요구사항을 따라 아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요. **JSON 코드 블록 외에 다른 설명은 절대 포함하지 마세요.**
```json
{{
    "conceptId": 0,
    "planId": 0,
    "theme": "{theme}",
    "playerCount": "{playerCount}",
    "averageWeight": {averageWeight},
    "ideaText": "[게임의 핵심 플레이 경험과 승리 목표를 서사 중심으로 생생하게 설명 (한국어)]",
    "mechanics": "[핵심 메커니즘들을 나열하고, 각 메커니즘이 테마와 어떻게 유기적으로 연결되는지 구체적으로 설명 (한국어)]",
    "storyline": "[플레이어가 몰입할 수 있는 매력적인 배경 세계관과 그 안에서 플레이어의 역할을 드라마틱하게 설명 (한국어)]",
    "createdAt": " "
}}
```"""
)
concept_generation_chain = LLMChain(llm=llm_generate, prompt=generate_concept_prompt_template)


llm_regenerate = ChatOpenAI(model_name="gpt-4o", temperature=0.9)
regenerate_concept_prompt_template = PromptTemplate(
    input_variables=["original_concept_json", "feedback", "plan_id"],
    template="""
# Mission: 당신은 기존 게임 컨셉에 새로운 생명을 불어넣는 '게임 컨셉 닥터'입니다. 원본 컨셉과 사용자의 피드백을 깊이 있게 분석하여, 컨셉을 한 단계 더 높은 차원으로 발전시키세요.

## Core Principles for Regeneration:
1.  **Interpret the Core Intent (피드백의 핵심 의도 파악):** 사용자의 피드백이 '더 캐주얼하게'라면, 단순히 난이도를 낮추는 것을 넘어 '짧은 플레이 시간', '직관적인 규칙', '더 많은 소셜 요소' 등을 고민해야 합니다.
2.  **Creative Evolution, Not Just Modification (단순 수정을 넘어 창의적 진화):** 피드백을 반영하되, 원본의 매력은 유지하거나 더 나은 방향으로 발전시켜야 합니다. 필요하다면 과감하게 기존 메커니즘을 버리고 새로운 것을 도입하세요.
3.  **Maintain Coherence (개연성 유지):** 수정된 모든 요소(테마, 메커니즘, 스토리)가 서로 유기적으로 연결되어 하나의 완성된 경험을 제공해야 합니다.

## Input Data:
- Original Concept: ```json\n{original_concept_json}\n```
- User's Feedback: {feedback}
- Plan ID to maintain: {plan_id}

## Output Language Requirement:
- **All generated text for `theme`, `ideaText`, `mechanics`, `storyline` MUST be in rich, natural KOREAN.** (결과물은 반드시 풍부하고 자연스러운 한국어로 작성되어야 합니다.)

## Final Output Instruction:
이제, 위의 모든 원칙과 요구사항을 따라 아래 JSON 형식에 맞춰 최종 결과물만을 생성해주세요. **JSON 코드 블록 외에 다른 설명은 절대 포함하지 마세요.**
```json
{{
    "conceptId": 0,
    "planId": {plan_id},
    "theme": "[피드백을 반영하여 수정되거나 완전히 새로워진 테마 (한국어)]",
    "playerCount": "[새로운 컨셉에 가장 적합한 플레이어 수 (한국어)]",
    "averageWeight": [피드백이 반영된 새로운 난이도 (1.0~5.0 사이의 실수)],
    "ideaText": "[피드백이 반영된 새로운 핵심 플레이 경험을 서사 중심으로 생생하게 설명 (한국어)]",
    "mechanics": "[수정된 컨셉의 핵심 메커니즘들을 구체적으로 설명 (한국어)]",
    "storyline": "[새로운 테마와 분위기에 맞는 매력적인 스토리라인 (한국어)]",
    "createdAt": " "
}}
```"""
)
regenerate_concept_chain = LLMChain(llm=llm_regenerate, prompt=regenerate_concept_prompt_template)


# Pydantic 모델 (변경 없음)
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


# API 엔드포인트 (로직 변경 없음)
@app.post("/api/plans/generate-concept", response_model=ConceptResponse, summary="새로운 보드게임 컨셉 생성")
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
        response = concept_generation_chain.invoke({
            "theme": request.theme,
            "playerCount": request.playerCount,
            "averageWeight": request.averageWeight,
            "retrieved_games": retrieved_games_info
        })
        
        match = re.search(r"```json\s*(\{.*?\})\s*```", response['text'], re.DOTALL)
        if not match:
            # 백틱 없이 바로 JSON만 반환하는 경우를 대비한 추가 처리
            try:
                concept = json.loads(response['text'])
            except json.JSONDecodeError:
                 raise ValueError("LLM 응답에서 유효한 JSON을 찾을 수 없습니다.")
        else:
            json_str = match.group(1)
            concept = json.loads(json_str)

        concept["conceptId"] = np.random.randint(1000, 9999)
        concept["planId"] = np.random.randint(1000, 9999)
        concept["createdAt"] = datetime.datetime.now().isoformat(timespec='seconds')
        
        return concept
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 체인 실행 중 오류 발생: {str(e)}")


@app.post("/api/plans/regenerate-concept", response_model=ConceptResponse, summary="기존 보드게임 컨셉 재생성")
async def regenerate_concept_api(request: RegenerateConceptRequest):
    try:
        original_concept_json_str = request.originalConcept.json()
        
        response = regenerate_concept_chain.invoke({
            "original_concept_json": original_concept_json_str,
            "feedback": request.feedback,
            "plan_id": request.originalConcept.planId
        })
        
        match = re.search(r"```json\s*(\{.*?\})\s*```", response['text'], re.DOTALL)
        if not match:
            try:
                concept = json.loads(response['text'])
            except json.JSONDecodeError:
                raise ValueError("LLM 응답에서 유효한 JSON을 찾을 수 없습니다.")
        else:
            json_str = match.group(1)
            concept = json.loads(json_str)

        concept["conceptId"] = np.random.randint(10000, 99999)
        concept["createdAt"] = datetime.datetime.now().isoformat(timespec='seconds')
        
        return concept
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 체인 실행 중 오류 발생: {str(e)}")
