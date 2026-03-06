import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 환경 변수 로드
load_dotenv()

class CompanyChatbot:
    def __init__(self, urls=None, index_path="faiss_index_gemini"):
        # [수정] API 키가 잘 로드되었는지 확인하는 디버깅 코드
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # .env에서 못 찾으면 여기서 직접 문자열로 넣어서 테스트 해보세요
            # api_key = "AIzaSy..." 
            raise ValueError("❌ .env 파일에서 OPENAI_API_KEY 찾을 수 없습니다! 파일 위치나 내용을 확인해주세요.")
        
        print(f"[System] API Key 확인됨: {api_key[:5]}*****") # 키 앞부분만 출력해서 확인

        self.index_path = index_path
        
        # [수정] 모델 생성 시 api_key를 명시적으로 넣어줌 (가장 확실한 방법)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        
        # 기존 DB가 있으면 로드, 없으면 새로 크롤링
        if os.path.exists(index_path):
            print(f"[Info] 저장된 데이터베이스('{index_path}')를 불러옵니다.")
            self.vector_db = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            if not urls:
                raise ValueError("학습할 URL이 필요합니다.")
            print(f"[Info] 새로운 데이터를 학습합니다...")
            self.vector_db = self._ingest_data(urls)
            
        self.qa_chain = self._create_chain()

    def _ingest_data(self, urls):
        # A. 데이터 로드
        loader = WebBaseLoader(urls)
        documents = loader.load()
        
        # B. 텍스트 분할 (청크 단위로 쪼개기)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # C. 임베딩 및 벡터 DB 생성
        vector_db = FAISS.from_documents(texts, self.embeddings)
        
        # D. 로컬에 저장 (다음에 재사용하기 위함)
        vector_db.save_local(self.index_path)
        print(f"[Info] 벡터 DB가 '{self.index_path}'에 저장되었습니다.")
        
        return vector_db

    def _create_chain(self):
        # 검색기 설정 (유사도 높은 상위 3개 문맥 참조)
        retriever = self.vector_db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # 시스템 프롬프트 (가드레일 핵심)
        template = """
        당신은 회사 정보를 안내하는 AI 챗봇입니다.
        아래의 [Context]에 있는 정보만 사용하여 질문에 답하십시오.
        
        [Context]:
        {context}

        [Question]:
        {question}

        [Rules]:
        1. 반드시 [Context]에 기반한 사실만 답변하세요.
        2. [Context]에 없는 내용이거나, 회사와 무관한 질문(날씨, 연예인, 잡담 등)이라면,
           절대로 임의로 지어내지 말고, 정확히 아래 문구만 출력하세요:
           "해당 질문과 관련된 사항을 찾을 수 없습니다."
        3. 답변은 사용자의 언어를 감지하여 같은 언어로 정중하게 답변하세요.

        [Answer]:
        """
        
        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # 체인 생성 (gpt-4o 또는 gpt-3.5-turbo 사용)
        # temperature=0 : 창의성을 제거하여 팩트 기반 답변 강화
        chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=os.getenv("OPENAI_API_KEY")),
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain

    def ask(self, query):
        try:
            return self.qa_chain.run(query)
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"