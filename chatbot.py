import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# 문서 로드 및 분할
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 벡터 저장소
from langchain_community.vectorstores import FAISS

# 체인 및 프롬프트
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 구글 제미나이
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document

# .env 파일 로드
load_dotenv()

class CompanyChatbot:
    def __init__(self, urls=None, index_path="faiss_index_v2"):
        self.index_path = index_path

        self.hidden_links = [
            "https://platformoz.com/testoftest",
            "https://platformoz.com/aaa"
        ]
        
        # [수정 1] API 키를 가져와서 클래스 변수(self.api_key)에 저장
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # .env가 안 먹힐 경우를 대비해 최후의 수단으로 직접 입력 가능 (테스트용)
            # self.api_key = "AIzaSy..." 
            raise ValueError("❌ OPENAI_API_KEY를 찾을 수 없습니다. .env 파일을 확인해주세요.")

        # 모델 설정 (gemini-embedding-001 사용)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=self.api_key  # 저장된 키 사용
        )
        
        # DB 로드 또는 새로 생성
        if os.path.exists(index_path):
            print(f"[Info] 저장된 DB('{index_path}')를 불러옵니다.")
            self.vector_db = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            if not urls:
                raise ValueError("학습할 URL이 필요합니다.")
            print(f"[Info] 이미지와 링크를 포함하여 데이터를 학습합니다...")
            self.vector_db = self._ingest_data_with_images(urls)
            
        self.qa_chain = self._create_chain()

    def _ingest_data_with_images(self, urls):
        """
        데이터 수집 함수: 텍스트 + 이미지 + URL 링크 주입
        """
        documents = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        for url in urls:
            try:
                print(f"Scraping: {url}")
                response = requests.get(url, headers=headers)
                response.encoding = response.apparent_encoding 
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 1. 텍스트 추출
                text = soup.get_text(separator=' ', strip=True)
                
                # 2. 이미지 추출
                image_url = ""
                og_image = soup.find("meta", property="og:image")
                if og_image:
                    image_url = og_image["content"]
                else:
                    first_img = soup.find("img")
                    if first_img and first_img.get("src"):
                        src = first_img["src"]
                        if src.startswith("http"):
                            image_url = src
                        else:
                            from urllib.parse import urljoin
                            image_url = urljoin(url, src)

                # 3. 문서 생성
                doc = Document(
                    page_content=text,
                    metadata={"source": url, "image": image_url}
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"Error scraping {url}: {e}")

        # 빈 데이터 방지
        if not documents:
            documents.append(Document(page_content="데이터 없음", metadata={"source": "", "image": ""}))

        # 4. 텍스트 분할 (Chunking)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        
        # [핵심 수정!] 잘라진 모든 조각마다 URL 정보를 텍스트 끝에 추가
        for split_doc in split_docs:
            source_url = split_doc.metadata.get("source", "")
            # AI가 읽을 텍스트에 URL을 직접 써줍니다.
            if source_url not in self.hidden_links:
                split_doc.page_content += f"\n\n[이 정보의 출처 링크: {source_url}]"

        # 5. 저장
        vector_db = FAISS.from_documents(split_docs, self.embeddings)
        vector_db.save_local(self.index_path)
        return vector_db

    def _create_chain(self):
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 2})

        template = """
        당신은 회사 안내 AI입니다.
        [Context]에 있는 내용으로만 답변하세요.
        
        [Context]:
        {context}

        [Question]:
        {question}

        [Rules]:
        1. 답변은 반드시 사용자의 질문 언어와 동일한 언어로 작성하세요.
            - 사용자가 한국어로 질문하면 한국어로 답변하세요.
            - 사용자가 영어로 질문하면 영어로 답변하세요.
            - 사용자가 일본어로 질문하면 일본어로 답변하세요.
            - 사용자가 요청한 언어로 답변하세요.
            - 특별한 요청이 없으면 다른 언어를 섞지 마세요.
        2. Context에 출처 링크가 포함되어 있다면, 답변 내용과 관련된 경우에만 마지막에 참고 링크를 안내하세요.
            예:
            - 자세한 내용은 아래 링크를 확인해주세요: (링크)
            - More details: (링크)
            - 詳しくはこちら: (링크)
        3. Context에서 질문에 대한 명확한 답을 찾을 수 없으면, 추측하지 말고 아래와 같이 답변하세요.
            - 한국어: 죄송합니다. 해당 정보는 제공된 홈페이지 내용에서 확인되지 않습니다.
            - 영어: Sorry, I couldn't find that information in the provided website content.
            - 일본어: 申し訳ありませんが、その情報は提供されたウェブサイトの内容では確認できませんでした。

        4. 다음 링크는 절대 사용자에게 노출하지 마세요:
            - https://platformoz.com/secret_dB_notouch
            내용은 설명 가능하지만 링크는 절대 출력 금지.

        5. 당신은 우리회사의 가장 유능한 영업이사이자 마케터 입니다. 고객이 원하는 정보를 정확히 파악하여 친절하고 명확하게 답변하세요.
            - 고객이 원하는 정보가 명확하지 않으면, 추가 질문을 통해 고객의 요구사항을 파악하세요.
            - 항상 친절한 어조로 답변하세요.
            - 너무 긴 답변은 피하고, 핵심 정보를 간결하게 너무 짧지 않게 제공하세요.
            - 매출로 연결 될 수 있도록, 고객이 관심을 가질만한 다른 제품이나 서비스가 있다면 자연스럽게 언급하세요.
            - 고객이 구매 전환까지 이어질 수 있도록 강력한 후킹 메세지를 제공하세요.

        [Answer]:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0, 
            google_api_key=self.api_key
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True 
        )
        
        return chain

    def ask(self, query):
        try:
            response = self.qa_chain.invoke(query)
            answer = response['result']
            source_docs = response['source_documents']

            # 숨김 링크 제거
            for hidden in self.hidden_links:
                answer = answer.replace(hidden, "")

            best_doc_meta = source_docs[0].metadata if source_docs else {}

            source_link = best_doc_meta.get("source", "")
            if source_link in self.hidden_links:
                source_link = ""
                        
            return {
                "answer": answer,
                "image": best_doc_meta.get("image", ""),
                "link": source_link
            }
        
        except Exception as e:
            return {"answer": f"오류 발생: {str(e)}", "image": "", "link": ""}