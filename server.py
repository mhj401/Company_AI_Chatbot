from flask import Flask, render_template, request, jsonify
from chatbot import CompanyChatbot
import os

app = Flask(__name__)

# [주의] 기존 faiss_index 폴더를 삭제하고 다시 실행하세요! (데이터 구조가 바뀜)
# 테스트할 URL 리스트
TARGET_URLS = [
    "https://platformoz.com/company",
    "https://platformoz.com/heradee",
    "https://platformoz.com/aixoz",
    "https://platformoz.com/chmo"
]

print("🤖 챗봇 로딩 중...")
# index_path를 바꿔서 새로 크롤링하게 유도하거나, 기존 폴더 삭제 후 실행
bot = CompanyChatbot(urls=TARGET_URLS, index_path="faiss_index_v2")
print("✅ 챗봇 준비 완료!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('msg')
    
    # 챗봇에게 질문 (이제 딕셔너리 {answer, image, link} 반환됨)
    response_data = bot.ask(user_input)
    
    # 프론트엔드로 그대로 전달
    return jsonify(response_data)

if __name__ == '__main__':
    # app.run(debug=True, port=5000)
    app.run(debug=False, port=5000)