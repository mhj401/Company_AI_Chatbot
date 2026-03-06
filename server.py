from flask import Flask, render_template, request, jsonify
from chatbot import CompanyChatbot

app = Flask(__name__)

# --- [설정] 회사 URL (여기에 시연할 URL 입력) ---
TARGET_URLS = [
    "https://samjungenr.imweb.me/main",
    "https://samjungenr.imweb.me/aboutus",
    "https://samjungenr.imweb.me/resource"
]

print("🤖 AI 로딩 중... (잠시만 기다려주세요)")
# 서버 켤 때 AI을 미리 한 번만 로드합니다.
bot = CompanyChatbot(urls=TARGET_URLS)
print("✅ AI 준비 완료!")

@app.route('/')
def home():
    # 메인 페이지(index.html)를 보여줌
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    # 화면에서 보낸 질문 받기
    user_input = request.json.get('msg')
    
    # AI에게 물어보기
    response = bot.ask(user_input)
    
    # 결과가 딕셔너리({result: ...}) 형태면 텍스트만 추출
    answer_text = response['result'] if isinstance(response, dict) and 'result' in response else str(response)

    # 화면으로 답변 보내기
    return jsonify({'answer': answer_text})

if __name__ == '__main__':
    # app.run(debug=True, port=5000)
    app.run(debug=False, port=5000)