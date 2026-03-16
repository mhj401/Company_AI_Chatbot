from chatbot import CompanyChatbot

# 1. 학습시킬 URL 정의 (3~4개)
# COMPANY_URLS = [
#     "https://platformoz.com/company",
#     "https://platformoz.com/heradee",
#     "https://platformoz.com/aixoz",
#     "https://platformoz.com/chmo",
#     "https://platformoz.com/secret_dB_notouch"
# ]
COMPANY_URLS = [
    "https://platformoz.com/testoftest",
    "https://platformoz.com/aaa"
]

def main():
    # 처음 실행 시엔 크롤링 수행, 두 번째부터는 저장된 DB 사용
    bot = CompanyChatbot(urls=COMPANY_URLS)
    
    print("\n💬 회사 AI 챗봇이 준비되었습니다. (종료: 'exit')")
    print("-" * 50)

    while True:
        user_input = input("\n질문: ")
        
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break
            
        # 답변 생성 / Frontend에 response 전송
        response = bot.ask(user_input)
        print(f"답변: {response}")

if __name__ == "__main__":
    main()