import traceback
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer

from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)

app = Flask(__name__)

line_bot_api = LineBotApi('channnel_access_token')
handler = WebhookHandler('channel_secret')


# パイプラインの準備
model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment') 
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
nlp = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    SAmessage = talk(event.message.text)
    a = SAmessage['label']
    b = int(SAmessage['score'] * 100)
    if SAmessage['label'] == 'ポジティブ':
        replay_message = f'あなたの発言は{a}で\n点数は{b}点です'
    else:
        replay_message = f'あなたの発言は{a}で\n点数は{100 - b}点です'

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=replay_message))


def talk(word):
    return nlp(word)[0]


if __name__ == "__main__":
    port = int(os.getenv("PORT"))
    app.run(host="0.0.0.0", port=port)
