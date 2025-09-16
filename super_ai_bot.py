# pip install python-telegram-bot transformers torch accelerate sentencepiece

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --- تنظیمات مدل ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("در حال بارگذاری مدل هوش مصنوعی... (اولین بار ممکنه چند دقیقه طول بکشه)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# --- ساخت پایپلاین ---
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    # device=-1  ← حذف شد برای سازگاری با accelerate
)

# --- ذخیره مکالمات بر اساس chat_id ---
user_conversations = {}

def get_conversation(chat_id):
    if chat_id not in user_conversations:
        user_conversations[chat_id] = [
            {"role": "system", "content": "You are a helpful, smart, creative, and funny AI assistant. Answer in Persian. Be detailed and polite."}
        ]
    return user_conversations[chat_id]

def generate_response(chat_id, user_input):
    conversation = get_conversation(chat_id)
    conversation.append({"role": "user", "content": user_input})

    # فرمت متن برای Qwen
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    # تولید پاسخ
    outputs = chatbot(prompt)
    response = outputs[0]['generated_text'][len(prompt):].strip()

    # پاک کردن اضافات
    response = response.split("<|")[0].strip()

    conversation.append({"role": "assistant", "content": response})
    return response

# --- دستورات ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🌟 سلام! من یه ربات هوش مصنوعی همه‌کاره هستم!\n"
        "هر چی دوست داری بپرس: از برنامه‌نویسی تا شعر، از ریاضی تا فلسفه!\n\n"
        "📌 دستورات:\n"
        "/start - شروع مجدد\n"
        "/clear - پاک کردن حافظه چت\n"
        "/help - راهنمای کاربردی"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💡 راهنمای استفاده:\n"
        "- هر سوالی داری بپرس — من جواب می‌دم!\n"
        "- می‌تونی از من بخوای شعر بگه، کد بنویسه، خلاصه کنه، ترجمه کنه و...\n"
        "- اگه جوابم رو فراموش کردم، از /clear استفاده کن.\n"
        "- برای شروع مجدد، /start رو بزن."
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_conversations[chat_id] = [
        {"role": "system", "content": "You are a helpful, smart, creative, and funny AI assistant. Answer in Persian. Be detailed and polite."}
    ]
    await update.message.reply_text("🧠 حافظه چت شما پاک شد!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    chat_id = update.effective_chat.id

    try:
        await update.message.reply_text("در حال تایپ... ⌨️", reply_to_message_id=update.message.id)
        response = generate_response(chat_id, user_input)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"⚠️ خطایی رخ داد: {str(e)}\nلطفاً دوباره امتحان کنید.")

# --- راه‌اندازی ربات ---
def main():
    # 🔐 توکن ربات تلگرامی خودت رو اینجا بذار
    TOKEN = "8496252455:AAEL6KwO7E-Ov4614OF3cwuZUFaY1rsQcdQ"

    if TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("❌ لطفاً توکن ربات تلگرامی خودت رو وارد کن!")
        return

    app = Application.builder().token(TOKEN).build()

    # افزودن هندلرها
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ ربات هوش مصنوعی همه‌کاره با موفقیت در حال اجراست...")
    print("📱 در تلگرام به ربات خودت پیام بده و هر چی دوست داری بپرس!")
    app.run_polling()

if __name__ == "__main__":
    main()