from telethon.sync import TelegramClient


api_id =  21138037         # 🔁 Replace with your API ID
api_hash = 'f62291f2ec7170893596772793ead07e'    # 🔁 Replace with your API Hash
channel_username = 'injiuniversity'  # ✅ or @Injiuni or t.me/Injiuni (your channel)
client = TelegramClient('injibara_session', api_id, api_hash)

client.start()  # Prompts for phone number and sends code to Telegram app
print("✅ Logged in successfully.")
