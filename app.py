import streamlit as st
import joblib
import gensim
from gensim import corpora
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import psycopg2
import pandas as pd
from datetime import datetime
import base64
from translate import Translator
import facebook
from telethon.sync import TelegramClient
from telethon.tl.types import PeerChannel
import pandas as pd
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio
from datetime import date
import nest_asyncio
nest_asyncio.apply()
from telethon.tl.types import Message
from sklearn.feature_extraction.text import CountVectorizer
import yake
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

#from fetch_telegram_messages import fetch_telegram_messages
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(pos_percent, neg_percent):
    sender_email = "mareye132@gmail"
    sender_password = "Maru@132zeleke"
    recipient_email = "worknehgashu3@gmail.com"

    subject = "🚨 Sentiment Alert: Threshold Exceeded"
    body = f"""
    Sentiment Summary Alert:

    ✅ Positive Comments: {pos_percent:.2f}%
    ❌ Negative Comments: {neg_percent:.2f}%

    Action may be required if negativity is rising or positivity has maxed out.
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

def translate_to_amharic(text):
    try:
        translator = Translator(to_lang="am")  # "am" is the ISO code for Amharic
        translation = translator.translate(text)
        return translation
    except Exception as e:
        return f"⚠️ Translation failed: {e}"
# Load model
category_model = joblib.load("comment_category_model.pkl")
category_vectorizer = joblib.load("comment_category_vectorizer.pkl")

def categorize_comment(comment_text):
    vec = category_vectorizer.transform([comment_text])
    prediction = category_model.predict(vec)[0]
    return prediction
def extract_keywords_yake(texts, max_keywords=10):
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=max_keywords)
    combined_text = " ".join(texts)
    keywords = kw_extractor.extract_keywords(combined_text)
    return [kw[0] for kw in keywords if kw[1] < 0.5]
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetRepliesRequest
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio

api_id = 21138037
api_hash = 'f62291f2ec7170893596772793ead07e'
channel_username = 'ashara_media'  # or @Injiuni

async def fetch_post_and_comments(limit=5):
    async with TelegramClient('injibara_session', api_id, api_hash) as client:
        try:
            entity = await client.get_entity(channel_username)

            # Step 1: Fetch recent posts
            history = await client(GetHistoryRequest(
                peer=entity,
                limit=limit,
                offset_date=None,
                offset_id=0,
                max_id=0,
                min_id=0,
                add_offset=0,
                hash=0
            ))

            results = []

            for post in history.messages:
                if post.message:
                    post_data = {
                        "post_text": post.message,
                        "comments": []
                    }

                    # Step 2: Fetch replies/comments to the post
                    replies = await client(GetRepliesRequest(
                        peer=entity,
                        msg_id=post.id,
                        offset_id=0,
                        offset_date=None,
                        add_offset=0,
                        limit=20,
                        max_id=0,
                        min_id=0,
                        hash=0
                    ))

                    for reply in replies.messages:
                        if reply.message:
                            post_data["comments"].append(reply.message)

                    results.append(post_data)

            return results

        except Exception as e:
            return [{"error": f"⚠️ Error fetching posts or comments: {e}"}]

def fetch_post_and_comments_sync(limit=5):
    return asyncio.run(fetch_post_and_comments(limit))

def fetch_all_comments():
    try:
        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()
        cur.execute("SELECT sentence FROM sentiment_logs")
        comments = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return comments
    except Exception as e:
        st.error(f"❌ Failed to fetch comments: {e}")
        return []

import requests
post_id = "61578229016788"  # 🔁 Replace this with real post ID
access_token = "598720782912797|102f75b894e9416aa7d2aa08cd315505"


def get_fb_comments(post_id, access_token):
    try:
        url = f"https://graph.facebook.com/v19.0/{post_id}/comments?access_token={access_token}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            comments = [c["message"] for c in data.get("data", []) if "message" in c]
            return comments
        else:
            st.error(f"❌ Facebook API Error {response.status_code}: {response.json().get('error', {}).get('message', '')}")
            return []
    except Exception as e:
        st.error(f"❌ Failed to fetch Facebook comments: {e}")
        return []


if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "user_type" not in st.session_state:
    st.session_state["user_type"] = None
if "last_comment_timestamp" not in st.session_state:
    st.session_state["last_comment_timestamp"] = None

if "new_comment_flag" not in st.session_state:
    st.session_state["new_comment_flag"] = False

# ========== PostgreSQL Functions ==========

def get_user_role(username):
    try:
        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE username = %s", (username,))
        role = cur.fetchone()
        cur.close()
        conn.close()
        return role[0] if role else None
    except Exception as e:
        st.error(f"❌ Error fetching user role: {e}")
        return None


def insert_log(sentence, sentiment, username):
    try:
        now = datetime.now()
        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO sentiment_logs (sentence, sentiment, created_at, username)
            VALUES (%s, %s, %s, %s)
        """, (sentence, sentiment, now, username.strip().lower()))
        conn.commit()
        cur.close()
        conn.close()
        st.session_state["last_comment_timestamp"] = now
        st.session_state["new_comment_flag"] = True  # Set notification
    except Exception as e:
        st.error(f"❌ Error saving log: {e}")

def fetch_logs():
    try:
        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()
        cur.execute("""
            SELECT sl.id, sl.sentence, sl.sentiment, sl.created_at, COALESCE(u.role, 'Unknown') AS role
            FROM sentiment_logs sl
            LEFT JOIN users u ON TRIM(LOWER(sl.username)) = TRIM(LOWER(u.username))
            ORDER BY sl.created_at DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        df = pd.DataFrame(rows, columns=["ID", "Sentence", "Sentiment", "Timestamp", "User Type"])
        return df
    except Exception as e:
        st.error(f"❌ Error fetching logs: {e}")
        return pd.DataFrame()

def delete_log(log_id):
    try:
        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trash (sentence, sentiment, deleted_at)
            SELECT sentence, sentiment, NOW() FROM sentiment_logs WHERE id = %s
        """, (log_id,))
        cur.execute("DELETE FROM sentiment_logs WHERE id = %s", (log_id,))
        conn.commit()
        cur.close()
        conn.close()
        #st.success(f"🗑️ Deleted log ID {log_id} and moved to trash.")
    except Exception as e:
        st.error(f"❌ Error deleting log: {e}")

def fetch_users():
    try:
        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()
        cur.execute("SELECT id, username, role FROM users ORDER BY id ASC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"❌ Error fetching users: {e}")
        return []

def delete_user(user_id):
    try:
        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        cur.close()
        conn.close()
        st.success(f"🗑️ Deleted user ID {user_id}")
    except Exception as e:
        st.error(f"❌ Error deleting user: {e}")

# ========== Load Model and Vectorizer ==========
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ========== Streamlit Config and Styling ==========
st.set_page_config(page_title="Web-based Comment Acceptance and Analyzer", layout="wide")

st.markdown("""
    <style>
        .header-container {
            background-color: #3b5998;
            color: white;
            padding: 10px 20px;
            width: 100vw;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            position: fixed;
            top: 0;
            left: 0;
            z-index: 10000;
        }
        .full-footer {
        background-color: #3b5998;
        color: white;
        width: 100vw;
        text-align: center;
        position: fixed;
        bottom: 0;
        left: 0;
        z-index: 9999;
        padding: 15px 0;
    }
        .header-container img {
            height: 60px;
            margin-right: 18px;
        }
        .stApp > header,
        .stApp > main {
            margin-top: 75px !important;
        }
        h2 {
            font-size: 20px;
            color: white;
            margin: 0;
            padding: 0;
        }
    </style>
""", unsafe_allow_html=True)

with open("logo.png", "rb") as image_file:
    logo_base64 = base64.b64encode(image_file.read()).decode()

st.markdown(f"""
    <div class='header-container'>
        <img src='data:image/png;base64,{logo_base64}' alt='University Logo'>
        <h2>Web-based Comment Acceptance and Analyzer System</h2>
    </div>
""", unsafe_allow_html=True)

# ========== Sidebar ==========

col1, col2 = st.columns([1, 4])

with col1:
    if not st.session_state["logged_in"]:
        st.markdown("### 🔐 Login")
        username_input = st.text_input("Username", key="login_username")
        password_input = st.text_input("Password", type="password", key="login_password")
        login_clicked = st.button("Login")
        if login_clicked:
            if username_input == "admin" and password_input == "Maru@132":
                st.session_state["logged_in"] = True
                st.session_state["username"] = username_input
                st.session_state["user_type"] = "Admin"
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
        st.stop()
    else:
        username = st.session_state["username"]
        user_type = st.session_state["user_type"]
        st.markdown(f"**👤 Logged in as:** `{username}`")
        st.markdown(f"**🧾 Role:** `{user_type}`")

        st.markdown("### 🧭 Menu")
        if user_type == "Admin":
            menu = st.radio("Select Page", ["🏠 Home", "✍️ Analyze Sentiment", "📜 View Logs", "📋 User Management", "📈 Visualization","📊 Report", "📊 Telegram Report","🗂️ Comment Categorization","🌐 Translate", "ℹ️ About Us","🚪 Logout"])
        else:
            menu = st.radio("Select Page", ["🏠 Home", "✍️ Analyze Sentiment", "ℹ️ About Us"])

# ========== Main Content ==========
with col2:
    if menu == "🏠 Home":
        if user_type == "Admin" and st.session_state.get("new_comment_flag"):
            last_time = st.session_state.get("last_comment_timestamp")

            #if last_time:
                #st.info(f"🕒 New comment received at {last_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Custom right-side banner notification
            

        st.subheader("🏠 Welcome")
        st.info(f"🕒 This system helps analyze text sentiment using machine learning and stores results in a PostgreSQL database")

    elif menu == "✍️ Analyze Sentiment":
        analysis_option = st.radio("Select analysis source", ["🧠 Manual", "📢 Telegram Comment Analysis","🔑 Topic Extraction", "📈 Trend Analysis","🧠 Automated Reply Suggestions","🧠 Comment Categorization"])

        if analysis_option == "🧠 Manual":
            st.subheader("✍️ Enter a Sentence for Sentiment Analysis")
            if "manual_sentiment_input" not in st.session_state:
                st.session_state["manual_sentiment_input"] = ""
            if "manual_sentiment_result" not in st.session_state:
                st.session_state["manual_sentiment_result"] = ""
            if "clear_manual_input" not in st.session_state:
                st.session_state["clear_manual_input"] = False

            if st.session_state.clear_manual_input:
                st.session_state.manual_sentiment_input = ""
                st.session_state.manual_sentiment_result = ""
                st.session_state.clear_manual_input = False

            user_input = st.text_area("💬 Type your sentence below:", key="manual_sentiment_input")
            if st.button("🔍 Analyze Sentiment", key="analyze_button"):
                if st.session_state.manual_sentiment_input.strip():
                    input_vector = vectorizer.transform([st.session_state.manual_sentiment_input])
                    prediction = model.predict(input_vector)[0]
                    sentiment = "Positive" if prediction == 1 else "Negative"

                    insert_log(st.session_state.manual_sentiment_input, sentiment, username=username)
                    st.session_state.manual_sentiment_result = f"✅ Sentiment: **{sentiment}**"
                    st.session_state.clear_manual_input = True  # clear on next run
                else:
                    st.warning("⚠️ Please enter a sentence.")

            # Show result (if any)
            if st.session_state.manual_sentiment_result:
                st.success(st.session_state.manual_sentiment_result)

            # Initialize session state variables
        
        # Input box
        #user_input = st.text_area("💬 Type your sentence below:", key="manual_sentiment_input")

        # Analyze button
        
           


    
        elif analysis_option == "📘 Facebook Comment Analysis":
            st.subheader("📘 Facebook Post Comment Sentiment")

            post_url = st.text_input("📌 Paste Facebook Post URL")
            access_token = st.text_input("🔑 Facebook App Token", type="password", value="598720782912797|102f75b894e9416aa7d2aa08cd315505")

            if st.button("📥 Analyze Facebook Comments"):
                if post_url and access_token:
                    # Extract post ID from the URL
                    try:
                        if "posts/" in post_url:
                            post_id = post_url.split("posts/")[1].split("/")[0].split("?")[0]
                        elif "/pfbid" in post_url:
                            st.warning("⚠️ This is a shortlink. Try using full numeric ID from Graph API.")
                            st.stop()
                        else:
                            st.error("❌ Couldn't extract Post ID from URL.")
                            st.stop()
                    except Exception:
                        st.error("❌ Invalid URL structure.")
                        st.stop()

                    with st.spinner("Fetching and analyzing comments..."):
                        comments = get_fb_comments(post_id, access_token)

                        if comments:
                            st.success(f"✅ Retrieved {len(comments)} comments.")
                            positive, negative = 0, 0
                            for comment in comments:
                                input_vector = vectorizer.transform([comment])
                                prediction = model.predict(input_vector)[0]
                                sentiment = "Positive" if prediction == 1 else "Negative"
                                if sentiment == "Positive":
                                    positive += 1
                                else:
                                    negative += 1
                                st.markdown(f"- 💬 `{comment}` → :{'green' if sentiment == 'Positive' else 'red'}[**{sentiment}**]")

                            # Pie chart
                            import matplotlib.pyplot as plt
                            st.subheader("📊 Sentiment Distribution")
                            fig, ax = plt.subplots()
                            ax.pie([positive, negative], labels=["Positive", "Negative"],
                                autopct="%1.1f%%", colors=["#4CAF50", "#F44336"], startangle=90)
                            ax.axis("equal")
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ No comments retrieved.")
                else:
                    st.warning("Please provide both the post URL and access token.")

        elif analysis_option == "📢 Telegram Comment Analysis":
            st.subheader("📨 Telegram Comment Sentiment Analysis")
            if st.button("🔍 Fetch and Analyze Telegram Comments"):
                today = date.today()
                positive_count = 0
                negative_count = 0

                try:
                    messages = asyncio.run(fetch_post_and_comments(limit=5))

                    if messages and not messages[0].get("error"):
                        # Connect to DB
                        conn = psycopg2.connect(dbname="Sentiment", user="postgres", password="Maru@132", host="localhost", port="5432")
                        cur = conn.cursor()
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS daily_telegram_sentiments (
                                id SERIAL PRIMARY KEY,
                                sentiment_date DATE UNIQUE,
                                positive_count INT,
                                negative_count INT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            );
                        """)

                        st.subheader("📝 Telegram Posts and Comment Sentiment")

                        for post in messages:
                            st.markdown("### 📌 Post")
                            st.write(post["post_text"])

                            if post["comments"]:
                                st.markdown("#### 💬 Comments")
                                for i, comment in enumerate(post["comments"], 1):
                                    vectorized = vectorizer.transform([comment])
                                    prediction = model.predict(vectorized)[0]
                                    sentiment = "Positive" if prediction == 1 else "Negative"
                                    color = "green" if sentiment == "Positive" else "red"

                                    if sentiment == "Positive":
                                        positive_count += 1
                                    else:
                                        negative_count += 1

                                    st.write(f"**{i}.** {comment}")
                                    st.markdown(f"➡️ **Sentiment:** :{color}[**{sentiment}**]")
                            else:
                                st.info("❌ No comments found.")

                        # Save daily summary
                        cur.execute("""
                            INSERT INTO daily_telegram_sentiments (sentiment_date, positive_count, negative_count)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (sentiment_date) DO UPDATE 
                            SET positive_count = EXCLUDED.positive_count,
                                negative_count = EXCLUDED.negative_count;
                        """, (today, positive_count, negative_count))

                        conn.commit()
                        cur.close()
                        conn.close()

                        # Summary
                        st.success("✅ Today's Sentiment Summary")
                        st.write(f"👍 Positive: {positive_count}")
                        st.write(f"👎 Negative: {negative_count}")

                        # Pie Chart
                        st.subheader("🥧 Sentiment Pie Chart")
                        fig, ax = plt.subplots()
                        ax.pie([positive_count, negative_count],
                            labels=["Positive", "Negative"],
                            autopct="%1.1f%%",
                            colors=["#4CAF50", "#F44336"],
                            startangle=90)
                        ax.axis("equal")
                        st.pyplot(fig)

                    else:
                        st.warning(messages[0].get("error", "❌ Unable to fetch data."))

                except Exception as e:
                    st.error(f"❌ DB Error: {e}")


        elif analysis_option == "🔑 Topic Extraction":
            st.subheader("🔑 Keyword and Topic Extraction from Telegram")
            if st.button("📤 Fetch Comments and Extract Keywords"):
                telegram_data = fetch_post_and_comments_sync(limit=5)
                all_comments = []
                for post in telegram_data:
                    all_comments.extend(post.get("comments", []))
                if all_comments:
                    keywords = extract_keywords_yake(all_comments, max_keywords=10)
                    st.success("✅ Extracted Top Keywords:")
                    for kw in keywords:
                        st.markdown(f"- 🔑 **{kw}**")       
        elif analysis_option == "📈 Trend Analysis":
            st.subheader("📈 Topic-Based Trend Analysis from Telegram Comments")

            if st.button("🧪 Analyze Topics (LDA)"):
                try:
                    messages = fetch_post_and_comments_sync(limit=10)

                    if messages and not messages[0].get("error"):
                        all_comments = []
                        for post in messages:
                            all_comments.extend(post.get("comments", []))

                        valid_comments = [c for c in all_comments if any(char.isalpha() for char in c)]

                        if not valid_comments:
                            st.warning("⚠️ No valid text comments found for topic modeling.")
                        else:
                            # Text preprocessing
                            stop_words = set(stopwords.words("english"))
                            texts = [[word for word in comment.lower().split() if word not in stop_words and word.isalpha()] for comment in valid_comments]
                            texts = [t for t in texts if len(t) > 2]

                            # Dictionary and corpus
                            dictionary = corpora.Dictionary(texts)
                            corpus = [dictionary.doc2bow(text) for text in texts]

                            # LDA Model
                            lda_model = gensim.models.LdaModel(
                                corpus=corpus,
                                id2word=dictionary,
                                num_topics=5,
                                passes=10,
                                random_state=42
                            )

                            st.success("✅ Top Topics from Comments:")
                            for idx, topic in lda_model.print_topics(num_topics=5, num_words=5):
                                st.markdown(f"**🔹 Topic {idx+1}:** {topic}")
                    else:
                        st.warning(messages[0].get("error", "⚠️ Unable to fetch Telegram comments."))

                except Exception as e:
                    st.error(f"❌ Trend Analysis Error: {e}")    
        elif analysis_option == "🧠 Automated Reply Suggestions":
            st.subheader("🧠 Automated Reply Suggestions for Admins")

            df = fetch_logs()
            if df.empty:
                st.info("ℹ️ No comments found.")
            else:
                st.markdown("#### 💬 Comments with Suggestions")
                for idx, row in df.iterrows():
                    sentence = row["Sentence"]
                    sentiment = row["Sentiment"]

                    # Generate basic suggestions
                    if sentiment == "Negative":
                        suggestion = "We're sorry to hear that. We appreciate your feedback and will work to improve."
                    elif sentiment == "Positive":
                        suggestion = "Thank you for your kind words! We're glad you're satisfied."
                    else:
                        suggestion = "Thanks for your comment. We'll take it into consideration."

                    col1, col2 = st.columns([4, 2])
                    with col1:
                        st.markdown(f"**Comment:** `{sentence}`")
                        st.markdown(f"**Sentiment:** :{'green' if sentiment=='Positive' else 'red' if sentiment=='Negative' else 'gray'}[{sentiment}]")
                    with col2:
                        st.markdown(f"**Suggested Reply:**")
                        st.success(suggestion)

                    st.markdown("---")
        elif analysis_option == "📧 Email Notification":
            st.subheader("📧 Sentiment Threshold Notification")

            # Fetch all sentiment entries
            try:
                conn = psycopg2.connect(
                    dbname="Sentiment", user='postgres', password='Maru@132',
                    host='localhost', port='5432'
                )
                cur = conn.cursor()
                cur.execute("SELECT sentiment FROM sentiment_logs")
                all_sentiments = cur.fetchall()
                cur.close()
                conn.close()

                total = len(all_sentiments)
                positives = sum(1 for s in all_sentiments if s[0] == "Positive")
                negatives = sum(1 for s in all_sentiments if s[0] == "Negative")

                pos_percent = (positives / total) * 100 if total else 0
                neg_percent = (negatives / total) * 100 if total else 0

                st.write(f"✅ Positive: {positives} ({pos_percent:.2f}%)")
                st.write(f"❌ Negative: {negatives} ({neg_percent:.2f}%)")

                if neg_percent > 50 or pos_percent > 90:
                    send_email_alert(pos_percent, neg_percent)
                    st.success("📧 Email sent to admin.")
                else:
                    st.info("📊 Thresholds not met yet. No email sent.")

            except Exception as e:
                st.error(f"❌ Failed to process sentiment stats: {e}")
        elif analysis_option == "🧠 Comment Categorization":
            st.subheader("🧠 Comment Categorization")

            user_input = st.text_area("Enter a comment to categorize:")

            if st.button("🔍 Categorize"):
                if user_input.strip():
                    category_result = categorize_comment(user_input.strip())
                    st.success(f"📌 This comment is categorized as: **{category_result}**")
                else:
                    st.warning("⚠️ Please enter a comment.")

    elif menu == "📜 View Logs" and user_type == "Admin":
        st.subheader("📜 Past Sentiment Logs (Admin Only)")

        if "show_search" not in st.session_state:
            st.session_state["show_search"] = False
        col_icon, col_input = st.columns([0.1, 0.9])
        with col_icon:
            if st.button("🔍", help="search"):
                st.session_state["show_search"] = not st.session_state["show_search"]
        search_term = ""
        with col_input:
            if st.session_state["show_search"]:
                search_term = st.text_input("Search by sentence")

        df = fetch_logs()
        # ====== Comment Statistics ======
        total_comments = len(df)
        positive_comments = len(df[df['Sentiment'] == 'Positive'])
        negative_comments = len(df[df['Sentiment'] == 'Negative'])

        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h4>📊 Comment Summary</h4>
            <ul>
                <li><b>Total Comments:</b> {total_comments}</li>
                <li><b>Positive Comments:</b> {positive_comments}</li>
                <li><b>Negative Comments:</b> {negative_comments}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if search_term:
            df = df[df['Sentence'].str.contains(search_term, case=False)]

        if not df.empty:
            st.write("📝 Matching comments:")
            for _, row in df.iterrows():
                col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 3, 2, 2, 2, 1, 1])
                with col1: st.write(f"{row['ID']}")
                with col2: st.write(row['Sentence'])
                with col3: st.write(row['Sentiment'])
                with col4: st.write(row['Timestamp'])
                with col5:
                    role = str(row['User Type']) if row['User Type'] not in [None, '', 'NULL'] else "Unknown"
                    st.write(role)
                with col6:
                    if st.button("Delete", key=f"delete_{row['ID']}"):
                        delete_log(row['ID'])
                        st.rerun()
                with col7:
                    if st.button("💬 Response", key=f"response_btn_{row['ID']}"):
                        st.session_state[f"show_response_{row['ID']}"] = not st.session_state.get(f"show_response_{row['ID']}", False)

                # Show response text area if toggled
                if st.session_state.get(f"show_response_{row['ID']}", False):
                    with st.expander(f"✍️ Respond to ID {row['ID']}"):
                        response_text = st.text_area(f"Your response to '{row['Sentence']}'", key=f"resp_{row['ID']}")
                        if st.button("📨 Send Response", key=f"send_resp_{row['ID']}"):
                            if response_text.strip():
                                try:
                                    # Fetch the original commenter's username
                                    conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
                                    cur = conn.cursor()
                                    cur.execute("SELECT username FROM sentiment_logs WHERE id = %s", (row['ID'],))
                                    recipient_result = cur.fetchone()

                                    if recipient_result:
                                        recipient_username = recipient_result[0]
                                        cur.execute("""
                                            INSERT INTO sentiment_logs (sentence, sentiment, created_at, username, recipient)
                                            VALUES (%s, %s, %s, %s, %s)
                                        """, (
                                            f"Response: {response_text}",
                                            'Neutral',
                                            datetime.now(),
                                            st.session_state["username"],  # responder (admin)
                                            recipient_username             # original commenter (tigist)
                                        ))
                                        conn.commit()
                                        st.success(f"✅ Response sent to {recipient_username}.")
                                        st.session_state[f"show_response_{row['ID']}"] = False
                                        st.rerun()
                                    else:
                                        st.error("❌ Could not find recipient for this comment.")

                                    cur.close()
                                    conn.close()

                                except Exception as e:
                                    st.error(f"❌ Error sending response: {e}")
                            else:
                                st.warning("⚠️ Response cannot be empty.")

                        
        else:
            st.info("ℹ️ No matching sentiment logs found.")

    elif menu == "📋 User Management" and user_type == "Admin":
        st.subheader("📋 User Management")
        search_input = st.text_input("🔍 Search User by Username")
        users = fetch_users()
        filtered = [u for u in users if search_input.lower() in u[1].lower()]

        if filtered:
            for user in filtered:
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                with col1: st.write(user[0])
                with col2: st.write(user[1])
                with col3: st.write(user[2])
                with col4:
                    if st.button("🗑️ Delete", key=f"del_user_{user[0]}"):
                        delete_user(user[0])
                        st.rerun()
        else:
            st.info("ℹ️ No users found.")
    elif menu == "📈 Visualization":
        st.subheader("📈 Sentiment Distribution Visualization")

        df = fetch_logs()

        # Count each sentiment
        sentiment_counts = df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Bar Chart
        st.markdown("#### 📊 Bar Chart")
        st.bar_chart(data=sentiment_counts.set_index('Sentiment'))

        # Pie Chart (using Plotly for nicer visuals)
        import plotly.express as px
        st.markdown("#### 🥧 Pie Chart")
        fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title='Sentiment Share')
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "📊 Report":
        st.subheader("📊 Sentiment Analysis Report")

        df = fetch_logs()

        # Report summary
        total_comments = len(df)
        positive_comments = len(df[df['Sentiment'] == 'Positive'])
        negative_comments = len(df[df['Sentiment'] == 'Negative'])

        st.markdown(f"""
            <div style="background-color:#e0f7fa; padding:20px; border-radius:10px; margin-bottom:20px;">
                <h4>📈 Report Summary</h4>
                <ul>
                    <li><b>Total Comments:</b> {total_comments}</li>
                    <li><b>Positive Comments:</b> {positive_comments}</li>
                    <li><b>Negative Comments:</b> {negative_comments}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Optional: show the detailed table
        st.write("📋 Detailed Logs")
        st.dataframe(df)

        # ✅ CSV Export Button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Report as CSV",
            data=csv_data,
            file_name="sentiment_report.csv",
            mime="text/csv"
        )
    elif menu == "📊 Telegram Report":
        st.subheader("📊 Telegram Sentiment Summary Report")

        try:
            # Fetch data from the sentiment summary table
            conn = psycopg2.connect(
                dbname="Sentiment", user="postgres", password="Maru@132", host="localhost", port="5432"
            )
            cur = conn.cursor()
            cur.execute("""
                SELECT sentiment_date, positive_count, negative_count 
                FROM daily_telegram_sentiments
                ORDER BY sentiment_date DESC
            """)
            rows = cur.fetchall()
            cur.close()
            conn.close()

            if rows:
                # Display as table
                df = pd.DataFrame(rows, columns=["Date", "Positive", "Negative"])
                st.dataframe(df)

                # Plot pie chart for the latest day
                latest = df.iloc[0]
                fig, ax = plt.subplots()
                ax.pie(
                    [latest["Positive"], latest["Negative"]],
                    labels=["Positive", "Negative"],
                    autopct="%1.1f%%",
                    colors=["#00C49F", "#FF4444"]
                )
                st.pyplot(fig)

                # Optional: download as CSV
                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Report as CSV",
                    data=csv_data,
                    file_name="telegram_sentiment_report.csv",
                    mime="text/csv"
                )

            else:
                st.info("📭 No Telegram sentiment records yet.")

        except Exception as e:
            st.error(f"❌ Error fetching report: {e}")
    elif menu == "🗂️ Comment Categorization":
        st.subheader("🗂️ Automated Comment Categorization")

        comments = fetch_all_comments()

        if comments:
            # Transform and predict
            comments_vectorized = category_vectorizer.transform(comments)
            predicted_categories = category_model.predict(comments_vectorized)

            # Count category frequencies
            category_counts = Counter(predicted_categories)
            labels = list(category_counts.keys())
            sizes = list(category_counts.values())
                    
            total_comments = sum(category_counts.values())
            percentages = {cat: (count / total_comments) * 100 for cat, count in category_counts.items()}
             # Display raw stats
            st.markdown("### 📊 Category Breakdown")
            for label, count in category_counts.items():
                st.write(f"**{label}**: {count} comments")

           

                    

            # Convert counts to percentages
            total_comments = sum(category_counts.values())
            percentages = {cat: (count / total_comments) * 100 for cat, count in category_counts.items()}

            # Sort categories for better presentation
            sorted_percentages = dict(sorted(percentages.items(), key=lambda x: x[1], reverse=True))

            # Plot bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = list(sorted_percentages.keys())
            values = list(sorted_percentages.values())

            bars = ax.barh(categories, values, color='mediumseagreen')

            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                        f'{width:.1f}%', va='center', fontsize=10)

            ax.set_xlabel("Percentage (%)")
            ax.set_title("Categorized Comment Distribution")
            ax.set_xlim(0, 100)
            ax.invert_yaxis()  # Highest at the top

            st.pyplot(fig)
        else:
             st.info("📭 No comments found in the database.")


            

            
           


    elif menu == "🌐 Translate":
        st.subheader("🌐 English → Amharic Translator")

        # Input
        english_text = st.text_area("✏️ Type in English", placeholder="Enter English text to translate...")

        # Translate and display
        if english_text.strip():
            amharic_translation = translate_to_amharic(english_text)
            st.markdown(f"#### 📘 Translated to Amharic:")
            st.success(amharic_translation)
        else:
            st.button("ℹ️ Translate.")


    elif menu == "ℹ️ About Us":
        st.subheader("ℹ️ About Mareye Zeleke Mekonen")
        st.image("mareyephoto.jpg", width=200, caption="Mareye Zeleke Mekonen", use_column_width=False)

        st.write("""
            Mareye Zeleke Mekonen is an Instructor at **Injibara University**, Ethiopia.
            He specializes in **Artificial Intelligence**, **Natural Language Processing**, and **Machine Learning**.
            This project is part of an ongoing initiative to enable local-language technology solutions using advanced AI tools.
        """)
    elif menu == "🚪 Logout":
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.success("✅ Logged out successfully.")
            st.rerun()

# ========== Footer ==========
st.markdown("""
    <div class='full-footer'>
        Copyright © 2025 Injibara University, Ethiopia. All rights reserved
    </div>
""", unsafe_allow_html=True)
