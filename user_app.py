import streamlit as st
import joblib
import psycopg2
import pandas as pd
from datetime import datetime
import base64
import bcrypt
from translate import Translator
import os
import io
def translate_to_amharic(text):
    try:
        translator = Translator(to_lang="am")  # "am" is the ISO code for Amharic
        translation = translator.translate(text)
        return translation
    except Exception as e:
        return f"⚠️ Translation failed: {e}"
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
if "form_type" not in st.session_state:
    st.session_state["form_type"] = ""


# ========== PostgreSQL Functions ==========

# ====== PostgreSQL Functions ======
def register_user(username, password, role):
    try:
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        conn = psycopg2.connect(
            dbname="Sentiment", user='postgres', password='Maru@132',
            host='localhost', port='5432'
        )
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s)", (username, hashed, role))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"❌ Registration failed: {e}")
        return False
def get_users_by_role(role):
    try:
        conn = psycopg2.connect(
            dbname="Sentiment", user='postgres', password='Maru@132',
            host='localhost', port='5432'
        )
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE role = %s", (role,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        st.error(f"❌ Failed to fetch users by role: {e}")
        return []

def validate_login(username, password):
    try:
        conn = psycopg2.connect(
            dbname="Sentiment", user='postgres', password='Maru@132',
            host='localhost', port='5432'
        )
        cur = conn.cursor()
        cur.execute("SELECT password_hash, role FROM users WHERE username = %s", (username,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            stored_hash, role = result
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                return role
        return None
    except Exception as e:
        st.error(f"❌ Login failed: {e}")
        return None

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def insert_user_comment(username, comment_type, comment_text, recipient, file_content, file_name):
    try:
        full_comment = f"{comment_type} - {comment_text}"
        input_vector = vectorizer.transform([comment_text])
        prediction = model.predict(input_vector)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO sentiment_logs (sentence, sentiment, created_at, username, recipient, file_content, file_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            full_comment,
            sentiment,
            datetime.now(),
            username,
            recipient,
            psycopg2.Binary(file_content) if file_content else None,
            file_name
        ))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        st.error(f"❌ Error saving comment: {e}")



def fetch_inbox_messages_for_user(username):
    conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
    cur = conn.cursor()
    cur.execute("""
        SELECT sentence, sentiment, created_at, username, file_content, file_name
        FROM sentiment_logs
        WHERE recipient = %s
        ORDER BY created_at DESC
    """, (username,))
    messages = cur.fetchall()
    cur.close()
    conn.close()
    return messages


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
# ====== Clear login fields callback ======
def clear_login_fields():
    st.session_state.login_username = ""
    st.session_state.login_password = ""

# ====== Clear register fields callback ======
def clear_register_fields():
    st.session_state.new_username = ""
    st.session_state.new_password = ""
def delete_log(comment_id):
    try:
        conn = psycopg2.connect(dbname="Sentiment", user='postgres', password='Maru@132', host='localhost', port='5432')
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO trash (sentence, sentiment, deleted_at)
            SELECT sentence, sentiment, NOW() FROM sentiment_logs WHERE id = %s
        """, (comment_id,))
        cur.execute("DELETE FROM sentiment_log WHERE id = %s", (comment_id,))
        conn.commit()
        cur.close()
        conn.close()
        st.success(f"🗑️ Deleted log ID {log_id} and moved to trash.")
    except Exception as e:
        st.error(f"❌ Error deleting log: {e}")
# ====== Main Logic ======
if not st.session_state.logged_in:
    st.markdown("<div class='center-box'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Login"):
            st.session_state.form_type = "login"
    with col2:
        if st.button("📝 Register"):
            st.session_state.form_type = "register"

    if st.session_state.form_type == "login":
        st.markdown("### Login")

        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔓 Login", key="login_btn"):
                role = validate_login(login_username.strip(), login_password.strip())
                if role:
                    st.session_state.logged_in = True
                    st.session_state.username = login_username.strip()
                    st.session_state.role = role
                    st.success(f"✅ Logged in as {login_username} ({role})")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        with col2:
            if st.button("❌ Cancel", key="cancel_login_btn", on_click=clear_login_fields):
                pass


    elif st.session_state.form_type == "register":
        st.markdown("### Register")

        new_username = st.text_input("New Username", key="new_username")
        new_password = st.text_input("New Password", type="password", key="new_password")

        roles = ["Anonymous", "Student", "Instructor", "Administrative Staff", "Outsource Worker", 
                "Higher Manager", "College Dean", "Department Head", "Director", "Other Members"]
        
        new_role = st.selectbox("Select Role", roles)

        # Only show university ID input for non-anonymous roles
        user_id = ""
        if new_role != "Anonymous":
            user_id = st.text_input(f"Enter {new_role} ID")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Register", key="reg_btn"):
                if new_role != "Anonymous":
                    csv_filename = f"{new_role.lower().replace(' ', '_')}.csv"
                    csv_path = os.path.join("role_csv", csv_filename)  # Folder where your CSVs are stored

                    if os.path.exists(csv_path):
                        import pandas as pd
                        df = pd.read_csv(csv_path)

                        if user_id in df["id"].astype(str).values:
                            if register_user(new_username.strip(), new_password.strip(), new_role):
                                st.success("✅ Registered successfully! You can now login.")
                        else:
                            st.error("❌ This ID does not exist in the selected role. Please check again or register as Anonymous.")
                    else:
                        st.error(f"❌ Missing role verification file: `{csv_filename}`")
                else:
                    # Anonymous user doesn't need ID check
                    if register_user(new_username.strip(), new_password.strip(), new_role):
                        st.success("✅ Registered successfully as Anonymous!")
        
        with col2:
            if st.button("❌ Cancel", key="cancel_register_btn", on_click=clear_register_fields):
                pass

        st.markdown("</div>", unsafe_allow_html=True)


# ========== Sidebar ==========
col1, col2 = st.columns([1, 4])
with col1:
    if not st.session_state["logged_in"]:
        

        

        
        
        st.stop()

    
    else:
        role_permissions = {
            "Anonymous": ["🏠 Home", "✍️ Give Comment", "ℹ️ About Us", "🚪 Logout"],
            "Student": ["🏠 Home", "✍️ Give Comment", "📥 Inbox","🧠 Comment Categorization", "🌐 Translate", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "Instructor": ["🏠 Home", "✍️ Give Comment", "📥 Inbox", "🧠 Comment Categorization","🌐 Translate", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "Administrative Staff": ["🏠 Home", "✍️ Give Comment", "📥 Inbox", "🧠 Comment Categorization","🌐 Translate", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "Outsource Worker": ["🏠 Home", "✍️ Give Comment", "📥 Inbox", "🧠 Comment Categorization","👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "Higher Manager": ["🏠 Home", "✍️ Give Comment", "📥 Inbox","🧠 Comment Categorization", "🌐 Translate", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "College Dean": ["🏠 Home", "✍️ Give Comment", "📥 Inbox","🧠 Comment Categorization", "🌐 Translate", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "Department Head": ["🏠 Home", "✍️ Give Comment", "📥 Inbox", "🧠 Comment Categorization","🌐 Translate", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "Director": ["🏠 Home", "✍️ Give Comment", "📥 Inbox", "🧠 Comment Categorization","🌐 Translate", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "Other Member": ["🏠 Home", "✍️ Give Comment", "📥 Inbox","🧠 Comment Categorization", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"],
            "admin": ["🏠 Home", "✍️ Give Comment", "📥 Inbox", "🧠 Comment Categorization","🌐 Translate", "👤 Change Profile", "ℹ️ About Us", "🚪 Logout"]
        }
        st.markdown("### 👤 Welcome")
        st.write(f"**{st.session_state.username}** ({st.session_state.role})")

        # 📥 Dynamic Inbox Label
        inbox_msgs = fetch_inbox_messages_for_user(st.session_state.username)

        # Initialize last_read_time if not already set
        if "last_read_time" not in st.session_state:
            st.session_state.last_read_time = datetime.min  # Very old

        # Check how many new messages
        new_msgs = [msg for msg in inbox_msgs if msg[2] > st.session_state.last_read_time]
        new_msg_count = len(new_msgs)

        # Update label for inbox
        inbox_label = f"📥 Inbox 🔴 ({new_msg_count})" if new_msg_count > 0 else "📥 Inbox"

        # Get current role
        current_role = st.session_state.role

        # Define full menu (mapping inbox dynamically)
        all_pages = {
            "🏠 Home": "home",
            "✍️ Give Comment": "give_comment",
            "📥 Inbox": inbox_label,
            "🌐 Translate": "translate",
            "👤 Change Profile": "change_profile",
            "ℹ️ About Us": "about_us",
            "🚪 Logout": "logout"
        }

        # Filter allowed options based on the user’s role
        allowed_labels = role_permissions.get(current_role, ["🏠 Home", "🚪 Logout"])

        # Replace '📥 Inbox' dynamically if needed
        if "📥 Inbox" in allowed_labels:
            allowed_labels = [inbox_label if label == "📥 Inbox" else label for label in allowed_labels]

        # 📌 Sidebar Navigation with filtered options
        selected_option = st.radio(
            "Select Page",
            allowed_labels,
            key="menu_radio"
        )

            

# ========== Main Content ==========
#col_menu, col_main = st.columns([1, 4])
with col2:
    if selected_option == "🏠 Home":
        st.subheader("🏠 Welcome to the Sentiment Analysis Portal")
        st.write("Use the menu on the left to navigate between features.")
            
            


    elif selected_option == "✍️ Give Comment":
            st.subheader("✍️ Submit a Comment")

            # Initialize session state
            if "comment_text" not in st.session_state:
                st.session_state.comment_text = ""
            if "selected_college" not in st.session_state:
                st.session_state.selected_college = ""
            if "selected_user" not in st.session_state:
                st.session_state.selected_user = ""

            category = st.selectbox("Who is this comment about?", [
                "Department", "College", "Registrar", "Human Resource",
                "Academic Vice Affairs", "President", "Instructor",
                "Student", "Outsource Workers", "Lounge"
            ])

            college_mapping = {
                "College of Agriculture": "dean_agri",
                "College of Business and Economics": "dean_be",
                "College of Education and Behavioural Science": "dean_edu",
                "College of Engineering and Technology": "dean_eng",
                "College of Medicine and Health Science": "dean_med",
                "College of Natural and Computational Science": "dean_ncs",
                "College of Social Science and Humanity": "dean_ssh",
                "School of Law": "dean_law"
            }

            full_category = category
            recipient_usernames = ["admin"]  # Start with admin

            if category == "College":
                st.session_state.selected_college = st.selectbox(
                    "Which College?", list(college_mapping.keys()), key="college_select"
                )
                if st.session_state.selected_college:
                    full_category = f"College - {st.session_state.selected_college}"
                    recipient_usernames.append(college_mapping[st.session_state.selected_college])

            elif category in ["Instructor", "Student","Registrar","Human Resource","Academic Vice Affairs", "President",
                                                "Outsource Worker director", "Lounge director"]:
                # Fetch actual users based on role
                available_users = get_users_by_role(category)  # You’ll define this function
                st.session_state.selected_user = st.selectbox(
                    f"Select {category}", available_users, key="user_select"
                )
                if st.session_state.selected_user:
                    recipient_usernames.append(st.session_state.selected_user)

            st.session_state.comment_text = st.text_area("💬 Enter your comment here:", key="comment_input")
            uploaded_file = st.file_uploader("📎 Upload a file (PDF, DOCX, Image, etc.):", type=["pdf", "docx", "png", "jpg", "jpeg"], key="comment_file")


            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Submit"):
                    if st.session_state.comment_text.strip():
                        # Initialize a clean set to store unique recipients
                        unique_recipients = set()

                        # Add selected user (like dean, student, or instructor) if available
                        if st.session_state.selected_user:
                            unique_recipients.add(st.session_state.selected_user)

                        # Add college dean if selected
                        if st.session_state.selected_college:
                            college_mapping = {
                                "College of Agriculture": "dean_agri",
                                "College of Business and Economics": "dean_be",
                                "College of Education and Behavioural Science": "dean_edu",
                                "College of Engineering and Technology": "dean_eng",
                                "College of Medicine and Health Science": "dean_med",
                                "College of Natural and Computational Science": "dean_ncs",
                                "College of Social Science and Humanity": "dean_ssh",
                                "School of Law": "dean_law"
                            }
                            college_dean = college_mapping.get(st.session_state.selected_college)
                            if college_dean:
                                unique_recipients.add(college_dean)

                        # Add admin only if not already a recipient
                        if "admin" in unique_recipients:
                            unique_recipients.add("admin")

                        # Final: Insert for each unique recipient
                        for recipient in unique_recipients:
                            insert_user_comment(
                            st.session_state.username,
                            full_category,
                            st.session_state.comment_text,
                            recipient,
                            uploaded_file.read() if uploaded_file else None,
                            uploaded_file.name if uploaded_file else None
                        )
                       
                        default_reply = "📩 Thank you for your feedback! Your comment has been received and will be reviewed."
                        insert_user_comment(
                            username="admin",  # sender is system/admin
                            comment_type="Auto-Reply",
                            comment_text=default_reply,
                            recipient=st.session_state.username,  # send back to the comment sender
                            file_content=None,
                            file_name=None
                        )


                        st.success("📜 Comment submitted successfully.")

                        # Clear input fields
                        st.session_state.comment_text = ""
                        st.session_state.selected_college = ""
                        st.session_state.selected_user = ""

                        st.rerun()
                    else:
                        st.warning("⚠️ Please write a comment before submitting.")





    elif selected_option.startswith("📥 Inbox"):
        st.subheader("📥 Inbox - Messages Received Here")
        
        inbox_msgs = fetch_inbox_messages_for_user(st.session_state.username)

      

        # Render messages
        if inbox_msgs:
            for sentence, sentiment, created_at, sender_username, file_content, file_name in inbox_msgs:
                with st.expander(f"🗨️ From: {sender_username} | Sentiment: {sentiment} | At: {created_at.strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.write(f"**Comment:** {sentence}")

                    if file_name and file_content:
                        st.download_button(
                            label=f"📎 Download attached file: {file_name}",
                            data=file_content.tobytes(),
                            file_name=file_name,
                            mime="application/octet-stream",
                            key=f"download_{created_at.timestamp()}_{sender_username}"
                        )

                    # Unique keys
                    response_key = f"response_{created_at.timestamp()}_{sender_username}"
                    upload_key = f"upload_{created_at.timestamp()}_{sender_username}"

                    response = st.text_area("✉️ Write your response:", key=response_key)
                    uploaded_file = st.file_uploader("📎 Attach a file (optional)", key=upload_key)

                    if st.button("📤 Send Response", key=f"btn_{response_key}"):
                        if response.strip():
                            insert_user_comment(
                                username=st.session_state.username,
                                comment_type="Response",
                                comment_text=response,
                                recipient=sender_username,
                                file_content=uploaded_file.read() if uploaded_file else None,
                                file_name=uploaded_file.name if uploaded_file else None
                            )
                            st.success(f"✅ Response sent to `{sender_username}`.")
                            st.rerun()
                        else:
                            st.warning("⚠️ Please write something before sending.")
            
            # ✅ Update last_read_time after showing messages
            latest_msg_time = max(msg[2] for msg in inbox_msgs)
            if latest_msg_time > st.session_state.last_read_time:
                st.session_state.last_read_time = latest_msg_time

        else:
            st.info("📭 No responses received yet.")

    elif selected_option == "🧠 Comment Categorization":
        st.markdown("### 🧠 Categorize Your Received Comments")

        # Load models
        category_model = joblib.load("comment_category_model.pkl")
        category_vectorizer = joblib.load("comment_category_vectorizer.pkl")
        sentiment_model = joblib.load("sentiment_model.pkl")
        sentiment_vectorizer = joblib.load("vectorizer.pkl")

        # Fetch comments received by user
        user_received_comments = fetch_inbox_messages_for_user(st.session_state.username)
       

        if not user_received_comments:
            st.info("📭 No comments received yet.")
        else:
            # Extract comment texts
            comments = [msg[0] for msg in user_received_comments]

            # ---- Sentiment Classification ----
            sentiment_vectors = sentiment_vectorizer.transform(comments)
            sentiments = sentiment_model.predict(sentiment_vectors)
            total_positive = sum(1 for s in sentiments if s == 1)
            total_negative = sum(1 for s in sentiments if s == 0)

            # ---- Category Classification ----
            category_vectors = category_vectorizer.transform(comments)
            categories = category_model.predict(category_vectors)

            # ---- Combine into DataFrame ----
            # Update fetch_inbox_messages_for_user() to include message_id from the DB
            user_received_comments = fetch_inbox_messages_for_user(st.session_state.username)

            # Assume each row = (id, sentence, sentiment, created_at, sender_username, ...)
            comments = [msg[0] for msg in user_received_comments]

            # Then later:
            df = pd.DataFrame({
                "Sentence": comments,
                "Sentiment": ["Positive" if s == 1 else "Negative" for s in sentiments],
                "Category": categories
            })



            # ---- Stats Display ----
            st.markdown(f"**📌 Total Comments Received:** {len(df)}")
            st.markdown(f"✅ Positive Comments: {total_positive}")
            st.markdown(f"❌ Negative Comments: {total_negative}")
            st.divider()

            # ---- Table-like Custom Row Display with Delete ----
            st.markdown("#### 🗃️ Categorized Comments")

            



           

          

            # ---- Percentage per Category ----
            category_counts = df['Category'].value_counts()
            category_percent = (category_counts / len(df)) * 100
            category_percent = category_percent.sort_index()  # Sort alphabetically

            # ---- Display Data Table ----
            st.markdown("#### 🗃️ Categorized Comments")
            st.dataframe(df, use_container_width=True)

            # ---- Bar Chart ----
            st.markdown("#### 📊 Category Percentage Chart")
            st.bar_chart(category_percent)
           

            # Convert categorized comments to DataFrame for download
            df_download = pd.DataFrame(df)

            if not df_download.empty:
                # Create a downloadable CSV file in-memory
                csv_buffer = io.StringIO()
                df_download.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📥 Download Categorized Comments (CSV)",
                    data=csv_data,
                    file_name="categorized_comments.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ No categorized comments available to download.")


    elif selected_option == "🌐 Translate":
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

    elif selected_option == "👤 Change Profile":
        st.subheader("🔒 Change Password")

        st.write("You can update your password below:")

        current_password = st.text_input("Current Password", type="password", key="current_pass")
        new_password = st.text_input("New Password", type="password", key="new_pass")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pass")

        if st.button("🔁 Update Password"):
            if not current_password or not new_password or not confirm_password:
                st.warning("⚠️ Please fill all fields.")
            elif new_password != confirm_password:
                st.error("❌ New passwords do not match.")
            else:
                try:
                    conn = psycopg2.connect(
                        dbname="Sentiment", user='postgres', password='Maru@132',
                        host='localhost', port='5432'
                    )
                    cur = conn.cursor()
                    cur.execute("SELECT password_hash FROM users WHERE username = %s", (st.session_state.username,))
                    result = cur.fetchone()

                    if result and bcrypt.checkpw(current_password.encode('utf-8'), result[0].encode('utf-8')):
                        new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        cur.execute("UPDATE users SET password_hash = %s WHERE username = %s", (new_hash, st.session_state.username))
                        conn.commit()
                        st.success("✅ Password updated successfully.")
                    else:
                        st.error("❌ Current password is incorrect.")

                    cur.close()
                    conn.close()
                except Exception as e:
                    st.error(f"❌ Failed to update password: {e}")


    elif selected_option == "ℹ️ About Us":
            st.subheader("ℹ️ About Mareye Zeleke Mekonen")
            st.image("mareyephoto.jpg", width=200, caption="Mareye Zeleke Mekonen", use_column_width=False)

            st.write("""
                Mareye Zeleke Mekonen is an Instructor at **Injibara University**, Ethiopia.
                He specializes in **Artificial Intelligence**, **Natural Language Processing**, and **Machine Learning**.
                This project is part of an ongoing initiative to enable local-language technology solutions using advanced AI tools.
            """)

    elif selected_option == "🚪 Logout":
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.success("✅ Logged out successfully.")
            st.rerun()

# ====== Footer ======
st.markdown("""
    <div class='full-footer'>
        Developed by Mareye Zeleke | 📞 +251906283518 | 📧 mareye132@gmail.com | © 2025 Injibara University, Ethiopia
    </div>
""", unsafe_allow_html=True)





