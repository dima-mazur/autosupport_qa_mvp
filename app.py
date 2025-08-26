
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


STAR_TO_SCORE = {5:10, 4:8, 3:7, 2:5, 1:2, 0:0}
LOW_OVERALL = 5.0

st.set_page_config(page_title="Support QA — ONE CSV (Simple + Charts)", layout="wide")
st.title("Support QA")

uploaded = st.file_uploader("Завантажте CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded, dtype=str)

    required = {'conversation_id','topic','user_text','agent_text','created_at_user','created_at_agent','resolved','user_csat_0_5','agent_name'}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    df['resolved'] = df['resolved'].fillna('0').astype(int)
    df['user_csat_0_5'] = df['user_csat_0_5'].fillna("")
    df['created_at_user'] = pd.to_datetime(df['created_at_user'], errors='coerce')
    df['created_at_agent'] = pd.to_datetime(df['created_at_agent'], errors='coerce')

    def score_speed(t_user, t_agent):
        try:
            ttfr = max((t_agent - t_user).total_seconds(), 0)
        except Exception:
            ttfr = 0
        if ttfr <= 600:  # <= 10 хв
            return 10
        if ttfr <= 1800:  # <= 30 хв
            return 8
        if ttfr <= 3600:  # <= 1 год
            return 5
        if ttfr > 7200:  # <= 2 год
            return 5
        return 0

    def score_empathy(text):
        text_l = (text or "").lower()
        markers = [
            'sorry','apolog','please','thank you','i understand','we understand',
        'перепрошу','будь ласка','дяку','розумію','спробуємо'
    ]
        hits = sum(1 for w in markers if w in text_l)
        return max(0, min(10, hits))

    def score_correctness(user_text, agent_text):
        u=(user_text or "").lower(); a=(agent_text or "").lower()
        topics={'login':['login','log in','sign in','password','403','вхід','увійти','пароль','facebook'],
                'payment':['paid','charge','payment','paypal','coins','платеж','оплат','монет','refund','receipt'],
                'progress':['progress','level','restore','віднов','прогрес','рівень','reset','backup'],
                'luck':['luck','rng','jackpot','rigged','джекпот','rtp','rpt','виграв'],
                'tech':['crash','freeze','loading','spin button','patch','update','версі','падає','лаги'],
                'kyc':['kyc','verify','verification','id','utility bill','sweepstake','redeem','вериф','документ']}
        user_topics = {k for k,kws in topics.items() if any(w in u for w in kws)}
        overlap = sum(1 for k in user_topics if any(w in a for w in topics[k]))
        actions = ['reset','refund','credit','restore','reinstall','update','link','verify','escalate','check','clear cache']
        action_hits = sum(1 for w in actions if w in a)
        base = 4*overlap + min(3, action_hits)
        if len(a.split()) < 6:
            base = max(0, base-2)
        return max(0, min(10, base))

    def score_resolution(resolved, agent_text):
        if not resolved:
            return 0
        l = (agent_text or "").lower()
        pts = 0
        if any(k in l for k in ['fixed', 'resolved', 'done', 'credited', 'refund', 'restored', 'approved',
                                'вирішено', 'готово', 'повернення', 'відновимо', 'відновили']):
            pts += 7
        if any(k in l for k in
               ['eta', 'today', 'tomorrow', 'завтра', 'сьогодні', 'within 24', '24h', '24 hours', 'ASAP']):
            pts += 3
        return max(0, min(10, pts))

    def score_csat(stars, resolved):
        if not resolved:
            return 0
        try:
            return STAR_TO_SCORE.get(int(stars), 0)
        except:
            return 0


    def build_summary_and_recommendation(row):
        agent = row['agent_name']
        topic = row['topic']
        resolved = int(row['resolved']) == 1
        csat = row['user_csat_0_5'] if resolved and str(row['user_csat_0_5']).strip() != "" else None

        if not resolved:
            summary = f"Агент {agent} обробив тікет по темі {topic}, але він не вирішений."
            recommendation = ""
            return summary, recommendation

        if csat is not None and int(csat) <= 2:
            summary = f"Агент {agent} закрив тікет по темі {topic}, але кастомер незадоволений ({int(csat)}★)."
            recommendation = "Додай більше емпатії й поясни наступні кроки та терміни виконання."
        else:
            summary = f"Агент {agent} обробив тікет по темі {topic}{f' (оцінка {csat}★)' if csat else ''}."
            recommendation = ""

        return summary, recommendation


    import pandas as pd

    df[['summary_text', 'recommendation']] = df.apply(
        lambda r: pd.Series(build_summary_and_recommendation(r)),
        axis=1
    )






    # Compute metrics
    df['speed'] = [score_speed(df.iloc[i]['created_at_user'], df.iloc[i]['created_at_agent']) for i in range(len(df))]
    df['empathy'] = df['agent_text'].apply(score_empathy)
    df['correctness'] = [score_correctness(df.iloc[i]['user_text'], df.iloc[i]['agent_text']) for i in range(len(df))]
    df['resolution_score'] = [score_resolution(int(df.iloc[i]['resolved']), df.iloc[i]['agent_text']) for i in range(len(df))]
    df['csat_score_0_10'] = [score_csat(df.iloc[i]['user_csat_0_5'], int(df.iloc[i]['resolved'])) for i in range(len(df))]

    df['overall'] = ((df['speed'] + df['empathy'] + df['correctness'] + df['resolution_score'] + df['csat_score_0_10']) / 5).round(2)

    st.subheader("Діалоги")
    st.dataframe(df[['conversation_id','agent_name','topic','resolved','user_csat_0_5',
        'speed','empathy','correctness','resolution_score','csat_score_0_10','overall',
        'summary_text','recommendation']].head(101), use_container_width=True)

    # --------- Глобальні діаграми ---------
    col1, col2 = st.columns(2)

    with col1:
        plt.figure()
        df['overall'].plot.hist(bins=10, rwidth=0.8)
        plt.title("Розподілення Overall")
        plt.xlabel("Overall"); plt.ylabel("Count")
        st.pyplot(plt.gcf())

    with col2:
        plt.figure()
        df[['speed','empathy','correctness','resolution_score','csat_score_0_10']].mean().plot.bar(rot=0)
        plt.title("Середнє значення метрик")
        plt.ylabel("Average")
        st.pyplot(plt.gcf())

    col3, col4 = st.columns(2)

    with col3:
        plt.figure()
        df['resolved'].astype(int).mean()
        (pd.Series({'Resolved': df['resolved'].astype(int).mean(), 'Unresolved': 1 - df['resolved'].astype(int).mean()})
            .plot.bar(rot=0))
        plt.title("Частка Resolved / Unresolved")
        plt.ylabel("Share")
        st.pyplot(plt.gcf())

    with col4:
        plt.figure()
        csat_counts = df.loc[df['resolved']==1, 'user_csat_0_5'].replace("", pd.NA).dropna().astype(int).value_counts().sort_index()
        csat_counts.reindex([0,1,2,3,4,5], fill_value=0).plot.bar(rot=0)
        plt.title("Розподілення CSAT")
        plt.xlabel("Stars"); plt.ylabel("Count")
        st.pyplot(plt.gcf())

    # --------- Теми ---------
    st.subheader("Breakdown по темам")
    topic_stats = df.groupby('topic').agg(
        tickets=('conversation_id','count'),
        avg_overall=('overall','mean')
    ).reset_index()

    col5, col6 = st.columns(2)
    with col5:
        plt.figure()
        topic_stats.sort_values('tickets', ascending=False).plot(x='topic', y='tickets', kind='bar', rot=0)
        plt.title("Кількість тікетів по темам")
        plt.ylabel("Tickets")
        st.pyplot(plt.gcf())

    with col6:
        plt.figure()
        topic_stats.sort_values('avg_overall', ascending=False).plot(x='topic', y='avg_overall', kind='bar', rot=0)
        plt.title("Средній Overall по темам")
        plt.ylabel("Avg overall")
        st.pyplot(plt.gcf())

    # --------- TTFR Histogram ---------
    st.subheader("TTFR (сек)")

    df['created_at_user'] = pd.to_datetime(df['created_at_user'], errors='coerce', utc=True)
    df['created_at_agent'] = pd.to_datetime(df['created_at_agent'], errors='coerce', utc=True)

    delta = df['created_at_agent'] - df['created_at_user']
    ttfr = delta.dt.total_seconds()  # float с NaN, если где-то NaT
    ttfr = ttfr.fillna(0).clip(lower=0)  # заменяем NaN и не даём уйти в минус

    # Если вдруг нет данных, покажем инфо и не падём на пустом графике
    if (ttfr.size == 0) or (ttfr.notna().sum() == 0):
        st.info("Немає даних для побудови TTFR.")
    else:
        plt.figure()
        pd.Series(ttfr).plot.hist(bins=20, rwidth=0.8)
        plt.title("Розподіл TTFR (сек)")
        plt.xlabel("Seconds");
        plt.ylabel("Count")
        st.pyplot(plt.gcf())

    # --------- Агентська аналітика ---------
    st.subheader("Агенти")
    agent_stats = df.groupby('agent_name').agg(
        tickets=('conversation_id','count'),
        resolved_rate=('resolved','mean'),
        avg_overall=('overall','mean')
    ).reset_index()

    agent_stats['low_overall_flag'] = (agent_stats['avg_overall'] < LOW_OVERALL).astype(int)

    st.dataframe(agent_stats.sort_values('avg_overall'), use_container_width=True)

    plt.figure()
    (agent_stats.sort_values('avg_overall', ascending=False)
        .plot(x='agent_name', y='avg_overall', kind='bar', rot=0))
    plt.title("Average Overall by Agent"); plt.ylabel("Avg overall")
    st.pyplot(plt.gcf())

    st.download_button("Завантажити зведення по агентам", agent_stats.to_csv(index=False).encode('utf-8'),
        file_name='agent_analytics_avg_overall.csv', mime='text/csv')

    # --- NEW: формируем текстовую характеристику и рекомендацию (только для низкого CSAT) ---
    summaries = []
    recs = []
    for i in range(len(df)):
        s, r = build_summary_and_recommendation(df.iloc[i])
        summaries.append(s)
        recs.append(r)

    df['summary_text'] = summaries
    df['recommendation'] = recs

else:
    st.info("Завантажте один CSV: conversation_id, topic, user_text, agent_text, created_at_user, created_at_agent, resolved, user_csat_0_5, agent_name")

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import io

def send_result_via_email(summary_text, df_csv, recipient_email):
    EMAIL_SENDER = "your_email@gmail.com"       # твій Gmail
    EMAIL_PASSWORD = "your_app_password"        # пароль додатку (App Password з Google)
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

    # Формуємо лист
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = recipient_email
    msg["Subject"] = "Результат аналітики"

    # Текст листа
    msg.attach(MIMEText(summary_text, "plain", "utf-8"))

    # Додаємо CSV як вкладення
    part = MIMEBase("application", "octet-stream")
    part.set_payload(df_csv.encode("utf-8"))
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", "attachment; filename=agent_analytics.csv")
    msg.attach(part)

    # Відправка
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL_SENDER, EMAIL_PASSWORD)
    server.sendmail(EMAIL_SENDER, recipient_email, msg.as_string())
    server.quit()


# --- Додаємо блок у Streamlit ---
if uploaded:
    st.subheader("Відправити результат на e-mail")

    email_input = st.text_input("Введіть вашу e-mail адресу:")
    if st.button("📩 Надіслати результат"):
        if email_input:
            try:
                # Формуємо summary
                avg_overall = agent_stats['avg_overall'].mean().round(2)
                total_tickets = df['conversation_id'].nunique()
                summary_text = (
                    f"Аналітика по агентам\n"
                    f"Середній overall: {avg_overall}\n"
                    f"Кількість тікетів: {total_tickets}\n\n"
                    f"Деталі див. у вкладеному CSV файлі."
                )

                # Формуємо CSV
                csv_text = agent_stats.to_csv(index=False)

                # Відправляємо
                send_result_via_email(summary_text, csv_text, email_input)
                st.success(f"✅ Результат відправлено на {email_input}")
            except Exception as e:
                st.error(f"❌ Помилка при відправці: {e}")
        else:
            st.warning("Будь ласка, введіть e-mail")


