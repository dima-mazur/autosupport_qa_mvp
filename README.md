
# Support QA — ONE CSV (Simple)

Один CSV, що містить: `conversation_id, topic, user_text, agent_text, created_at_user, created_at_agent, resolved, user_csat_0_5, agent_name`

Метрики: speed, empathy, correctness, resolution_score (0/10), csat_score(0..10), overall = середнє пʼяти.

Візуалізації:
- Розподілення overall rating
- Середнє по метрикам
- Resolved rate (загальний)
- Розподілення CSAT (0..5)
- Breakdown по темам: кількість, середній overall rating
- TTFR (сек) гістограмма
- Агентська аналітика: таблиця, барчарт avg overall rating по агентам

Запуск:
```
pip install -r requirements.txt
streamlit run app.py
```
