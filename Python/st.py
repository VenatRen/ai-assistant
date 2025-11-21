import Simple_RAG_PDF as sr
import streamlit as st
from loguru import logger

# Настройка логирования с использованием loguru
logger.add("log/st.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")

@st.cache_data
def load_all():
    db = sr.get_index_db()
    logger.debug('Данные загружены')
    return db

db = load_all()

# Поле ввода
question_input = st.text_input("Введите вопрос: ", key="input_text_field")

response_area = st.empty()


if question_input:
    logger.debug(f'question_input={question_input}')
    message_content = sr.get_message_content(question_input, db, 3)
    logger.debug(f'message_content={message_content}')
    model_response = sr.get_model_response(question_input, message_content)
    logger.debug(f'message_content={model_response}')
    response_area.text_area("Ответ", value=model_response, height=400)






