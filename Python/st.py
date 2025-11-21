import Simple_RAG_PDF as sr
import streamlit as st
from loguru import logger
import os

# Настройка заголовка
st.set_page_config(page_title="RAG Assistant", page_icon="🤖")

# Настройка логирования
logger.add("st.log", format="{time} {level} {message}", level="DEBUG")

@st.cache_resource
def load_all():
    try:
        db = sr.get_index_db()
        logger.debug('Данные загружены')
        return db
    except Exception as e:
        st.error(f"Ошибка при загрузке базы данных: {e}")
        return None

# Проверка доступности Ollama
def check_ollama():
    try:
        import requests
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        return response.status_code == 200
    except:
        return False

st.title("🤖 RAG Assistant")
st.write("Задавайте вопросы по вашим документам")

# Информация о доступности моделей
if not check_ollama():
    st.warning("⚠️ Локальная модель Ollama недоступна. Используется демо-режим.")

db = load_all()

if db is None:
    st.error("Не удалось загрузить базу данных документов. Убедитесь, что файлы присутствуют.")
    st.stop()

# Поле ввода
question_input = st.text_input("Введите вопрос:", key="input_text_field")

response_area = st.empty()

if question_input:
    with st.spinner("Ищем ответ..."):
        try:
            logger.debug(f'question_input={question_input}')
            message_content = sr.get_message_content(question_input, db, 3)
            logger.debug(f'message_content={message_content}')
            
            # Проверяем доступность Ollama перед вызовом
            if check_ollama():
                model_response = sr.get_model_response(question_input, message_content)
            else:
                # Демо-режим, если Ollama недоступен
                model_response = f"Демо-ответ на вопрос: '{question_input}'\n\nНа основе найденной информации:\n{message_content[:500]}..."
            
            logger.debug(f'model_response={model_response}')
            response_area.text_area("Ответ", value=model_response, height=400)
            
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")
            logger.error(f"Error: {e}")