import Simple_RAG_PDF as sr
import streamlit as st
from loguru import logger
import subprocess
import threading
import time
import requests
import os

# Настройка логирования
logger.remove()
logger.add("log/st.log", format="{time} {level} {message}", level="DEBUG", enqueue=True)

class TunnelManager:
    def __init__(self):
        self.public_url = None
        self.process = None
        self.subdomain = "rag-assistant-" + str(int(time.time()))[-6:]
        
    def start_tunnel(self):
        """Запуск LocalTunnel туннеля"""
        try:
            # Запускаем localtunnel в отдельном процессе
            self.process = subprocess.Popen(
                ["lt", "--port", "8501", "--subdomain", self.subdomain],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Даем время на запуск
            time.sleep(5)
            
            # Пытаемся получить URL
            self.public_url = f"https://{self.subdomain}.loca.lt"
            
            # Проверяем доступность
            if self.check_tunnel_availability():
                logger.info(f"LocalTunnel туннель запущен: {self.public_url}")
                return self.public_url
            else:
                logger.warning("Туннель запущен, но недоступен для проверки")
                return self.public_url
                
        except Exception as e:
            logger.error(f"Ошибка запуска туннеля: {e}")
            return None
    
    def check_tunnel_availability(self):
        """Проверка доступности туннеля"""
        try:
            response = requests.get(self.public_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def stop_tunnel(self):
        """Остановка туннеля"""
        if self.process:
            self.process.terminate()
            logger.info("Туннель остановлен")

# Создаем менеджер туннелей
tunnel_manager = TunnelManager()

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