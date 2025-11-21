# -*- coding: utf-8 -*-
"""
Этот модуль реализует функцию, которая запрашивает локальную языковую модель с целью получения ответов на вопросы.
Он использует библиотеку `loguru` для логирования процесса, отслеживая этапы выполнения
и результаты промежуточных вычислений.
Модуль сконфигурирован на работу с моделью обработки естественного языка и позволяет формулировать запросы
в форме промпта, который встраивается в конфигурацию модели для генерации ответов на заданные вопросы.
Основное предназначение этого модуля — выполнение упрощённой генерации текста на основе поступающего
от пользователя вопроса, используя заданную локальную модель.
"""

from loguru import logger

# Настройка логирования с использованием loguru
logger.add("log/Simple_Request_Local_Model.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")

def get_model_response(topic):
    """
    Функция для получения ответа от локальной языковой модели на заданную тему.

    Параметры:
    - topic (str): Вопрос или тема, на которую нужно получить ответ.

    Возвращает:
    - str: Сгенерированный языковой моделью ответ.
    """

    logger.debug('...get_model_response')

    # Загрузка модели для обработки языка (LLM)
    from langchain_ollama import ChatOllama
    logger.debug('LLM')
    # Инициализация локальной языковой модели с параметрами
    local_llm = "qwen2.5-coder:7b"
    llm = ChatOllama(model=local_llm, temperature=0)


    # Промпт
    rag_prompt = """Ты являешься помощником для выполнения заданий по ответам на вопросы. 
    Внимательно подумайте над вопросом пользователя:
    {question}
    Дайте ответ на этот вопрос. 
    Используйте не более трех предложений и будьте лаконичны в ответе.
    Ответ:"""

    # Формирование запроса для LLM
    from langchain_core.messages import HumanMessage
    rag_prompt_formatted = rag_prompt.format(question=topic)

    # Отправка запроса (сформированного промпта) в модель и получение ответа
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    model_response = generation.content
    logger.debug(model_response)
    return model_response

if __name__ == "__main__":
    # Задание темы для запроса к модели
    topic = 'Объясни понятие RAG (Retrieval-Augmented Generation).' # Вопрос пользователя
    logger.debug(topic)

    # Получение ответа от модели на заданную тему
    model_response = get_model_response(topic)