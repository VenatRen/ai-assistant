# -*- coding: utf-8 -*-
""" Описание модуля
Этот модуль реализует метод генерации ответа на заданную тему, используя модель обучения на языковых примерах.
Основные шаги включают загрузку и обработку PDF-документов, создание векторной Базы-Знаний для поиска по схожести содержимого
и использование модели для генерации ответа.
Векторная База-Знаний хранится и загружается с локального диска для ускорения работы.

"""


import os
from loguru import logger
from langchain_community.embeddings import HuggingFaceEmbeddings

# Настройка логирования с использованием loguru
logger.add("log/02_Simple_RAG_PDF.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")


def get_index_db():
    """
    Функция для получения или создания векторной Базы-Знаний.
    """
    logger.debug('...get_index_db')
    
    try:
        # Используем совместимые эмбеддинги
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Используем меньшую модель для совместимости
        model_id = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        db_file_name = 'db/db_01'
        file_path = db_file_name + "/index.faiss"
        
        # В облачной среде всегда пересоздаем базу (данные временные)
        logger.debug('Создание/обновление векторной Базы-Знаний для облачной среды')
        
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.document_loaders import Docx2txtLoader
        import os

        dir = 'pdf'
        logger.debug(f'Document loaders. dir={dir}')
        documents = []
        current_files = []
        
        # Проверяем, есть ли файлы в папке pdf
        if not os.path.exists(dir):
            os.makedirs(dir)
            logger.warning(f'Создана пустая папка {dir}. Добавьте PDF/DOCX файлы.')
            
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith(".pdf"):
                    logger.debug(f'Обработка PDF: {file_path}')
                    try:
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                        current_files.append(file_path)
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке PDF {file_path}: {e}")
                        
                elif file.endswith(".docx"):
                    logger.debug(f'Обработка DOCX: {file_path}')
                    try:
                        loader = Docx2txtLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        current_files.append(file_path)
                        logger.debug(f"Успешно загружен DOCX файл: {file_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке DOCX файла {file_path}: {e}")

        if not documents:
            logger.warning('Не найдено ни одного файла для обработки!')
            from langchain_core.documents import Document
            # Создаем демо-документ для тестирования
            documents = [Document(
                page_content="Это демонстрационный документ. Добавьте PDF или DOCX файлы в папку 'pdf' для работы с реальными документами.", 
                metadata={"source": "demo.txt"}
            )]

        # Разделение документов на меньшие части (chunks)
        logger.debug('Разделение на chunks')
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        source_chunks = text_splitter.split_documents(documents)
        logger.debug(f'Количество chunks: {len(source_chunks)}')

        logger.debug('Создание векторной Базы-Знаний')
        from langchain_community.vectorstores import FAISS
        db = FAISS.from_documents(source_chunks, embeddings)

        logger.debug('Сохранение векторной Базы-Знаний в файл')
        os.makedirs(db_file_name, exist_ok=True)
        db.save_local(db_file_name)
        
        # Сохраняем список обработанных файлов
        processed_files_file = db_file_name + "/processed_files.txt"
        with open(processed_files_file, 'w', encoding='utf-8') as f:
            for file_path in current_files:
                f.write(file_path + '\n')
        logger.debug(f'Список обработанных файлов сохранен в {processed_files_file}')

        return db
        
    except Exception as e:
        logger.error(f"Ошибка при создании базы данных: {e}")
        # Возвращаем минимальную рабочую базу в случае ошибки
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
        
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        demo_docs = [Document(
            page_content="Система временно недоступна. Проверьте логи для подробностей.", 
            metadata={"source": "error.txt"}
        )]
        return FAISS.from_documents(demo_docs, embeddings)

def get_message_content(topic, db, NUMBER_RELEVANT_CHUNKS):
    """
        Функция для извлечения релевантных кусочков текста из Базы-Знаний.
        Выполняется поиск по схожести, извлекаются top-N релевантных частей.
    """
    # Similarity search
    import re
    logger.debug('...get_message_content: Similarity search')
    docs = db.similarity_search(topic, k = NUMBER_RELEVANT_CHUNKS)
    # Форматирование извлеченных данных
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n#### {i+1} Relevant chunk ####\n' + str(doc.metadata) + '\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    logger.debug(message_content)
    return message_content


def get_model_response(topic, message_content):
    """
    Функция для генерации ответа модели на основе переданного контекста и вопроса.
    """
    logger.debug('...get_model_response')

    try:
        # Загрузка модели для обработки языка (LLM)
        from langchain_ollama import ChatOllama
        logger.debug('LLM')
        local_llm = "qwen2.5-coder:7b"
        
        # Добавляем таймауты и обработку ошибок
        llm = ChatOllama(
            model=local_llm, 
            temperature=0,
            base_url='http://localhost:11434',
            timeout=30.0
        )

        # Промпт
        rag_prompt = """Ты являешься помощником для выполнения заданий по ответам на вопросы. 
        Вот контекст, который нужно использовать для ответа на вопрос:
        {context} 
        Внимательно подумайте над приведенным контекстом. 
        Теперь просмотрите вопрос пользователя:
        {question}
        Дайте ответ на этот вопрос, используя только вышеуказанный контекст. 
        Используйте не более трех предложений и будьте лаконичны в ответе.
        Ответ:"""

        # Формирование запроса для LLM
        from langchain_core.messages import HumanMessage
        rag_prompt_formatted = rag_prompt.format(context=message_content, question=topic)
        generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
        model_response = generation.content
        logger.debug(model_response)
        return model_response
        
    except Exception as e:
        logger.error(f"Ошибка при обращении к модели: {e}")
        # Возвращаем демо-ответ в случае ошибки
        return f"На основе предоставленного контекста можно сделать вывод, что ответ на вопрос '{topic}' содержится в документах. Для получения точного ответа убедитесь, что локальная модель Ollama запущена."