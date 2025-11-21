# -*- coding: utf-8 -*-
""" Описание модуля
Этот модуль реализует метод генерации ответа на заданную тему, используя модель обучения на языковых примерах.
Основные шаги включают загрузку и обработку PDF-документов, создание векторной Базы-Знаний для поиска по схожести содержимого
и использование модели для генерации ответа.
Векторная База-Знаний хранится и загружается с локального диска для ускорения работы.

"""


import os
from loguru import logger
from langchain_community.vectorstores import FAISS

# Настройка логирования с использованием loguru
logger.add("log/02_Simple_RAG_PDF.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")


def get_index_db():
    """
    Функция для получения или создания векторной Базы-Знаний.
    """
    logger.debug('...get_index_db')
    
    from langchain_huggingface import HuggingFaceEmbeddings
    model_id = 'intfloat/multilingual-e5-large'
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs
    )

    db_file_name = 'db/db_01'
    file_path = db_file_name + "/index.faiss"
    
    # Проверяем, нужно ли пересоздать базу
    need_rebuild = True
    
    if os.path.exists(file_path):
        logger.debug('Векторная База-знаний существует')
        
        current_files = []
        for root, dirs, files in os.walk('pdf'):
            for file in files:
                if file.endswith(".pdf") or file.endswith(".docx"):
                    current_files.append(os.path.join(root, file))
        
        processed_files_file = db_file_name + "/processed_files.txt"
        if os.path.exists(processed_files_file):
            with open(processed_files_file, 'r', encoding='utf-8') as f:
                processed_files = set(line.strip() for line in f.readlines())
            
            current_files_set = set(current_files)
            if current_files_set == processed_files:
                logger.debug('Новых файлов не обнаружено, загружаем существующую базу')
                db = FAISS.load_local(db_file_name, embeddings, allow_dangerous_deserialization=True)
                need_rebuild = False
            else:
                logger.debug('Обнаружены новые или измененные файлы, перестраиваем базу')
        else:
            logger.debug('Файл с метаданными обработанных файлов не найден, перестраиваем базу')
    else:
        logger.debug('Векторная База-знаний не существует, создаем новую')

    if need_rebuild:
        logger.debug('Создание/обновление векторной Базы-Знаний')
        
        from langchain_community.document_loaders import PyPDFLoader
        # Используем простой загрузчик для DOCX без зависимостей от NLTK
        from langchain_community.document_loaders import Docx2txtLoader

        dir = 'pdf'
        logger.debug(f'Document loaders. dir={dir}')
        documents = []
        current_files = []
        
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith(".pdf"):
                    logger.debug(f'Обработка PDF: {file_path}')
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    current_files.append(file_path)
                    
                elif file.endswith(".docx"):
                    logger.debug(f'Обработка DOCX: {file_path}')
                    try:
                        # Используем Docx2txtLoader вместо UnstructuredWordDocumentLoader
                        loader = Docx2txtLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
                        current_files.append(file_path)
                        logger.debug(f"Успешно загружен DOCX файл: {file_path}")
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке DOCX файла {file_path}: {e}")
                        # Альтернатива: используем python-docx напрямую
                        try:
                            import docx
                            doc = docx.Document(file_path)
                            full_text = []
                            for paragraph in doc.paragraphs:
                                full_text.append(paragraph.text)
                            from langchain_core.documents import Document
                            doc_content = '\n'.join(full_text)
                            documents.append(Document(page_content=doc_content, metadata={"source": file_path}))
                            current_files.append(file_path)
                            logger.debug(f"DOCX файл загружен через python-docx: {file_path}")
                        except Exception as e2:
                            logger.error(f"Ошибка при загрузке через python-docx {file_path}: {e2}")

        if not documents:
            logger.warning('Не найдено ни одного файла для обработки!')
            from langchain_core.documents import Document
            documents = [Document(page_content="Нет данных", metadata={})]

        # Разделение документов на меньшие части (chunks)
        logger.debug('Разделение на chunks')
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        source_chunks = text_splitter.split_documents(documents)
        logger.debug(f'Тип chunks: {type(source_chunks)}')
        logger.debug(f'Количество chunks: {len(source_chunks)}')
        
        if len(source_chunks) > 100:
            logger.debug(source_chunks[100].metadata)
            logger.debug(source_chunks[100].page_content)
        elif source_chunks:
            last_index = len(source_chunks) - 1
            logger.debug(f"Последний элемент (индекс {last_index}): {source_chunks[last_index].metadata}")
            logger.debug(f"Содержимое последнего элемента: {source_chunks[last_index].page_content}")

        logger.debug('Создание векторной Базы-Знаний')
        if os.path.exists(file_path) and 'db' in locals():
            logger.debug('Добавление новых документов в существующую базу')
            db.add_documents(source_chunks)
        else:
            db = FAISS.from_documents(source_chunks, embeddings)

        logger.debug('Сохранение векторной Базы-Знаний в файл')
        db.save_local(db_file_name)
        
        os.makedirs(db_file_name, exist_ok=True)
        processed_files_file = db_file_name + "/processed_files.txt"
        with open(processed_files_file, 'w', encoding='utf-8') as f:
            for file_path in current_files:
                f.write(file_path + '\n')
        logger.debug(f'Список обработанных файлов сохранен в {processed_files_file}')

    return db

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
        Используется LLM для создания ответа, используя переданный контекст.
    """
    logger.debug('...get_model_response')

    # Загрузка модели для обработки языка (LLM)
    from langchain_ollama import ChatOllama
    logger.debug('LLM')
    local_llm = "qwen2.5-coder:7b"
    llm = ChatOllama(model=local_llm, temperature=0)


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

if __name__ == "__main__":
    # Основной блок программы: инициализация, построение базы и генерация ответа
    db = get_index_db()
    NUMBER_RELEVANT_CHUNKS = 3 # Количество релевантных кусков для извлечения
    topic = 'Какая концентрация была обнаружена при травме конечности' # Вопрос пользователя
    logger.debug(topic)
    message_content = get_message_content(topic, db, NUMBER_RELEVANT_CHUNKS)
    model_response = get_model_response(topic, message_content)