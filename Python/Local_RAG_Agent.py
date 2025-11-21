"""
Этот модуль является реализацией интеллектуальной системы поиска и генерации ответов на вопросы пользователей.
Основной функционал модуля заключается в маршрутизации запросов между векторизированным хранилищем документов и
веб-поиском, чтобы предоставить наиболее релевантный ответ.
Система использует модели обработки естественного языка для генерации ответов на вопросы,
оценивает релевантность найденных документов и проверяет "галлюцинации" (информацию, не основанную на фактах).
Основные компоненты функционала включают векторное хранилище, обработку документов, механизм маршрутизации запросов и
алгоритмы оценки качества и актуальности полученных ответов и документов.
"""


import os
from dotenv import load_dotenv
from loguru import logger
import json
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage


# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение API ключа из переменных окружения
# Аккаунт нужно создать здесь: https://tavily.com/
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Настройка логирования с использованием loguru
logger.add("log/Local_RAG_Agent.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")

# Инструкции для маршрутизации запросов
# TODO здесь нужно настроить тему, для примера сказано про математику: 'related to mathematics'
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

    The vectorstore contains documents related to mathematics.

    Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

    Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

# Шаблоны инструкций для проверки документов и генерации ответов
# Doc Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

    This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# RAG prompt
rag_prompt = """You are an assistant for question-answering tasks. 

    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context. 

    Use three sentences maximum and keep the answer concise.

    Answer:"""

hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

hallucination_grader_instructions = """

    You are a teacher grading a quiz. 

    You will be given FACTS and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

answer_grader_instructions = """You are a teacher grading a quiz. 

    You will be given a QUESTION and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) The STUDENT ANSWER helps to answer the QUESTION

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

# Инструмент веб-поиска
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

# Инициализация модели LLM для генерации ответов
logger.debug('LLM')
from langchain_ollama import ChatOllama
local_llm = "qwen2.5-coder:7b"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

def get_index_db():
    """
    Функция для получения или создания векторной Базы-Знаний.
    Если база уже существует, она загружается из файла,
    иначе происходит чтение PDF-документов и создание новой базы.
    """
    logger.debug('...get_index_db')
    # Создание векторных представлений (Embeddings)
    logger.debug('Embeddings')
    from langchain_huggingface import HuggingFaceEmbeddings
    model_id = 'intfloat/multilingual-e5-large'
    model_kwargs = {'device': 'cpu'} # Настройка для использования CPU (можно переключить на GPU)
    # model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs
    )

    db_file_name = 'db/db_01'
    # Загрузка векторной Базы-Знаний из файла
    logger.debug('Загрузка векторной Базы-Знаний из файла')
    file_path = db_file_name + "/index.faiss"
    import os.path
    # Проверка наличия файла с векторной Базой-Знаний
    if os.path.exists(file_path):
        logger.debug('Уже существует векторная База-знаний')
        # Загрузка существующей Базы-Знаний
        db = FAISS.load_local(db_file_name, embeddings, allow_dangerous_deserialization=True)

    else:
        logger.debug('Еще не создана векторная База-Знаний')
        # Если базы нет, происходит создание новой путем чтения PDF-документов
        # Document loaders
        ## Document loaders: https://python.langchain.com/docs/integrations/document_loaders
        ## PyPDFLoader: https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
        from langchain_community.document_loaders import PyPDFLoader

        dir = 'pdf'
        logger.debug(f'Document loaders. dir={dir}')
        documents = []
        # Чтение всех PDF-файлов в указанной директории
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".pdf"):
                    logger.debug(f'root={root} file={file}')
                    loader = PyPDFLoader(os.path.join(root, file))
                    documents.extend(loader.load())

        # Разделение документов на меньшие части (chunks)
        logger.debug('Разделение на chunks')
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        source_chunks = text_splitter.split_documents(documents)
        logger.debug(type(source_chunks))
        logger.debug(len(source_chunks))
        logger.debug(source_chunks[100].metadata)
        logger.debug(source_chunks[100].page_content)

        # Создание векторной Базы-Знаний из chunks
        logger.debug('Векторная База-Знаний')
        db = FAISS.from_documents(source_chunks, embeddings)

        # Сохранение созданной Базы-Знаний в файл
        logger.debug('Сохранение векторной Базы-Знаний в файл')
        db.save_local(db_file_name)

    return db

db = get_index_db()
# Create retriever
logger.debug('Create retriever')
retriever = db.as_retriever(k=3)

# Функция для форматирования документов
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Создание графа состояния для управления потоками данных
import operator
from typing_extensions import TypedDict
from typing import List, Annotated


class GraphState(TypedDict):
    """
    Описание структуры данных состояния графа.
    Состояние графа - это словарь, содержащий информацию, которую мы хотим передавать и изменять в каждом узле графа.
    """

    question: str     # Вопрос пользователя
    generation: str   # LLM генерация
    web_search: str   # Двоичное решение о запуске веб-поиска
    max_retries: int  # Максимальное количество повторных попыток генерации
    answers: int      # Количество сгенерированных ответов
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # Список найденных документов

# Nodes - Определение узлов для каждого этапа работы системы
from langchain.schema import Document
from langgraph.graph import END
def retrieve(state):
    """
    Получение документов из векторного хранилища.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        state (dict): Новый ключ, добавленный в state, documents, который содержит найденные документы
    """
    logger.debug("---ЗАПРОСИТЬ---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    logger.debug(f'documents = {documents}')
    return {"documents": documents}


def generate(state):
    """
    Генерация ответа на основе извлеченных документов

    Args:
        state (dict): Текущее состояние графа

    Returns:
        state (dict): Новый ключ, добавленный в state, generation, которое содержит генерацию LLM
    """
    logger.debug("---СГЕНЕРИРОВАТЬ---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    logger.debug('RAG generation')
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    logger.debug(f'generation={generation}')
    return {"generation": generation, "loop_step": loop_step + 1}


def grade_documents(state):
    """
    Определяет, имеют ли найденные документы отношение к вопросу.
    Если какой-либо документ не является релевантным, мы установим флаг для запуска веб-поиска

    Args:
        state (dict): Текущее состояние графа

    Returns:
        state (dict): Отфильтрованые нерелевантные документы и обновлено состояние web_search state
    """

    logger.debug("---ПРОВЕРЬКА СООТВЕТСТВИЕ ДОКУМЕНТА ВОПРОСУ---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            logger.debug("---ОЦЕНКА: ДОКУМЕНТ РЕЛЕВАНТЕН---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            logger.debug("---ОЦЕНКА: ДОКУМЕНТ НЕ РЕЛЕВАНТЕН---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}


def web_search(state):
    """
    Выполнение веб-поиска по запросу пользователя.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        state (dict): Добавленные найденные в web результаты  в documents
    """

    logger.debug("---ПОИСК в ИНТЕРНЕТЕ---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    logger.debug(f'web_results={web_results}')
    documents.append(web_results)
    return {"documents": documents}


# Определение логики маршрутизации запросов
def route_question(state):
    """
    Маршрутизация вопроса к веб-поиску или векторному хранилищу.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        str: Следующий узел для вызова
    """

    logger.debug("---ВОПРОС МАРШРУТА---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "websearch":
        logger.debug("---НАПРАВИТЬ ВОПРОС НА ПОИСК В ИНТЕРНЕТЕ---")
        return "websearch"
    elif source == "vectorstore":
        logger.debug("---НАПРАВИТЬ ВОПРОС В RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Решение о том, по какому пути продолжить генерацию ответа.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        str: Двоичное решение для следующего узла вызова
    """

    logger.debug("---ОЦЕНИВИТЬ GRADED ДОКУМЕНТЫ---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.debug(
            "---РЕШЕНИЕ: НЕ ВСЕ ДОКУМЕНТЫ ИМЕЮТ ОТНОШЕНИЕ К ВОПРОСУ, ВКЛЮЧИТЕ ВЕБ-ПОИСК---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        logger.debug("---РЕШЕНИЕ: ГЕНЕРИРОВАТЬ---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Проверка соответствия сгенерированного ответа документам и вопросу.

    Args:
        state (dict): Текущее состояние графа

    Returns:
        str: Решение для следующего узла для вызова
    """

    logger.debug("---ПРОВЕРИТЬ ГАЛЛЮЦИНАЦИИ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    max_retries = state.get("max_retries", 3)  # Default to 3 if not provided

    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents), generation=generation.content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)["binary_score"]

    # Check hallucination
    if grade == "yes":
        logger.debug("---РЕШЕНИЕ: ГЕНЕРАЦИЯ ОСНОВАНА НА ДОКУМЕНТАХ---")
        # Check question-answering
        logger.debug("---Оценка: ГЕНЕРАЦИЯ против ВОПРОСА---")
        # Test using question and generation from above
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation.content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        if grade == "yes":
            logger.debug("---РЕШЕНИЕ: GENERATION ОБРАЩАЕТСЯ К ВОПРОСУ---")
            return "useful"
        elif state["loop_step"] <= max_retries:
            logger.debug("---РЕШЕНИЕ: GENERATION НЕ ОТВЕЧАЕТ НА ВОПРОС---")
            return "not useful"
        else:
            logger.debug("---РЕШЕНИЕ: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОВТОРНЫХ ПОПЫТОК ДОСТИГНУТО---")
            return "max retries"
    elif state["loop_step"] <= max_retries:
        logger.debug("---РЕШЕНИЕ: ГЕНЕРАЦИЯ НЕ ОСНОВАНА НА ДОКУМЕНТАХ, ПОВТОРИТЕ ПОПЫТКУ---")
        return "not supported"
    else:
        logger.debug("---РЕШЕНИЕ: МАКСИМАЛЬНОЕ КОЛИЧЕСТВО ПОВТОРНЫХ ПОПЫТОК ДОСТИГНУТО---")
        return "max retries"

# Control Flow - управление потоками выполнения
from langgraph.graph import StateGraph
from IPython.display import Image, display

workflow = StateGraph(GraphState)

# Определение узлов
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

# Построение графа
workflow.set_conditional_entry_point(
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# Компиляция графа
graph = workflow.compile()
# graph_image = Image(graph.get_graph().draw_mermaid_png())
# display(graph_image)

# Сохраняем картинку в файл
graph_image = graph.get_graph().draw_mermaid_png()
with open("../graph_image.png", "wb") as png:
    png.write(graph_image)


# Открытие и отображение изображения
from PIL import Image as PILImage
import io
img = PILImage.open("../graph_image.png")
img.show()

if __name__ == "__main__":
    inputs = {"question": "О чем теорема Ферма? Для чего ее используют?", "max_retries": 3}

    for event in graph.stream(inputs, stream_mode="values"):
        logger.debug(event)

    # inputs = {"question": "О чем теорема Геделя о неполноте? Для чего ее используют?", "max_retries": 3}
    #
    # for event in graph.stream(inputs, stream_mode="values"):
    #     logger.debug(event)