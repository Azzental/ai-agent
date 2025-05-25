import time
import functools
from typing import Dict, Callable

from dotenv import dotenv_values
from langchain_gigachat.chat_models import GigaChat
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from database.data import stuff_database


def green_lightner(func: Callable) -> Callable:
    """Подсвечивает вызов функции зелёным цветом"""
    @functools.wraps(func) # Переносит метаданные
    def wrapper(*args, **kwargs):
        print("\033[92m" + f"Bot requested {func.__name__}()" + "\033[0m")
        return func(*args, **kwargs)
    return wrapper


@tool
@green_lightner
def get_all_phone_names() -> str:
    """Возвращает названия моделей всех телефонов ф формате json"""
    return ", ".join([stuff["name"] for stuff in stuff_database])


@tool
@green_lightner
def get_phone_data_by_name(name: str) -> Dict:
    """
    Возвращает цену в долларах, характеристики и описание телефона по точному названию модели.

    Args:
        name (str): Точное название модели телефона.

    Returns:
        Dict: Словарь с информацией о телефоне (цена, характеристики и описание).
    """
    for stuff in stuff_database:
        if stuff["name"] == name.strip():
            return stuff

    return {"error": "Телефон с таким названием не найден"}


@tool
@green_lightner
def create_order(name: str, phone: str) -> None:
    """
    Создает новый заказ на телефон.

    Args:
        name (str): Название телефона.
        phone (str): Телефонный номер пользователя.

    Returns:
        str: Статус заказа.
    """
    print(f"!!! NEW ORDER !!! {name} {phone}")


system_prompt = '''Ты бот-продавец телефонов. Твоя задача продать телефон пользователю, 
            получив от него заказ. 
            Если тебе не хватает каких-то данных, запрашивай их у пользователя.'''

memory = MemorySaver()

config = dotenv_values(".env")

model = GigaChat(
    credentials=config.get("GIGACHAT_KEY"),
    scope=config.get("GIGACHAT_SCOPE"),
    model=config.get("GIGACHAT_MODEL"),
    verify_ssl_certs=False,
)

tools = [create_order, get_phone_data_by_name,get_all_phone_names]

agent = create_react_agent(model,
                           tools=tools,
                           checkpointer=MemorySaver(),
                           prompt=system_prompt)


def chat(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    print("Для выхода ничего не вводите и нажмите Enter.")
    while(True):
        rq = input("\nHuman: ")
        print("User: ", rq)
        if rq == "":
            break
        resp = agent.invoke({"messages": [("user", rq)]}, config=config)
        print("Assistant: ", resp["messages"][-1].content)
        time.sleep(1) # For notebook capability


def main():
    chat("123456")


if __name__ == "__main__":
    main()
