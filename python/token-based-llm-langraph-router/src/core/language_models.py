from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from config.config_loader import load_config


class LanguageModels:

    def __init__(self):
        """Initialize the language models"""
        self.config = load_config()
        self.router_llm = ChatAnthropic(api_key=self.config["api_keys"]["anthropic"], model=self.config["models"]["evaluator"])
        self.advanced_llm = ChatAnthropic(api_key=self.config["api_keys"]["anthropic"], model=self.config["models"]["advanced"])
        self.simple_llm = ChatOpenAI(api_key=self.config["api_keys"]["open_ai"], model=self.config["models"]["simple"])

    def get_router_llm(self):
        return self.router_llm

    def get_advanced_llm(self):
        return self.advanced_llm

    def get_simple_llm(self):
        return self.simple_llm
