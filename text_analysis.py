from openai import OpenAI
import os
from abc import ABC, abstractmethod


class BaseClient(ABC):
    """Client基类，用于处理文本模型的请求"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url
        )
    
    @abstractmethod
    def process_request(self, model: str, messages: list, **kwargs):
        """处理请求的抽象方法"""
        pass


class TextClient(BaseClient):
    """文本客户端，用于处理纯文本输入并获取分析结果"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(api_key, base_url)
    
    def process_request(self, model: str, messages: list, **kwargs):
        """
        处理文本请求
        
        Args:
            model: 模型名称
            messages: 对话消息列表
            **kwargs: 其他参数
        """
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        return completion


# 使用示例
if __name__ == "__main__":
    text_client = TextClient()

    completion = text_client.process_request(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你是谁？"},
        ]
    )
    
    print(completion.model_dump_json())