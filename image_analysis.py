from openai import OpenAI
import os
from abc import ABC, abstractmethod


class BaseVLClient(ABC):
    """VL Client基类，用于处理视觉语言模型的请求"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url
        )
    
    @abstractmethod
    def process_request(self, model: str, content: list, **kwargs):
        """处理请求的抽象方法"""
        pass


class ImageAnalysisClient(BaseVLClient):
    """图像分析客户端，用于处理图像输入并获取分析结果"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(api_key, base_url)
    
    def process_request(self, model: str, image_url: str, prompt: str, stream: bool = True, enable_thinking: bool = True, **kwargs):
        """
        处理图像分析请求
        
        Args:
            model: 模型名称
            image_url: 图像URL
            prompt: 分析提示
            stream: 是否使用流式输出
            enable_thinking: 是否启用思考过程
            **kwargs: 其他参数
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            extra_body={
                'enable_thinking': enable_thinking,
                "thinking_budget": 81920
            },
            **kwargs
        )
        
        return completion


def print_stream_response(completion, enable_thinking: bool = True):
    """
    打印流式响应
    
    Args:
        completion: API响应对象
        enable_thinking: 是否启用思考过程
    """
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False    # 判断是否结束思考过程并开始回复

    if enable_thinking:
        print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering is False:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content

    print("\n" + "=" * 20 + "完整思考过程" + "=" * 20 + "\n")
    print(reasoning_content)
    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
    print(answer_content)