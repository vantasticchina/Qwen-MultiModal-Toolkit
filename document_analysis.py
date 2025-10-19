from openai import OpenAI
import os
from abc import ABC, abstractmethod
from pathlib import Path


class BaseClient(ABC):
    """Client基类，用于处理文件模型的请求"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url
        )
    
    @abstractmethod
    def process_request(self, model: str, file_path: str, user_prompt: str, **kwargs):
        """处理请求的抽象方法"""
        pass


class DocumentClient(BaseClient):
    """文档理解客户端，用于处理文档文件并获取分析结果"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(api_key, base_url)
    
    def process_request(self, model: str, file_path: str, user_prompt: str = "这篇文章讲了什么？", **kwargs):
        """
        处理文档理解请求
        
        Args:
            model: 模型名称
            file_path: 文档文件路径（支持格式：txt, pdf, docx, pptx, xlsx, html, md）
            user_prompt: 用户查询提示
            **kwargs: 其他参数
        """
        # 检查文件格式
        supported_formats = ['.txt', '.pdf', '.docx', '.pptx', '.xlsx', '.html', '.md']
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in supported_formats:
            raise ValueError(f"不支持的文件格式: {file_ext}。支持的格式: {', '.join(supported_formats)}")
        
        # 上传文件
        file_object = self.client.files.create(file=Path(file_path), purpose="file-extract")
        
        # 创建对话请求
        completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': f'fileid://{file_object.id}'},
                {'role': 'user', 'content': user_prompt}
            ],
            **kwargs
        )
        
        # 删除已上传的文件
        self.client.files.delete(file_object.id)
        
        return completion


# 使用示例
if __name__ == "__main__":
    doc_client = DocumentClient()

    completion = doc_client.process_request(
        model="qwen-long",
        file_path="百炼系列手机产品介绍.docx",
        user_prompt="这篇文章讲了什么？"
    )
    
    print(completion.model_dump_json())