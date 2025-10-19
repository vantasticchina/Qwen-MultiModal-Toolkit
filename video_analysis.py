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


class VideoAnalysisClient(BaseVLClient):
    """视频分析客户端，用于处理视频输入并获取分析结果"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(api_key, base_url)
    
    def process_request(self, model: str, video_urls: list, prompt: str, **kwargs):
        """
        处理视频分析请求
        
        Args:
            model: 模型名称
            video_urls: 视频帧URL列表
            prompt: 分析提示
            **kwargs: 其他参数
        """
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_urls
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]
        
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        return completion


# 使用示例
if __name__ == "__main__":
    # 创建视频分析客户端
    video_client = VideoAnalysisClient()
    
    # 定义视频帧URL列表
    video_frames = [
        "https://img.alicdn.com/imgextra/i3/O1CN01K3SgGo1eqmlUgeE9b_!!6000000003923-0-tps-3840-2160.jpg",
        "https://img.alicdn.com/imgextra/i4/O1CN01BjZvwg1Y23CF5qIRB_!!6000000003000-0-tps-3840-2160.jpg",
        "https://img.alicdn.com/imgextra/i4/O1CN01Ib0clU27vTgBdbVLQ_!!6000000007859-0-tps-3840-2160.jpg",
        "https://img.alicdn.com/imgextra/i1/O1CN01aygPLW1s3EXCdSN4X_!!6000000005710-0-tps-3840-2160.jpg"
    ]
    
    # 创建视频分析请求
    completion = video_client.process_request(
        model="qwen-vl-max-latest",
        video_urls=video_frames,
        prompt="描述这个视频的具体过程",
    )
    
    # 输出结果
    print(completion.model_dump_json())