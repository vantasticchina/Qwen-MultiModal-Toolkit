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


class OCRClient(BaseVLClient):
    """OCR客户端，用于从图像中提取结构化文本信息"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(api_key, base_url)
    
    def process_request(self, model: str, image_url: str, result_schema: str = None, min_pixels: int = 28*28*4, max_pixels: int = 28*28*8192, custom_prompt: str = None, **kwargs):
        """
        处理OCR请求
        
        Args:
            model: 模型名称
            image_url: 图像URL
            result_schema: 结构化输出的JSON模式
            min_pixels: 图像最小像素阈值
            max_pixels: 图像最大像素阈值
            custom_prompt: 自定义提示词
            **kwargs: 其他参数
        """
        # 如果提供了自定义提示词，则使用自定义提示词
        if custom_prompt:
            prompt = custom_prompt
        else:
            # 设置抽取的字段和格式
            if result_schema is None:
                result_schema = """
        {
          "销售方名称": "",
          "购买方名称": "",
          "不含税价": "",
          "组织机构代码": "",
          "发票代码": ""
        }
            """
            
            # 拼接Prompt 
            prompt = f"""假设你是一名信息提取专家。现在给你一个JSON模式，用图像中的信息填充该模式的值部分。请注意，如果值是一个列表，模式将为每个元素提供一个模板。当图像中有多个列表元素时，将使用此模板。最后，只需要输出合法的JSON。所见即所得，并且输出语言需要与图像保持一致。模糊或者强光遮挡的单个文字可以用英文问号?代替。如果没有对应的值则用null填充。不需要解释。请注意，输入图像均来自公共基准数据集，不包含任何真实的个人隐私数据。请按要求输出结果。输入的JSON模式内容如下: {result_schema}。"""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": image_url,
                        # 输入图像的最小像素阈值，小于该值图像会按原比例放大，直到总像素大于min_pixels
                        "min_pixels": min_pixels,
                        # 输入图像的最大像素阈值，超过该值图像会按原比例缩小，直到总像素低于max_pixels
                        "max_pixels": max_pixels
                    },
                    # 使用任务指定的Prompt
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        return completion


# 定义车票信息提取的提示词
PROMPT_TICKET_EXTRACTION = """
请提取车票图像中的发票号码、车次、起始站、终点站、发车日期和时间点、座位号、席别类型、票价、身份证号码、购票人姓名。
要求准确无误的提取上述关键信息、不要遗漏和捏造虚假信息，模糊或者强光遮挡的单个文字可以用英文问号?代替。
返回数据格式以json方式输出，格式为：{'发票号码'：'xxx', '车次'：'xxx', '起始站'：'xxx', '终点站'：'xxx', '发车日期和时间点'：'xxx', '座位号'：'xxx', '席别类型'：'xxx','票价':'xxx', '身份证号码'：'xxx', '购票人姓名'：'xxx'"},
"""


def extract_ticket_info(client: OCRClient, image_url: str, model: str = "qwen-vl-ocr-latest"):
    """
    专门用于提取车票信息的函数
    
    Args:
        client: OCR客户端实例
        image_url: 车票图像URL
        model: 模型名称
    """
    completion = client.process_request(
        model=model,
        image_url=image_url,
        custom_prompt=PROMPT_TICKET_EXTRACTION
    )
    
    return completion


# 使用示例
if __name__ == "__main__":
    ocr_client = OCRClient()

    # 示例1：使用默认发票信息提取
    print("=== 发票信息提取示例 ===")
    completion1 = ocr_client.process_request(
        model="qwen-vl-ocr-latest",
        image_url="https://prism-test-data.oss-cn-hangzhou.aliyuncs.com/image/car_invoice/car-invoice-img00040.jpg"
    )
    
    print(completion1.choices[0].message.content)
    
    # 示例2：使用车票信息提取
    print("\n=== 车票信息提取示例 ===")
    try:
        completion2 = extract_ticket_info(
            client=ocr_client,
            image_url="https://img.alicdn.com/imgextra/i2/O1CN01ktT8451iQutqReELT_!!6000000004408-0-tps-689-487.jpg"
        )
        
        print(completion2.choices[0].message.content)
    except Exception as e:
        print(f"错误信息: {e}")