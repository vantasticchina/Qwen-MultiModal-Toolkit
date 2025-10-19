from image_analysis import ImageAnalysisClient, print_stream_response
from video_analysis import VideoAnalysisClient
from ocr_extraction import OCRClient
from text_analysis import TextClient
from document_analysis import DocumentClient

def main():
    print("选择要运行的模式:")
    print("1. 图像分析")
    print("2. 视频分析")
    print("3. OCR结构化文本提取")
    print("4. 文本对话")
    print("5. 文档理解")
    
    choice = input("请输入选择 (1, 2, 3, 4 或 5): ")
    
    if choice == "1":
        # 图像分析
        print("选择图像分析类型:")
        print("1. 解答题目")
        print("2. 提取文本")
        sub_choice = input("请输入选择 (1 或 2): ")
        
        image_client = ImageAnalysisClient()
        if sub_choice == "1":
            completion = image_client.process_request(
                model="qwen3-vl-plus",
                image_url="https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_!!6000000002727-0-tps-1024-406.jpg",
                prompt="这道题怎么解答？",
                stream=True,
                enable_thinking=True
            )
        elif sub_choice == "2":
            completion = image_client.process_request(
                model="qwen-vl-max-latest",
                image_url="https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241108/ctdzex/biaozhun.jpg",
                prompt="请仅输出图像中的文本内容。",
                stream=True,
                enable_thinking=False  # 文本提取不需要思考过程
            )
        else:
            print("无效选择")
            return
        
        print_stream_response(completion, enable_thinking=(sub_choice == "1"))
    
    elif choice == "2":
        # 视频分析
        video_client = VideoAnalysisClient()
        video_frames = [
            "https://img.alicdn.com/imgextra/i3/O1CN01K3SgGo1eqmlUgeE9b_!!6000000003923-0-tps-3840-2160.jpg",
            "https://img.alicdn.com/imgextra/i4/O1CN01BjZvwg1Y23CF5qIRB_!!6000000003000-0-tps-3840-2160.jpg",
            "https://img.alicdn.com/imgextra/i4/O1CN01Ib0clU27vTgBdbVLQ_!!6000000007859-0-tps-3840-2160.jpg",
            "https://img.alicdn.com/imgextra/i1/O1CN01aygPLW1s3EXCdSN4X_!!6000000005710-0-tps-3840-2160.jpg"
        ]
        completion = video_client.process_request(
            model="qwen-vl-max-latest",
            video_urls=video_frames,
            prompt="描述这个视频的具体过程",
        )
        print(completion.model_dump_json())
    
    elif choice == "3":
        # OCR结构化文本提取
        print("选择OCR类型:")
        print("1. 发票信息提取")
        print("2. 车票信息提取")
        ocr_choice = input("请输入选择 (1 或 2): ")
        
        ocr_client = OCRClient()
        if ocr_choice == "1":
            completion = ocr_client.process_request(
                model="qwen-vl-ocr-latest",
                image_url="https://prism-test-data.oss-cn-hangzhou.aliyuncs.com/image/car_invoice/car-invoice-img00040.jpg"
            )
        elif ocr_choice == "2":
            from ocr_extraction import extract_ticket_info
            completion = extract_ticket_info(
                client=ocr_client,
                image_url="https://img.alicdn.com/imgextra/i2/O1CN01ktT8451iQutqReELT_!!6000000004408-0-tps-689-487.jpg"
            )
        else:
            print("无效选择")
            return
        
        print(completion.choices[0].message.content)
    
    elif choice == "4":
        # 文本对话
        text_client = TextClient()
        completion = text_client.process_request(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "你是谁？"},
            ]
        )
        print(completion.model_dump_json())
    
    elif choice == "5":
        # 文档理解
        print("请输入文档文件路径:")
        file_path = input().strip()
        print("请输入查询内容 (直接回车使用默认查询):")
        user_prompt = input().strip()
        if not user_prompt:
            user_prompt = "这篇文章讲了什么？"
        
        doc_client = DocumentClient()
        try:
            completion = doc_client.process_request(
                model="qwen-long",
                file_path=file_path,
                user_prompt=user_prompt
            )
            print(completion.model_dump_json())
        except Exception as e:
            print(f"处理文档时出错: {e}")
    
    else:
        print("无效选择")


if __name__ == "__main__":
    main()