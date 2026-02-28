import os
from http.client import HTTPException
from backend.knowledge.services.crawler.parser import HtmlParser
import requests
from backend.knowledge.config.settings import settings


class KnowledgeAPIClient:
    """提供一个方法 获取网络知识"""

    @staticmethod
    def fetch_knowledge(knowledge_no: int) -> str:
        """根据知识库编号 获取联想知识库"""
        try:
            # 1. 定义 URL
            # knowledge_base_url=settings.KNOWLEDGE_BASE_URL
            knowledge_base_url = f"{settings.KNOWLEDGE_BASE_URL}/knowledgeapi/api/knowledge/knowledgeDetails"

            # 2. 定义 param
            params = {"knowledgeNo": knowledge_no}
            # 3. 发送请求
            response = requests.get(url=knowledge_base_url, params=params, timeout=10)
            response.raise_for_status()

            # 4. 得到结果字典 知识库内容
            response_dict = response.json()

            # 5. 获取 data
            return response_dict['data']
        except HTTPException as e:
            raise HTTPException(f"发送知识库请求失败：{e}")


if __name__ == '__main__':
    content = KnowledgeAPIClient.fetch_knowledge(knowledge_no=1)
    print(f"知识库内容：/n{content}")

    parser = HtmlParser()
    md_content = parser.parse_html_to_markdown(1, content)

    file_name_path = os.path.dirname(__file__)
    file_name = os.path.join(file_name_path, "test_01.md")

    with open(file_name, "w", encoding='utf-8') as f:
        f.write(md_content)
