from typing import Dict,Any
from backend.knowledge.utils.text_utils import TextUtils

class HtmlParser:
    """专门负责解析 HTML 格式数据成为 Markdown 格式的数据"""

    def parse_html_to_markdown(self, knowledge_no:int, html_data:Dict[str, Any]) -> str:
        """
        解析 HTML 格式数据成为 Markdown 格式的数据
        :param html_data
        :return: Markdown
        """
        # 1. 判断内容是否存在
        if not html_data or not html_data['content']:
            raise ValueError("解析数据不存在")

        # 2. 提取 html_data 中提取 md 中需要的数据
        # 2.1 提取知识库编号，构建知识库
        items = [f"# 知识库 {knowledge_no}\n"] # 知识库的项

        # 2.2 提取知识库标题
        html_data_title = html_data.get('title', "暂无标题")
        items.append(f"## 标题\n{html_data_title.strip()}\n")

        # 2.3 提取摘要，比标题还具有代表性
        html_data_digest = html_data['digest']
        if html_data_digest and html_data_digest.strip():
            items.append(f"## 问题描述\n{html_data_digest.strip()}\n")

        # 2.4 提取知识库分类
        first_topic_name = html_data['firstTopicName']
        sub_topic_name = html_data['subTopicName']
        question_category_name = html_data['questionCategoryName']

        categories = []
        # 主分类 子分类 问题分类
        if first_topic_name and first_topic_name.strip():
            categories.append(f"主分类: {first_topic_name.strip()}\n")
        if sub_topic_name and sub_topic_name.strip():
            categories.append(f"子分类: {sub_topic_name.strip()}\n")
        elif question_category_name and question_category_name.strip():
            categories.append(f"问题分类: {question_category_name.strip()}\n")

        if categories:
            items.append(f"## 分类\n" + "\n".join(categories) + "\n")

        # 2.5 提取知识库关键字【原数据】
        # 打散 清洗再组合
        # 用于相似性检索关键字、提高召回率
        html_data_keywords = html_data['keyWords']
        keyword_list = []
        if html_data_keywords:
            for key_word in html_data_keywords:
                if isinstance(key_word, str):
                    # [ "U盘装系统,U盘系统盘,安装,win7,U盘" ]
                    keyword_list.extend([key_word.strip() for key_word in key_word.split(",") if key_word.strip()])

            if keyword_list:
                keywords = ",".join(keyword_list)
                items.append(f"## 关键词\n{keywords}\n")


        # 2.6 构建元信息（时效性、版本）
        meta_data = []
        html_data_create_time = html_data['createTime']
        html_data_version_no = html_data['versionNo']
        if html_data_create_time and html_data_create_time.strip():
            meta_data.append(f"创建时间:\n{html_data_create_time.strip()} ")
        if html_data_version_no and html_data_version_no.strip():
            meta_data.append(f" 版本\n{html_data_version_no.strip()}")

        if meta_data:
            items.append(f"## 元信息\n" + "|".join(meta_data) + "\n")

        # 2.7 构建内容（解决方案）
        html_data_content = html_data['content']
        if html_data_content:

            # 清洗（将无效内容 从css 广告等清洗）
            md_content = TextUtils.html_to_markdown(html_data_content)

            items.append(f"## 解决方案\n{md_content}\n")

        # 2.8 构建标题作为直属库的注释：防止切块后导致文档上下文丢失
        items.append(f"<!-- 文档主题：{html_data_title} (知识库编号：{knowledge_no}) -->")

        return "\n".join(items)