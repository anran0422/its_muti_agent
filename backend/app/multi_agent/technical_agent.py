import asyncio

from agents import Agent, ModelSettings, Runner, RunConfig

from infrastructure.ai.prompt_loader import load_prompt
from infrastructure.ai.openai_client import sub_model
from infrastructure.tools.local.knowledge_base import query_knowledge
from infrastructure.tools.mcp.mcp_servers import search_mcp_client

# 1. 定义技术智能体
technical_agent = Agent(
    name="咨询与技术专家",
    instructions=load_prompt("technical_agent"),
    model=sub_model,
    model_settings=ModelSettings(temperature=0),  # 不要发挥内容（软件层面限制模型的发挥）
    tools=[query_knowledge],
    mcp_servers=[search_mcp_client]
)


async def run_single_test(case_name: str, input_text: str):
    print(f"\n{'=' * 80}")
    print(f"测试用例: {case_name}")
    print(f"输入: \"{input_text}\"")
    print("-" * 80)
    try:
        await search_mcp_client.connect()
        print("思考中...")
        result = await Runner.run(technical_agent, input=input_text, run_config=RunConfig(tracing_disabled=True))

        print(f"\n\nAgent的最终输出: {result.final_output}")
    except Exception as e:
        print(f"\n Error: {e}\n")
    finally:
        try:
            await search_mcp_client.cleanup()
        except:
            pass


async def main():
    # 技术问题测试案例
    test_cases = [
        # ("Case 1: 技术问题", "如何使用U盘安装Windows 7操作系统"),
        # ("Case 2: 实时问题", "今天小米股价"),
        # ("Case 3  技术问题", "导航去武汉步行街"),  # 应拒绝
        ("Case 4  闲聊", "你好啊"),  # 应拒绝
    ]

    for name, question in test_cases:
        await run_single_test(name, question)


if __name__ == "__main__":
    asyncio.run(main())