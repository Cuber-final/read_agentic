from __future__ import annotations

import logging
from string import Template
from typing import Any, Dict, List, Optional, Union

from autochain.agent.base_agent import BaseAgent
from autochain.agent.message import ChatMessageHistory, SystemMessage, UserMessage
from autochain.agent.openai_functions_agent.output_parser import (
    OpenAIFunctionOutputParser,
)
from autochain.agent.openai_functions_agent.prompt import ESTIMATE_CONFIDENCE_PROMPT
from autochain.agent.structs import AgentAction, AgentFinish
from autochain.models.base import BaseLanguageModel, Generation
from autochain.tools.base import Tool
from autochain.utils import print_with_color
from colorama import Fore

# 设置日志记录器
logger = logging.getLogger(__name__)


class OpenAIFunctionsAgent(BaseAgent):
    """
    支持OpenAI原生函数调用的Agent，利用函数消息来确定应该使用哪个工具
    当未选择工具时，表现得像对话式Agent
    工具描述从工具的类型定义中生成
    """

    llm: BaseLanguageModel = None  # 语言模型
    allowed_tools: Dict[str, Tool] = {}  # 允许使用的工具字典，键为工具名称
    tools: List[Tool] = []  # 工具列表
    prompt: Optional[str] = None  # 可选的系统提示
    min_confidence: int = 3  # 最小置信度阈值

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Optional[List[Tool]] = None,
        output_parser: Optional[OpenAIFunctionOutputParser] = None,
        prompt: str = None,
        min_confidence: int = 3,
        **kwargs: Any,
    ) -> OpenAIFunctionsAgent:
        """
        从语言模型和工具列表创建Agent的类方法
        
        Args:
            llm: 语言模型
            tools: 可选的工具列表
            output_parser: 可选的输出解析器
            prompt: 可选的系统提示
            min_confidence: 最小置信度阈值
            **kwargs: 其他参数
            
        Returns:
            OpenAIFunctionsAgent: 创建的Agent实例
        """
        tools = tools or []  # 如果未提供工具，则使用空列表

        # 创建工具字典，键为工具名称
        allowed_tools = {tool.name: tool for tool in tools}
        # 如果未提供输出解析器，则创建默认的
        _output_parser = output_parser or OpenAIFunctionOutputParser()
        return cls(
            llm=llm,
            allowed_tools=allowed_tools,
            output_parser=_output_parser,
            tools=tools,
            prompt=prompt,
            min_confidence=min_confidence,
            **kwargs,
        )

    def plan(
        self,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        retries: int = 2,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        规划下一步行动
        
        Args:
            history: 聊天消息历史
            intermediate_steps: 中间步骤列表
            retries: 重试次数
            **kwargs: 其他参数
            
        Returns:
            Union[AgentAction, AgentFinish]: Agent的行动或完成状态
        """
        while retries > 0:
            print_with_color("Planning", Fore.LIGHTYELLOW_EX)  # 打印规划状态

            # 准备消息列表
            final_messages = []
            if self.prompt:  # 如果有系统提示，添加到消息列表
                final_messages.append(SystemMessage(content=self.prompt))
            final_messages += history.messages  # 添加历史消息

            # 记录规划输入
            logger.info(f"\nPlanning Input: {[m.content for m in final_messages]} \n")
            # 使用语言模型生成输出
            full_output: Generation = self.llm.generate(
                final_messages, self.tools
            ).generations[0]

            # 解析Agent输出
            agent_output: Union[AgentAction, AgentFinish] = self.output_parser.parse(
                full_output.message
            )
            # 打印规划输出
            print(
                f"Planning output: \nmessage content: {repr(full_output.message.content)}; "
                f"function_call: "
                f"{repr(full_output.message.function_call)}",
                Fore.YELLOW,
            )
            if isinstance(agent_output, AgentAction):
                print_with_color(
                    f"Plan to take action '{agent_output.tool}'", Fore.LIGHTYELLOW_EX
                )

            # 检查生成结果的置信度
            generation_is_confident = self.is_generation_confident(
                history=history,
                agent_output=agent_output,
                min_confidence=self.min_confidence,
            )
            if not generation_is_confident:
                # 如果置信度不够，减少重试次数并继续
                retries -= 1
                print_with_color(
                    f"Generation is not confident, {retries} retries left",
                    Fore.LIGHTYELLOW_EX,
                )
                continue
            else:
                # 置信度足够，返回输出
                return agent_output

    def is_generation_confident(
        self,
        history: ChatMessageHistory,
        agent_output: Union[AgentAction, AgentFinish],
        min_confidence: int = 3,
    ) -> bool:
        """
        估计生成结果的置信度
        
        Args:
            history: 对话历史
            agent_output: Agent的输出
            min_confidence: 被认为有足够置信度的最小分数
            
        Returns:
            bool: 是否有足够的置信度
        """

        def _format_assistant_message(action_output: Union[AgentAction, AgentFinish]):
            """
            格式化Assistant消息
            
            Args:
                action_output: Agent的行动或完成状态
                
            Returns:
                str: 格式化的消息
            """
            if isinstance(action_output, AgentFinish):
                # 如果是完成状态，格式化为Assistant消息
                assistant_message = f"Assistant: {action_output.message}"
            elif isinstance(action_output, AgentAction):
                # 如果是行动，格式化为Action消息
                assistant_message = f"Action: {action_output.tool} with input: {action_output.tool_input}"
            else:
                # 不支持的类型
                raise ValueError("Unsupported action for estimating confidence score")

            return assistant_message

        # 使用模板创建置信度估计提示
        prompt = Template(ESTIMATE_CONFIDENCE_PROMPT).substitute(
            policy=self.prompt,
            conversation_history=history.format_message(),
            assistant_message=_format_assistant_message(agent_output),
        )
        # 记录置信度估计提示
        logger.info(f"\nEstimate confidence prompt: {prompt} \n")

        # 创建用户消息
        message = UserMessage(content=prompt)

        # 使用语言模型生成输出
        full_output: Generation = self.llm.generate([message], self.tools).generations[
            0
        ]

        # 解析估计的置信度
        estimated_confidence = self.output_parser.parse_estimated_confidence(
            full_output.message
        )

        # 返回置信度是否达到最小阈值
        return estimated_confidence >= min_confidence
