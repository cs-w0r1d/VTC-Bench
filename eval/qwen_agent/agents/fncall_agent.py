# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import re
from typing import Dict, Iterator, List, Literal, Optional, Union

from qwen_agent import Agent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, FUNCTION, ContentItem, Message
from qwen_agent.log import logger
from qwen_agent.memory import Memory
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import encode_image_as_base64, extract_files_from_messages, is_image


class FnCallAgent(Agent):
    """This is a widely applicable function call agent integrated with llm and tool use ability."""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 **kwargs):
        """Initialization the agent.

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter().
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
            files: A file url list. The initialized files for the agent.
        """
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description)

        if not hasattr(self, 'mem'):
            # Default to use Memory to manage files
            if 'qwq' in self.llm.model.lower() or 'qvq' in self.llm.model.lower() or 'qwen3' in self.llm.model.lower():
                if 'dashscope' in self.llm.model_type:
                    mem_llm = {
                        'model': 'qwen-turbo',
                        'model_type': 'qwen_dashscope',
                        'generate_cfg': {
                            'max_input_tokens': 30000
                        }
                    }
                else:
                    mem_llm = None
            else:
                mem_llm = self.llm
            self.mem = Memory(llm=mem_llm, files=files, **kwargs)

    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', **kwargs) -> Iterator[List[Message]]:
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response = []
        num_retry = 5
        consecutive_no_tool_count = 0  # Track consecutive rounds without tool usage
        round_count = 0  # Debug: Count the rounds
        while True and num_llm_calls_available > 0 and round_count < MAX_LLM_CALL_PER_RUN + 10:
            round_count += 1
            # print("ROUND_COUNT:", round_count)
            # print("num_llm_calls_available:",num_llm_calls_available)
            num_llm_calls_available -= 1

            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
            
            # 如果是 num_llm_calls_available == 0 的时候，不允许模型使用tool了 ，即 call模型的时候不传输tool
            if num_llm_calls_available == 0:
                output_stream = self._call_llm(messages=messages,
                                            functions=[],
                                            extra_generate_cfg=extra_generate_cfg)
            else:
                output_stream = self._call_llm(messages=messages,
                                            functions=[func.function for func in self.function_map.values()],
                                            extra_generate_cfg=extra_generate_cfg)
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output

            # 检查是否需要重试：当最后一轮对话时，如果输出中不包含答案
            should_retry = False
            if num_llm_calls_available == 0 and num_retry > 0:
                if len(output) == 0:
                    should_retry = True
                elif len(output) > 0:
                    content = output[0].content
                    if isinstance(content, str):
                        should_retry = "<answer>" not in content 
                    elif isinstance(content, list) and len(content) > 0:
                        # 如果是 ContentItem 列表，检查其中的文本
                        for item in content:
                            if item.text and "<answer>" not in item.text:
                                should_retry = True
                                break


            if should_retry:
                num_llm_calls_available = 1
                num_retry -= 1
                # 输出为空，需要重试
                if 'o3' in self.llm.model.lower() or 'o4'in self.llm.model.lower() or '5.2' in self.llm.model.lower():
                    format_prompt = Message(
                            role='user',
                            content="Please answer the questions directly based on the results of the previous execution. Please provide your final answer in the required format:\n<answer>Your final answer here</answer>."
                        )
                else:
                    format_prompt = Message(
                            role='user',
                            content="Please answer the questions directly based on the results of the previous processing. Please provide your final answer in the required format:\n<answer>Your final answer here</answer>."
                        )
                messages.append(format_prompt)
                response.append(format_prompt)
                yield response
                continue

            if output:
                response.extend(output)
                messages.extend(output)
                used_any_tool = False
                for out in output:
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool:
                        tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                        # 处理 tool_result 中的图片路径，将其转换为 base64
                        tool_result = self._process_tool_result_with_images(tool_result)

                        # 添加提示信息到内容中（不破坏结构）
                        prompt_text = ""
                        if num_llm_calls_available <= 5 and num_llm_calls_available > 1:
                            prompt_text = f"\n[Note: You have {num_llm_calls_available} more conversation opportunities remaining.]"

                        if num_llm_calls_available == 1:
                            # 再重复一遍题目
                            pattern2 = r'<image>\n\n(.*?)\n\n### User Image Path'
                            match = re.search(pattern2, messages[1]['content'][1]['text'], re.DOTALL)
                            question = ""
                            if match:
                                question = match.group(1)
                            if 'o3' in self.llm.model.lower() or 'o4'in self.llm.model.lower() or '5.2' in self.llm.model.lower(): 
                                prompt_text += f"\n[You have just 1 more conversation opportunity. Please answer the following questions directly based on the results of the previous execution: [{question}]], ### **Output Format (strict adherence required):**\n<think>Your detailed problem-solving approach, should go here.</think>\n<answer>Your final answer to the user's question goes here.</answer>"
                            else:
                                prompt_text += f"\n[You have just 1 more conversation opportunity. Please answer the following questions directly based on the results of the previous processing: [{question}]], ### **Output Format (strict adherence required):**\n<think>Your detailed reasoning process, should go here.</think>\n<answer>Your final answer to the user's question goes here.</answer>"

                        # 合并提示信息到结果中
                        if prompt_text:
                            if isinstance(tool_result, list):
                                tool_result.append(ContentItem(text=prompt_text))
                            elif isinstance(tool_result, str):
                                tool_result += prompt_text
                        
                        # 在这里判断是否是gemini 2.5 pro，gemini 2.5 pro不支持在function message中包含图像数据
                        # 如果是gemini 2.5 pro的话，则把图像用user的message返回给API，并指明 "这是工具生成的图片" 或者 "This is tool processed image"

                        # 检测是否为 Gemini 2.5 Pro
                        is_gemini = 'gemini' in self.llm.model.lower() or '4o' in self.llm.model.lower()

                        if is_gemini:
                            # 对于 Gemini 2.5 Pro，需要特殊处理：分离文本和图片内容
                            text_content = None
                            image_content_items = []

                            if isinstance(tool_result, list):
                                # 提取文本和图片
                                for item in tool_result:
                                    if isinstance(item, ContentItem):
                                        if item.text:
                                            if text_content is None:
                                                text_content = item.text
                                            else:
                                                text_content += '\n' + item.text
                                        if item.image:
                                            image_content_items.append(item)
                            elif isinstance(tool_result, str):
                                text_content = tool_result

                            # 创建 function message，只包含文本内容
                            fn_msg_content = text_content if text_content else "Tool executed successfully."
                            fn_msg = Message(role=FUNCTION,
                                             name=tool_name,
                                             content=fn_msg_content,
                                             tool_call_id=out.extra.get('function_id', '1'),
                                             extra={'function_id': out.extra.get('function_id', '1')})
                            messages.append(fn_msg)
                            response.append(fn_msg)

                            # 如果有图片，创建一个 user message 来传递图片
                            if image_content_items:
                                image_msg_content = [ContentItem(text="This is tool processed image(s):")]
                                image_msg_content.extend(image_content_items)
                                image_msg = Message(role='user',
                                                   content=image_msg_content)
                                messages.append(image_msg)
                                response.append(image_msg)
                        else:
                            # 其他模型，正常处理
                            fn_msg = Message(role=FUNCTION,
                                             name=tool_name,
                                             content=tool_result,
                                             tool_call_id=out.extra.get('function_id', '1'),
                                             extra={'function_id': out.extra.get('function_id', '1')})
                            messages.append(fn_msg)
                            response.append(fn_msg)

                        # fn_msg = Message(role=FUNCTION,
                        #                     name=tool_name,
                        #                     content=tool_result,
                        #                     tool_call_id=out.extra.get('function_id', '1'),
                        #                     extra={'function_id': out.extra.get('function_id', '1')})
                        # messages.append(fn_msg)
                        # response.append(fn_msg)
                        
                        
                        yield response
                        used_any_tool = True

                if not used_any_tool:
                    # Model didn't call any tool this round
                    # Check if the response contains a properly formatted answer
                    has_answer_tag = False
                    if output:
                        # Check if any message contains <answer> tags
                        for msg in output:
                            if isinstance(msg.content, str):
                                if '<answer>' in msg.content or '</answer>' in msg.content:
                                    has_answer_tag = True
                                    break
                            elif isinstance(msg.content, list):
                                for item in msg.content:
                                    if hasattr(item, 'text') and item.text:
                                        if '<answer>' in item.text or '</answer>' in item.text:
                                            has_answer_tag = True
                                            break

                    if has_answer_tag:
                        # Model provided a properly formatted answer, we can safely end
                        break
                    # Force one more round with a formatting prompt
                    if num_llm_calls_available == 0:
                        num_llm_calls_available += 1
                        # Add a message asking for formatted answer
                        format_prompt = Message(
                            role='user',
                            content="Please provide your final answer in the required format:\n<answer>Your final answer here</answer>"
                        )
                        messages.append(format_prompt)
                        response.append(format_prompt)
                        yield response
                    else:
                        # Add a message asking for formatted answer
                        format_prompt = Message(
                            role='user',
                            content="Please continue the analysis based on the tool"
                        )
                        messages.append(format_prompt)
                        response.append(format_prompt)
                        yield response

        yield response

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> str:
        
        if tool_name not in self.function_map:
            return f'Tool {tool_name} does not exists.'
        # Temporary plan: Check if it is necessary to transfer files to the tool
        # Todo: This should be changed to parameter passing, and the file URL should be determined by the model
        if self.function_map[tool_name].file_access:
            assert 'messages' in kwargs
            files = extract_files_from_messages(kwargs['messages'], include_images=True) + self.mem.system_files
            return super()._call_tool(tool_name, tool_args, files=files, **kwargs)
        else:
            return super()._call_tool(tool_name, tool_args, **kwargs)

    def _extract_image_paths_from_result(self, result: str) -> List[str]:
        """从tool_result中提取图片路径。
        
        支持以下格式：
        - Markdown 格式：![alt](path) 或 ![alt](url)
        - 纯路径：/path/to/image.png
        
        Args:
            result: tool_result 字符串
            
        Returns:
            图片路径列表
        """
        image_paths = []
        
        # 方式1：提取 Markdown 格式的图片路径
        # 匹配 ![...](path) 格式
        markdown_pattern = r'!\[[^\]]*\]\(([^\)]+)\)'
        markdown_matches = re.findall(markdown_pattern, result)
        image_paths.extend(markdown_matches)
        
        # 方式2：提取纯路径形式的图片路径（文件路径或 URL）
        # 匹配 /path/to/image.png 或 http(s)://... 格式的图片
        path_pattern = r'(?:(?:file://)?(?:/[^\s\)]+\.(?:png|jpg|jpeg|webp))|(?:https?://[^\s\)]+\.(?:png|jpg|jpeg|webp)))'
        path_matches = re.findall(path_pattern, result, re.IGNORECASE)
        
        for match in path_matches:
            if match not in image_paths:
                image_paths.append(match)
        
        # 过滤出真正存在的图片文件
        valid_paths = []
        for path in image_paths:
            if is_image(path):
                valid_paths.append(path)
        
        return valid_paths

    def _process_tool_result_with_images(self, tool_result: Union[str, List[ContentItem]]) -> Union[str, List[ContentItem]]:
        """处理 tool_result，提取图片路径，转换为 base64，并重新构造内容。
        
        Args:
            tool_result: tool 返回的结果，可能是字符串或 ContentItem 列表
            
        Returns:
            处理后的结果，如果包含图片则为 ContentItem 列表，否则保持原样
        """
        from qwen_agent.utils.utils import encode_image_as_base64
        from qwen_agent.log import logger

        # 如果已经是 ContentItem 列表，直接返回
        if isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result
        
        # 处理字符串结果
        if not isinstance(tool_result, str):
            return tool_result
        
        # 提取图片路径
        # 
        image_paths = self._extract_image_paths_from_result(tool_result)
        
        if not image_paths:
            # 没有找到图片，返回原始结果
            return tool_result
        
        # 构造包含图片的 ContentItem 列表
        content_items = []
        
        # 先添加文本内容
        if tool_result.strip():
            # 移除 Markdown 图片引用，只保留纯文本
            text_content = re.sub(r'!\[[^\]]*\]\([^\)]+\)', '', tool_result)
            text_content = text_content.strip()
            if text_content:
                content_items.append(ContentItem(text=text_content))
        
        # 添加图片内容
        for image_path in image_paths:
            try:
                # 如果是 HTTP URL 或 base64，直接使用
                if image_path.startswith(('http://', 'https://', 'data:')):
                    content_items.append(ContentItem(image=image_path))
                    continue

                # 处理本地文件路径
                # 如果是相对路径，转换为绝对路径
                if not os.path.isabs(image_path):
                    image_path = os.path.abspath(image_path)

                # 检查文件是否存在
                if not os.path.exists(image_path):
                    logger.warning(f'Image file does not exist: {image_path}')
                    continue

                # 添加图片（使用绝对路径）
                content_items.append(ContentItem(text=f'The saved image path is: {image_path}'))
                content_items.append(ContentItem(image=image_path))

            except Exception as e:
                # 如果处理失败，记录错误但继续处理
                logger.warning(f'Failed to process image {image_path}: {e}')
                # 尝试添加原始路径
                try:
                    content_items.append(ContentItem(image=image_path))
                except Exception:
                    pass  # 如果还是失败，就跳过这张图片
        
        # 如果没有生成任何 ContentItem，返回原始结果
        if not content_items:
            return tool_result
        
        # 如果只有文本，返回字符串
        if len(content_items) == 1 and content_items[0].text:
            return tool_result
        
        # 添加辅助信息帮助模型继续处理
        # content_items.append(ContentItem(text="图片已处理并附加到消息中，您可以直接看到处理后的结果。请继续分析或提供最终答案。"))
        
        return content_items
