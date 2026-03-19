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
import logging
import os
from pprint import pformat
from typing import Dict, Iterator, List, Optional

import openai

from qwen_agent.utils.utils import format_as_text_message

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
else:
    from openai import OpenAIError

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, FunctionCall, Message
from qwen_agent.log import logger


@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'gpt-4o-mini'
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = (api_base or '').strip()

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

        if openai.__version__.startswith('0.'):
            if api_base:
                openai.api_base = api_base
            if api_key:
                openai.api_key = api_key
            self._complete_create = openai.Completion.create
            self._chat_complete_create = openai.ChatCompletion.create
        else:
            api_kwargs = {}
            if api_base:
                api_kwargs['base_url'] = api_base
            if api_key:
                api_kwargs['api_key'] = api_key

            def _chat_complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                # Handle model-specific parameters
                if 'gemini' in self.model:
                    if 'gemini-2.5' in self.model:
                        kwargs['reasoning_effort'] = 'high'
                    elif 'gemini-3' in self.model:
                        kwargs['extra_body'] = kwargs.get('extra_body', {})
                        kwargs['extra_body']['google'] = {
                            'thinking_config': {
                                'thinking_level': 'high'
                            }
                        }
                if "o3" in self.model or "o4" in self.model or "5.2" in self.model:
                    kwargs['reasoning_effort'] = 'high'

                if 'messages' in kwargs:
                    for msg in kwargs['messages']:
                        if msg.get('role') == 'assistant' and msg.get('function_call'):
                            if 'tool_calls' not in msg:
                                msg['tool_calls'] = []

                            fc = msg['function_call']
                            fn_id = msg.get('tool_call_id') or (msg.get('extra') or {}).get('function_id', 'call_default')

                            # Validate and fix arguments if needed
                            import json
                            arguments_str = fc.get('arguments', '{}')
                            try:
                                # Try to parse to ensure it's valid JSON
                                if isinstance(arguments_str, str) and arguments_str:
                                    json.loads(arguments_str)
                            except json.JSONDecodeError as e:
                                # Try to fix common JSON issues
                                logger.warning(f"Invalid JSON in function arguments, attempting to fix: {e}")
                                try:
                                    import re
                                    fixed_args = re.sub(r',(\s*[}\]])', r'\1', arguments_str)  # Remove trailing commas
                                    fixed_args = re.sub(r'("(?:[^"\\]|\\.)*")\s+("(?:[^"\\]|\\.)*"\s*:)', r'\1, \2', fixed_args)  # Add missing commas
                                    
                                    # Try to fix incomplete JSON by adding missing closing braces/brackets
                                    open_braces = fixed_args.count('{')
                                    close_braces = fixed_args.count('}')
                                    open_brackets = fixed_args.count('[')
                                    close_brackets = fixed_args.count(']')
                                    
                                    if open_braces > close_braces:
                                        fixed_args += '}' * (open_braces - close_braces)
                                    if open_brackets > close_brackets:
                                        fixed_args += ']' * (open_brackets - close_brackets)
                                    
                                    json.loads(fixed_args)  # Validate
                                    arguments_str = fixed_args
                                    logger.info(f"Successfully fixed function arguments JSON")
                                except Exception as fix_err:
                                    # If we can't fix it, raise error instead of sending invalid request
                                    error_msg = f"Failed to fix function arguments JSON. Original error: {e}. Arguments: {arguments_str}"
                                    logger.error(error_msg)
                                    raise ValueError(error_msg) from e

                            tool_call = {
                                'id': fn_id,
                                'type': 'function',
                                'function': {
                                    'name': fc.get('name'),
                                    'arguments': arguments_str
                                }
                            }

                            # Add thought_signature for Gemini 3 Pro
                            # thought_signature should be in extra_content.google.thought_signature
                            if 'gemini-3' in self.model:
                                thought_sig = (msg.get('extra') or {}).get('thought_signature')
                                if thought_sig:
                                    tool_call['extra_content'] = {
                                        'google': {
                                            'thought_signature': thought_sig
                                        }
                                    }

                            msg['tool_calls'].append(tool_call)
                            del msg['function_call']

                # print(kwargs)
                client = openai.OpenAI(**api_kwargs)
                
                return client.chat.completions.create(*args, **kwargs)

            def _complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                # Handle model-specific parameters
                if 'gemini' in self.model:
                    if 'gemini-2.5' in self.model:
                        kwargs['reasoning_effort'] = 'high'
                    elif 'gemini-3' in self.model:
                        kwargs['extra_body'] = kwargs.get('extra_body', {})
                        kwargs['extra_body']['google'] = {
                            'thinking_config': {
                                'thinking_level': 'high'
                            }
                        }
                if "o3" in self.model or "o4" in self.model or "5.2" in self.model:
                    kwargs['reasoning_effort'] = 'high'

                if 'messages' in kwargs:
                    for msg in kwargs['messages']:
                        if msg.get('role') == 'assistant' and msg.get('function_call'):
                            if 'tool_calls' not in msg:
                                msg['tool_calls'] = []

                            fc = msg['function_call']
                            fn_id = msg.get('tool_call_id') or (msg.get('extra') or {}).get('function_id', 'call_default')

                            # Validate and fix arguments if needed
                            import json
                            arguments_str = fc.get('arguments', '{}')
                            try:
                                # Try to parse to ensure it's valid JSON
                                if isinstance(arguments_str, str) and arguments_str:
                                    json.loads(arguments_str)
                            except json.JSONDecodeError as e:
                                # Try to fix common JSON issues
                                logger.warning(f"Invalid JSON in function arguments, attempting to fix: {e}")
                                try:
                                    import re
                                    fixed_args = re.sub(r',(\s*[}\]])', r'\1', arguments_str)  # Remove trailing commas
                                    fixed_args = re.sub(r'("(?:[^"\\]|\\.)*")\s+("(?:[^"\\]|\\.)*"\s*:)', r'\1, \2', fixed_args)  # Add missing commas
                                    
                                    # Try to fix incomplete JSON by adding missing closing braces/brackets
                                    open_braces = fixed_args.count('{')
                                    close_braces = fixed_args.count('}')
                                    open_brackets = fixed_args.count('[')
                                    close_brackets = fixed_args.count(']')
                                    
                                    if open_braces > close_braces:
                                        fixed_args += '}' * (open_braces - close_braces)
                                    if open_brackets > close_brackets:
                                        fixed_args += ']' * (open_brackets - close_brackets)
                                    
                                    json.loads(fixed_args)  # Validate
                                    arguments_str = fixed_args
                                    logger.info(f"Successfully fixed function arguments JSON")
                                except Exception as fix_err:
                                    # If we can't fix it, raise error instead of sending invalid request
                                    error_msg = f"Failed to fix function arguments JSON. Original error: {e}. Arguments: {arguments_str}"
                                    logger.error(error_msg)
                                    raise ValueError(error_msg) from e

                            tool_call = {
                                'id': fn_id,
                                'type': 'function',
                                'function': {
                                    'name': fc.get('name'),
                                    'arguments': arguments_str
                                }
                            }

                            # Add thought_signature for Gemini 3 Pro
                            # thought_signature should be in extra_content.google.thought_signature
                            if 'gemini-3' in self.model:
                                thought_sig = (msg.get('extra') or {}).get('thought_signature')
                                if thought_sig:
                                    tool_call['extra_content'] = {
                                        'google': {
                                            'thought_signature': thought_sig
                                        }
                                    }

                            msg['tool_calls'].append(tool_call)
                            del msg['function_call']

                client = openai.OpenAI(**api_kwargs)
                # 
                return client.completions.create(*args, **kwargs)

            self._complete_create = _complete_create
            self._chat_complete_create = _chat_complete_create

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)
        logger.debug(f'LLM Input generate_cfg: \n{generate_cfg}')
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=True, **generate_cfg)
            if delta_stream:
                for chunk in response:
                    if chunk.choices:
                        if hasattr(chunk.choices[0].delta,
                                   'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            yield [
                                Message(role=ASSISTANT,
                                        content='',
                                        reasoning_content=chunk.choices[0].delta.reasoning_content)
                            ]
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            yield [Message(role=ASSISTANT, content=chunk.choices[0].delta.content)]
            else:
                full_response = ''
                full_reasoning_content = ''
                full_tool_calls = []
                finish_reason = None
                
                for chunk in response:
                    if chunk.choices:
                        # print(chunk.choices[0])
                        # Check for malformed function call
                        if hasattr(chunk.choices[0], 'finish_reason') and chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason
                            if finish_reason == 'malformed_function_call':
                                logger.warning(f'Detected malformed_function_call, will retry')
                                raise ModelServiceError(
                                    message='Malformed function call detected',
                                    code='malformed_function_call'
                                )
                        
                        if hasattr(chunk.choices[0].delta,
                                   'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            full_reasoning_content += chunk.choices[0].delta.reasoning_content
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                        if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                            # 
                            for tc in chunk.choices[0].delta.tool_calls:
                                if full_tool_calls and (not tc.id or
                                                        tc.id == full_tool_calls[-1]['extra']['function_id']):
                                    if tc.function.name:
                                        full_tool_calls[-1].function_call['name'] += tc.function.name
                                    if tc.function.arguments:
                                        full_tool_calls[-1].function_call['arguments'] += tc.function.arguments
                                    # Capture thought_signature if available (for Gemini 3 Pro)
                                    # thought_signature is in extra_content.google.thought_signature
                                    if hasattr(tc, 'extra_content') and tc.extra_content:
                                        # Handle both dict and object types
                                        if isinstance(tc.extra_content, dict):
                                            google_data = tc.extra_content.get('google', {})
                                            thought_sig = google_data.get('thought_signature') if isinstance(google_data, dict) else None
                                            if thought_sig:
                                                full_tool_calls[-1].extra['thought_signature'] = thought_sig
                                        else:
                                            google_data = getattr(tc.extra_content, 'google', None)
                                            if google_data and hasattr(google_data, 'thought_signature'):
                                                full_tool_calls[-1].extra['thought_signature'] = google_data.thought_signature
                                else:
                                    extra_info = {'function_id': tc.id}
                                    # Capture thought_signature if available (for Gemini 3 Pro)
                                    # thought_signature is in extra_content.google.thought_signature
                                    if hasattr(tc, 'extra_content') and tc.extra_content:
                                        # Handle both dict and object types
                                        if isinstance(tc.extra_content, dict):
                                            google_data = tc.extra_content.get('google', {})
                                            thought_sig = google_data.get('thought_signature') if isinstance(google_data, dict) else None
                                            if thought_sig:
                                                extra_info['thought_signature'] = thought_sig
                                        else:
                                            google_data = getattr(tc.extra_content, 'google', None)
                                            if google_data and hasattr(google_data, 'thought_signature'):
                                                extra_info['thought_signature'] = google_data.thought_signature
                                    full_tool_calls.append(
                                        Message(role=ASSISTANT,
                                                content='',
                                                function_call=FunctionCall(name=tc.function.name or '',
                                                                           arguments=tc.function.arguments or ''),
                                                extra=extra_info,
                                                tool_call_id=tc.id))

                        res = []
                        if full_reasoning_content:
                            res.append(Message(role=ASSISTANT, content='', reasoning_content=full_reasoning_content))
                        if full_response:
                            # print("BEFORE FULL_RESPONSE: ",full_response)
                            # if '</think>' in full_response:
                            #     full_response = full_response.split('</think>')[-1]
                            # print("AFTER FULL_RESPONSE: ",full_response)
                            res.append(Message(
                                role=ASSISTANT,
                                content=full_response,
                            ))
                        if full_tool_calls:
                            res += full_tool_calls
                        yield res
                print("RES:", res)
                        
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            if hasattr(response.choices[0].message, 'reasoning_content'):
                return [
                    Message(role=ASSISTANT,
                            content=response.choices[0].message.content,
                            reasoning_content=response.choices[0].message.reasoning_content)
                ]
            else:
                return [Message(role=ASSISTANT, content=response.choices[0].message.content)]
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def convert_messages_to_dicts(self, messages: List[Message]) -> List[dict]:
        
        # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
        #  At this time, in order to be compatible with lower versions of vLLM,
        #  and reasoning content is currently not useful
        messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
        messages = [msg.model_dump() for msg in messages]
        messages = self._conv_qwen_agent_messages_to_oai(messages)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'LLM Input: \n{pformat(messages, indent=2)}')
        return messages
