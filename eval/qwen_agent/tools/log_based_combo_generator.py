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

import json
import os
from typing import Dict, List, Optional, Union, Set

from qwen_agent.tools.base import BaseTool, register_tool, TOOL_REGISTRY
from qwen_agent.tools.log_analyzer import LogAnalyzer
from qwen_agent.tools.dynamic_combo_tool import ComboToolRegistry
from qwen_agent.utils.utils import logger


@register_tool('log_based_combo_generator')
class LogBasedComboGenerator(BaseTool):
    """
    Tool that reads conversation logs and provides information about tool usage patterns.
    The LLM can then decide how to combine tools based on the analysis.
    """

    description = """Read and analyze conversation logs to understand tool usage patterns and generate optimized combo tools.

This tool has three main actions:

1. ANALYZE action:
   - Reads log files from a pre-configured directory (analyzes recent conversations)
   - Extracts all tool calls from the conversations
   - Analyzes tool usage frequencies and combinations
   - Returns detailed tool sequences that you can analyze to identify patterns
   - Use this to understand which tools are frequently used together

2. REGISTER_COMBO action (Create New Tools):
   - Register new combo tools that you design based on the analyzed patterns
   - You can provide:
     * combo_name: A unique, descriptive name for the new tool
     * combo_tools: List of tool names to combine (at least 2 tools required)
     * description: Clear explanation of what the combo tool does
     * implementation_code: (OPTIONAL but RECOMMENDED) Your custom Python code that implements the tool

   **IMPORTANT**: You can write custom Python implementation code!
   - The code receives: params (dict), cfg (dict), TOOL_REGISTRY (dict), logger (object)
   - Your code must set the 'result' variable with the final output
   - Wrap your code in try/except for error handling
   - This allows you to implement optimizations like passing outputs from one tool to another

3. LIST action:
   - List all currently registered combo tools with their descriptions

IMPORTANT NOTES:
- The log directory is automatically configured (no need to specify it)
- The analyze action automatically uses recent conversations (90% of worker count)
- Generated combo tools are immediately available for use in subsequent tasks
- Use custom implementation_code to create sophisticated tool combinations, not just simple chaining"""

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["analyze", "register_combo", "list"],
                "description": "Action to perform: 'analyze' (analyze logs), 'register_combo' (register a user-defined combo tool), 'list' (list registered combo tools)"
            },
            "max_recent_logs": {
                "type": "integer",
                "description": "Number of most recent log files to analyze (default: 10)",
                "default": 10
            },
            "combo_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tool names to combine (required for register_combo action)"
            },
            "combo_name": {
                "type": "string",
                "description": "Name for the combo tool"
            },
            "implementation_code": {
                "type": "string",
                "description": "OPTIONAL but RECOMMENDED - Custom Python code that implements the combo tool. The code receives: params (dict), cfg (dict), TOOL_REGISTRY (dict), logger (object). Must set 'result' variable with the final output. Example: call tool1 to get output, pass it as input to tool2, combine results. If not provided, tools will be chained automatically."
            },
            "description": {
                "type": "string",
                "description": "Description of what the combo tool does"
            }
        },
        "required": ["action"]
    }

    # Class variables
    _default_log_dir = None
    _default_max_recent_logs = 10  # Default number of recent logs to analyze
    _analyzed_files: Set[str] = set()  # Track analyzed files to avoid duplicate analysis
    _analyzed_files_cache = None  # Path to cache file for persistence

    @classmethod
    def set_default_log_dir(cls, log_dir: str):
        """Set the default log directory for this tool."""
        cls._default_log_dir = log_dir
        # Set up cache file for tracking analyzed files
        cls._analyzed_files_cache = os.path.join(log_dir, '.analyzed_files.json')
        # Load previously analyzed files
        cls._load_analyzed_files()

    @classmethod
    def set_default_max_recent_logs(cls, max_recent_logs: int):
        """Set the default number of most recent logs to analyze."""
        cls._default_max_recent_logs = max(1, max_recent_logs)

    @classmethod
    def _load_analyzed_files(cls):
        """Load the set of previously analyzed files from cache."""
        if cls._analyzed_files_cache and os.path.exists(cls._analyzed_files_cache):
            try:
                with open(cls._analyzed_files_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cls._analyzed_files = set(data.get('analyzed_files', []))
                    logger.info(f"Loaded {len(cls._analyzed_files)} previously analyzed files")
            except Exception as e:
                logger.warning(f"Could not load analyzed files cache: {e}")
                cls._analyzed_files = set()
        else:
            cls._analyzed_files = set()

    @classmethod
    def _save_analyzed_files(cls):
        """Save the set of analyzed files to cache for persistence."""
        if cls._analyzed_files_cache:
            try:
                os.makedirs(os.path.dirname(cls._analyzed_files_cache), exist_ok=True)
                with open(cls._analyzed_files_cache, 'w', encoding='utf-8') as f:
                    json.dump({'analyzed_files': list(cls._analyzed_files)}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Could not save analyzed files cache: {e}")

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, dict]:
        """
        Execute the log-based combo generator.

        Args:
            params: Parameters containing action and optional combo configuration
            kwargs: Additional keyword arguments

        Returns:
            Analysis results or registration status
        """
        try:
            if isinstance(params, str):
                params = json.loads(params)

            action = params.get('action', 'analyze')

            # Use the default log directory if not provided
            log_dir = params.get('log_dir') or self._default_log_dir
            if not log_dir:
                return {
                    'status': 'error',
                    'message': 'Log directory is not configured. Please set it via set_default_log_dir().'
                }

            # Perform the requested action
            if action == 'analyze':
                # Use the default max_recent_logs if not provided
                max_recent_logs = params.get('max_recent_logs', self._default_max_recent_logs)
                return self._analyze_logs(log_dir, max_recent_logs)
            elif action == 'register_combo':
                combo_tools = params.get('combo_tools', [])
                combo_name = params.get('combo_name')
                implementation_code = params.get('implementation_code')
                description = params.get('description')
                return self._register_combo_tool(combo_tools, combo_name, implementation_code, description)
            elif action == 'list':
                return self._list_combo_tools()
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown action: {action}'
                }

        except Exception as e:
            logger.error(f"Error in log-based combo generator: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _analyze_logs(self, log_dir: str, max_recent_logs: int) -> Dict:
        """
        Analyze logs and return tool sequences for the LLM to decide on combos.

        Uses a sliding window approach: previously analyzed files are excluded,
        so each analysis session sees new tool usage patterns from recent conversations.

        Args:
            log_dir: Directory containing log files
            max_recent_logs: Number of recent logs to analyze

        Returns:
            Tool sequences for analysis
        """
        logger.info(f"Analyzing logs in {log_dir}")

        # Use exclude_files to implement sliding window: exclude already analyzed files
        analyzer = LogAnalyzer(log_dir, max_recent_logs, exclude_files=self._analyzed_files)

        if not analyzer.load_logs():
            return {
                'status': 'error',
                'message': f'Failed to load logs from {log_dir}'
            }

        # Update the set of analyzed files to include newly analyzed ones
        self._analyzed_files.update(analyzer.analyzed_files)
        self._save_analyzed_files()

        # Return only tool sequences, let LLM do the analysis
        tool_sequences = [
            ' -> '.join(tools) for tools in analyzer.tool_calls
        ]

        logger.info(f"Analysis complete. Total files analyzed so far: {len(self._analyzed_files)}")

        return {
            'status': 'success',
            'action': 'analyze',
            'total_conversations': len(analyzer.tool_calls),
            'newly_analyzed_files': len(analyzer.analyzed_files),
            'total_analyzed_so_far': len(self._analyzed_files),
            'tool_sequences': tool_sequences,
            'instructions': 'Analyze these tool sequences from NEW conversations to identify emerging patterns and tool combinations. Design combo tools based on the most frequently used patterns you observe in these recent interactions.'
        }

    def _register_combo_tool(self, combo_tools: List[str], combo_name: Optional[str],
                            implementation_code: Optional[str] = None,
                            description: Optional[str] = None) -> Dict:
        """
        Register a combo tool with tools specified by the LLM.

        Args:
            combo_tools: List of tool names to combine
            combo_name: Name for the combo tool
            implementation_code: Python code that implements the combo functionality
            description: Description of the combo tool

        Returns:
            Registration result
        """
        if not combo_tools or len(combo_tools) < 2:
            return {
                'status': 'error',
                'message': 'At least 2 tools are required for a combo'
            }

        if not combo_name:
            combo_name = f"combo_{'+'.join(combo_tools)}"

        # Validate that tools exist in registry
        invalid_tools = [name for name in combo_tools if name not in TOOL_REGISTRY]
        if invalid_tools:
            return {
                'status': 'error',
                'message': f'Some tools not found in registry: {invalid_tools}'
            }

        # If no implementation code provided, generate a basic one
        if not implementation_code:
            implementation_code = self._generate_default_implementation(combo_tools)

        # If no description provided, generate one
        if not description:
            description = f"Combo tool that combines: {', '.join(combo_tools)}"

        try:
            registered_name = ComboToolRegistry.register_combo_tool(
                combo_name,
                combo_tools,
                implementation_code,
                description,
                allow_overwrite=True
            )

            return {
                'status': 'success',
                'action': 'register_combo',
                'generated_combo_tool': registered_name,
                'combined_tools': combo_tools,
                'implementation_code': implementation_code,
                'message': f"Successfully registered combo tool: {registered_name}"
            }

        except Exception as e:
            logger.error(f"Error registering combo tool: {e}")
            return {
                'status': 'error',
                'message': f'Failed to register combo tool: {str(e)}'
            }

    def _generate_default_implementation(self, tool_names: List[str]) -> str:
        """
        Generate a default Python implementation that chains the tools together.

        Args:
            tool_names: List of tool names to chain

        Returns:
            Python code string that implements the chaining logic
        """
        code_lines = [
            "# Default implementation: chain tools together",
            "result = None",
            "try:",
        ]

        for i, tool_name in enumerate(tool_names):
            code_lines.append(f"    # Step {i+1}: Call {tool_name}")
            code_lines.append(f"    tool_class_{i} = TOOL_REGISTRY.get('{tool_name}')")
            code_lines.append(f"    if tool_class_{i}:")
            code_lines.append(f"        tool_instance_{i} = tool_class_{i}(cfg)")
            code_lines.append(f"        tool_result_{i} = tool_instance_{i}.call(params)")
            code_lines.append(f"        result = tool_result_{i}")
            code_lines.append(f"    else:")
            code_lines.append(f"        result = {{'error': 'Tool {tool_name} not found'}}")
            code_lines.append(f"        raise Exception(result)")
            code_lines.append("")

        code_lines.append("except Exception as e:")
        code_lines.append("    result = {'status': 'error', 'message': str(e)}")

        return "\n".join(code_lines)

    def _list_combo_tools(self) -> Dict:
        """
        List all registered combo tools.

        Returns:
            Dictionary containing list of combo tools
        """
        combo_tools = ComboToolRegistry.list_combo_tools()
        combo_info = []

        for tool_name in combo_tools:
            info = ComboToolRegistry.get_combo_tool_info(tool_name)
            if info:
                combo_info.append(info)

        return {
            'status': 'success',
            'action': 'list',
            'total_combo_tools': len(combo_tools),
            'combo_tools': combo_info
        }
