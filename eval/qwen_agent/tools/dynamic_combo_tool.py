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
import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from qwen_agent.tools.base import BaseTool, register_tool, TOOL_REGISTRY
from qwen_agent.utils.utils import logger


class DynamicComboToolFactory:
    """Factory for creating dynamically generated combo tools with Python implementation code."""

    @staticmethod
    def create_combo_tool(combo_name: str,
                         tool_names: List[str],
                         implementation_code: str,
                         description: str) -> type:
        """
        Create a dynamic combo tool class with a Python implementation.

        Args:
            combo_name: Name for the new combo tool
            tool_names: List of tool names being combined
            implementation_code: Python code that implements the tool functionality
            description: Human-readable description of what the tool does

        Returns:
            A new tool class that can be instantiated and used
        """

        # Validate that all tools exist
        invalid_tools = [name for name in tool_names if name not in TOOL_REGISTRY]
        if invalid_tools:
            logger.warning(f"Some tools not found in registry: {invalid_tools}")

        # Create a namespace for the implementation code
        tool_registry_ref = TOOL_REGISTRY

        # Create parameters for the tool (extracted from implementation or set generic)
        parameters = {
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": f"Parameters for {combo_name}. See implementation for details."
                }
            },
            "required": ["params"]
        }

        # Create the combo tool class dynamically
        # Use local variables to avoid closure issues
        _combo_name = combo_name
        _description = description
        _parameters = parameters
        _impl_code = implementation_code
        _tool_list = tool_names

        class DynamicComboTool(BaseTool):
            name = _combo_name
            description = _description
            parameters = _parameters
            _implementation_code = _impl_code
            _tool_names = _tool_list

            def __init__(self, cfg: Optional[Dict] = None):
                super().__init__(cfg)
                self.tool_names = _tool_list
                self.combo_name = _combo_name
                self._execution_namespace = {
                    'TOOL_REGISTRY': tool_registry_ref,
                    'logger': logger,
                    'json': json
                }

            def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict]:
                """
                Execute the combo tool by running the implementation code.

                The implementation code is Python code that:
                1. Takes 'params' as input (the parameters passed in)
                2. Uses tools from TOOL_REGISTRY to execute operations
                3. Returns the final result
                """
                try:
                    if isinstance(params, str):
                        params = json.loads(params)

                    # Extract the actual parameters
                    tool_params = params.get('params', params)

                    # Create execution environment
                    exec_env = self._execution_namespace.copy()
                    exec_env['params'] = tool_params
                    exec_env['cfg'] = self.cfg

                    # Execute the implementation code
                    exec(self._implementation_code, exec_env)

                    # Get the result from the implementation
                    result = exec_env.get('result', 'Execution completed')

                    return {
                        'combo_name': self.combo_name,
                        'status': 'success',
                        'result': result
                    }

                except Exception as e:
                    logger.error(f"Error executing combo tool {self.combo_name}: {e}")
                    import traceback
                    return {
                        'combo_name': self.combo_name,
                        'status': 'error',
                        'message': str(e),
                        'traceback': traceback.format_exc()
                    }

        return DynamicComboTool


class ComboToolRegistry:
    """Manages dynamic combo tool registration with code-based implementations."""

    # Class variables for persistence management
    _persistence_dir = None
    _enable_persistence = False  # Default: persistence disabled

    @classmethod
    def set_persistence_dir(cls, directory: str, enable: bool = True):
        """Set the directory for persisting combo tools.

        Args:
            directory: Directory path for storing combo tools
            enable: Whether to enable persistence (default: True if directory is set)
        """
        cls._persistence_dir = directory
        cls._enable_persistence = enable
        if directory and enable:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_persistence_enabled(cls) -> bool:
        """Check if persistence is enabled."""
        return cls._enable_persistence

    @classmethod
    def _get_tool_file_path(cls, combo_name: str) -> Optional[str]:
        """Get the file path for a combo tool."""
        if not cls._enable_persistence or not cls._persistence_dir:
            return None
        return os.path.join(cls._persistence_dir, f"{combo_name}.json")

    @classmethod
    def _save_tool_to_file(cls, combo_name: str, tool_names: List[str],
                          implementation_code: str, description: str) -> bool:
        """Save combo tool metadata and code to file."""
        file_path = cls._get_tool_file_path(combo_name)
        if not file_path:
            return False

        try:
            tool_data = {
                'combo_name': combo_name,
                'combined_tools': tool_names,
                'description': description,
                'implementation_code': implementation_code,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(tool_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved combo tool to: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save combo tool {combo_name}: {e}")
            return False

    @classmethod
    def _load_tool_from_file(cls, combo_name: str) -> Optional[Dict]:
        """Load combo tool metadata and code from file."""
        file_path = cls._get_tool_file_path(combo_name)
        if not file_path or not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load combo tool {combo_name}: {e}")
            return None

    @classmethod
    def load_persisted_tools(cls) -> int:
        """Load all persisted combo tools from storage.

        Returns:
            Number of tools loaded
        """
        if not cls._enable_persistence or not cls._persistence_dir or not os.path.exists(cls._persistence_dir):
            return 0

        loaded_count = 0
        try:
            for file_name in os.listdir(cls._persistence_dir):
                if not file_name.endswith('.json'):
                    continue

                combo_name = file_name[:-5]  # Remove .json extension
                tool_data = cls._load_tool_from_file(combo_name)

                if tool_data:
                    try:
                        combo_tool_class = DynamicComboToolFactory.create_combo_tool(
                            tool_data['combo_name'],
                            tool_data['combined_tools'],
                            tool_data['implementation_code'],
                            tool_data['description']
                        )
                        TOOL_REGISTRY[combo_name] = combo_tool_class
                        loaded_count += 1
                        logger.info(f"Loaded combo tool from file: {combo_name}")
                    except Exception as e:
                        logger.error(f"Failed to load combo tool {combo_name}: {e}")

        except Exception as e:
            logger.error(f"Error loading persisted tools: {e}")

        return loaded_count

    @classmethod
    def register_combo_tool(cls, combo_name: str,
                           tool_names: List[str],
                           implementation_code: str,
                           description: str,
                           allow_overwrite: bool = True) -> str:
        """
        Register a new combo tool with a Python implementation.

        Args:
            combo_name: Name for the combo tool
            tool_names: List of tool names being combined
            implementation_code: Python code implementing the tool (receives 'params' as input)
            description: Description of the combo tool
            allow_overwrite: Whether to overwrite existing tool with same name

        Returns:
            The name of the registered combo tool
        """

        # Check if tool already exists
        if combo_name in TOOL_REGISTRY and not allow_overwrite:
            logger.warning(f"Combo tool '{combo_name}' already exists. Use allow_overwrite=True to replace.")
            return combo_name

        # Create the combo tool class
        combo_tool_class = DynamicComboToolFactory.create_combo_tool(
            combo_name,
            tool_names,
            implementation_code,
            description
        )

        # Register it
        TOOL_REGISTRY[combo_name] = combo_tool_class
        logger.info(f"Successfully registered combo tool: {combo_name}")
        logger.info(f"  Combines: {', '.join(tool_names)}")
        logger.info(f"  Implementation: {len(implementation_code)} characters of Python code")

        # Save to file if persistence is enabled
        if cls._enable_persistence and cls._persistence_dir:
            cls._save_tool_to_file(combo_name, tool_names, implementation_code, description)
            logger.info(f"✅ Persisted combo tool: {combo_name}")
        else:
            if not cls._enable_persistence:
                logger.debug(f"💾 Combo tool '{combo_name}' not persisted (persistence disabled). Enable via set_persistence_dir(..., enable=True).")
            else:
                logger.warning(f"⚠️  Combo tool '{combo_name}' not persisted. Set persistence directory via set_persistence_dir().")

        return combo_name

    @staticmethod
    def get_combo_tool_info(combo_name: str) -> Optional[Dict]:
        """Get information about a registered combo tool."""
        if combo_name not in TOOL_REGISTRY:
            return None

        tool_class = TOOL_REGISTRY[combo_name]
        return {
            'name': combo_name,
            'description': tool_class.description if hasattr(tool_class, 'description') else '',
            'is_combo': hasattr(tool_class, '_implementation_code'),
            'combined_tools': tool_class._tool_names if hasattr(tool_class, '_tool_names') else [],
            'implementation_size': len(tool_class._implementation_code) if hasattr(tool_class, '_implementation_code') else 0
        }

    @staticmethod
    def list_combo_tools() -> List[str]:
        """List all registered combo tools."""
        combo_tools = []
        for name, tool_class in TOOL_REGISTRY.items():
            if hasattr(tool_class, '_implementation_code'):
                combo_tools.append(name)
        return combo_tools

    @staticmethod
    def get_combo_tool_code(combo_name: str) -> Optional[str]:
        """Get the implementation code of a combo tool."""
        if combo_name not in TOOL_REGISTRY:
            return None

        tool_class = TOOL_REGISTRY[combo_name]
        if hasattr(tool_class, '_implementation_code'):
            return tool_class._implementation_code
        return None
