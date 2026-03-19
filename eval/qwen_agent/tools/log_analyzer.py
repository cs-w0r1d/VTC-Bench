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
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
from pathlib import Path

from qwen_agent.utils.utils import logger


class LogAnalyzer:
    """Analyzes conversation logs to extract tool usage patterns."""

    def __init__(self, log_dir: str, max_recent_logs: int = 10, exclude_files: Set[str] = None):
        """Initialize the log analyzer.

        Args:
            log_dir: Directory containing log files
            max_recent_logs: Number of recent log files to analyze
            exclude_files: Set of filenames to exclude from analysis (previously analyzed)
        """
        self.log_dir = log_dir
        self.max_recent_logs = max_recent_logs
        self.exclude_files = exclude_files or set()
        self.tool_calls: List[List[str]] = []
        self.tool_combinations: Dict[str, int] = {}
        self.analyzed_files: List[str] = []  # Track which files were actually analyzed

    def _extract_tool_calls_from_response(self, response_list: list) -> List[str]:
        """Extract tool names from a response list."""
        tools = []
        for message in response_list:
            if isinstance(message, dict):
                if 'function_call' in message and isinstance(message['function_call'], dict):
                    tool_name = message['function_call'].get('name')
                    if tool_name:
                        tools.append(tool_name)
        return tools

    def _get_tool_combinations(self, tools: List[str], max_length: int = 3) -> List[Tuple[str, ...]]:
        """Generate all possible combinations of tools from a list."""
        combinations = []
        tools_unique = list(set(tools))

        for length in range(1, min(len(tools_unique) + 1, max_length + 1)):
            for i in range(len(tools_unique) - length + 1):
                combination = tuple(sorted(tools_unique[i:i + length]))
                combinations.append(combination)

        return combinations

    def load_logs(self) -> bool:
        """Load and parse log files from the specified directory.

        Loads the most recent log files, excluding any that were previously analyzed.
        This implements a sliding window: once a file is analyzed, it's excluded next time,
        allowing analysis of newly completed conversations in subsequent rounds.
        """
        if not os.path.isdir(self.log_dir):
            logger.error(f"Log directory does not exist: {self.log_dir}")
            return False

        json_files = []
        for file in os.listdir(self.log_dir):
            if file.endswith('.json') and file not in self.exclude_files:
                file_path = os.path.join(self.log_dir, file)
                mtime = os.path.getmtime(file_path)
                json_files.append((file_path, file, mtime))

        if not json_files:
            logger.warning(f"No new JSON log files found in: {self.log_dir}")
            return False

        # Sort by modification time (most recent first)
        json_files.sort(key=lambda x: x[2], reverse=True)
        json_files = json_files[:self.max_recent_logs]

        logger.info(f"Loading {len(json_files)} most recent log files (excluding {len(self.exclude_files)} previously analyzed)")

        for file_path, filename, _ in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if 'response_list' in data and isinstance(data['response_list'], list):
                        for response_item in data['response_list']:
                            if isinstance(response_item, list):
                                tools = self._extract_tool_calls_from_response(response_item)
                                if tools:
                                    self.tool_calls.append(tools)
                        self.analyzed_files.append(filename)
            except Exception as e:
                logger.error(f"Error parsing log file {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(self.tool_calls)} conversations from {len(self.analyzed_files)} log files")
        return len(self.tool_calls) > 0

    def analyze_tool_usage(self) -> Dict:
        """Analyze tool usage patterns from loaded logs."""
        if not self.tool_calls:
            logger.warning("No tool calls loaded. Please call load_logs() first.")
            return {}

        all_tools = []
        combinations_counter = defaultdict(int)

        for tools in self.tool_calls:
            all_tools.extend(tools)
            tool_combos = self._get_tool_combinations(tools, max_length=3)
            for combo in tool_combos:
                combo_key = '|'.join(combo)
                combinations_counter[combo_key] += 1

        tool_frequency = Counter(all_tools)

        if combinations_counter:
            most_common_combo = max(combinations_counter.items(), key=lambda x: x[1])
            most_common_tools = most_common_combo[0].split('|')
        else:
            most_common_tools = []

        analysis_result = {
            'all_tools': set(all_tools),
            'tool_frequency': tool_frequency,
            'tool_combinations': dict(sorted(
                combinations_counter.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            'most_common_combination': most_common_tools,
            'most_common_frequency': most_common_combo[1] if combinations_counter else 0,
        }

        return analysis_result

    def get_analysis_summary(self) -> str:
        """Get a human-readable summary of the analysis."""
        analysis = self.analyze_tool_usage()

        if not analysis:
            return "No analysis available. Please load logs first."

        summary = []
        summary.append("=" * 60)
        summary.append("TOOL USAGE ANALYSIS SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Total conversations analyzed: {len(self.tool_calls)}")
        summary.append(f"Unique tools found: {len(analysis['all_tools'])}")
        summary.append(f"Tools: {', '.join(sorted(analysis['all_tools']))}")
        summary.append("")

        summary.append("TOP 10 MOST COMMON TOOL COMBINATIONS:")
        for i, (combo, count) in enumerate(list(analysis['tool_combinations'].items())[:10], 1):
            tools = combo.split('|')
            summary.append(f"  {i}. {' + '.join(tools)} (frequency: {count})")

        summary.append("")
        summary.append("MOST COMMON TOOL COMBINATION:")
        if analysis['most_common_combination']:
            summary.append(f"  {' + '.join(analysis['most_common_combination'])}")
            summary.append(f"  Frequency: {analysis['most_common_frequency']}")
        else:
            summary.append("  No combinations found")

        summary.append("=" * 60)

        return '\n'.join(summary)


def analyze_logs(log_dir: str, max_recent_logs: int = 10) -> Dict:
    """Quick function to analyze logs in a directory."""
    analyzer = LogAnalyzer(log_dir, max_recent_logs)
    if analyzer.load_logs():
        return analyzer.analyze_tool_usage()
    return {}
