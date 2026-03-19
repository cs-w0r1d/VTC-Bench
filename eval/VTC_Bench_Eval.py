#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import csv
import yaml
import time
import argparse
import signal
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from PIL import Image
from multiprocessing import Pool, Manager
import logging
from itertools import cycle
from contextlib import contextmanager
import subprocess
import uuid

from qwen_agent.agents import FnCallAgent
from qwen_agent.llm.schema import ContentItem, Message

import socket

import cv2
cv2.setNumThreads(1) 

try:
    import pandas as pd
    from vlmeval.smp import load, dump
    from vlmeval.dataset.image_vqa import VTC_BenchDataset
    HAS_EVAL_SUPPORT = True
except ImportError:
    HAS_EVAL_SUPPORT = False
    print("⚠️  Warning: VLMEvalKit or pandas not found. Post-processing features will be disabled.")

# 强制设置全局 Socket 超时，防止 requests 库无限期挂起
socket.setdefaulttimeout(3000)

# ==================== Timeout Management ====================
class TaskTimeoutError(Exception):
    """Custom exception for task timeout."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TaskTimeoutError(f"Task exceeded maximum timeout of {signum} seconds")


@contextmanager
def task_timeout(seconds: int, task_id: str = ""):
    """
    Context manager for task timeout using signal (Unix/Linux only).
    Falls back to threading-based timeout on Windows.

    Args:
        seconds: Timeout duration in seconds
        task_id: Task identifier for logging
    """
    # Check if we're on Unix/Linux
    if hasattr(signal, 'SIGALRM'):
        # Unix/Linux: use signal-based timeout
        def timeout_handler_with_id(signum, frame):
            raise TaskTimeoutError(f"Task '{task_id}' exceeded timeout of {seconds}s")

        # Set the signal handler and a alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler_with_id)
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)
            # Restore old handler
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows or other: use threading-based timeout (less reliable but better than nothing)
        result = {'timeout': False}

        def timeout_callback():
            result['timeout'] = True

        timer = threading.Timer(seconds, timeout_callback)
        timer.daemon = True
        timer.start()

        try:
            yield
            if result['timeout']:
                raise TaskTimeoutError(f"Task '{task_id}' exceeded timeout of {seconds}s (detected via threading)")
        finally:
            timer.cancel()


# ==================== Global Variables ====================
# Global API key rotator for multiprocessing
_api_key_rotator = None
_api_keys_pool = []  # Store all API keys for worker initialization
# Global agent instance for each worker (thread-local storage)
_worker_agent = None
_worker_config = None
_worker_id = None  # Worker ID for tracking which worker this is


def initialize_api_key_rotator(api_keys: List[str]):
    """Initialize global API key rotator (call once before multiprocessing)."""
    global _api_key_rotator, _api_keys_pool
    if api_keys and len(api_keys) > 0:
        _api_keys_pool = api_keys  # Store for worker initialization
        _api_key_rotator = cycle(api_keys)
        return True
    return False


def get_next_api_key(worker_api_keys: Optional[List[str]] = None, worker_idx: int = 0) -> Optional[str]:
    """
    Get next API key from rotator.

    If worker_api_keys is provided, use it for rotation starting from worker_idx offset.
    Otherwise use global rotator.

    Args:
        worker_api_keys: List of API keys for this worker
        worker_idx: Index offset for starting the rotation (for worker-level distribution)
    """
    global _api_key_rotator

    if worker_api_keys and len(worker_api_keys) > 0:
        # Use worker-specific key rotation
        # Each worker maintains its own rotation offset
        worker_api_keys_rotator = cycle(worker_api_keys)
        # Advance the rotator by worker_idx to distribute keys across workers
        for _ in range(worker_idx % len(worker_api_keys)):
            next(worker_api_keys_rotator)
        return next(worker_api_keys_rotator)
    elif _api_key_rotator:
        # Fallback to global rotator
        return next(_api_key_rotator)
    return None


def init_worker(config: Dict, model_name: str, worker_id: Optional[int] = None):
    """Initialize worker process with a reusable agent."""
    global _worker_agent, _worker_config, _api_key_rotator, _worker_id

    socket.setdefaulttimeout(3600)
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    _worker_config = config
    _worker_id = worker_id if worker_id is not None else 0

    api_keys = config.get('llm', {}).get('api_keys', [])
    if api_keys and len(api_keys) > 0:
        _api_key_rotator = cycle(api_keys)
        for _ in range(_worker_id % len(api_keys)):
            next(_api_key_rotator)
        print(f"🔧 Worker {os.getpid()} (ID: {_worker_id}) initialized with API key rotation ({len(api_keys)} keys)")
        print(f"   Starting key offset: {_worker_id % len(api_keys)} (key #{(_worker_id % len(api_keys)) + 1})")
    else:
        print(f"🔧 Worker {os.getpid()} (ID: {_worker_id}) initialized without API key rotation")
        _api_key_rotator = None

    _worker_agent = init_agent(config, model_name=model_name, question_id='', use_api_key_rotation=True)
    print(f"🔧 Worker {os.getpid()} (ID: {_worker_id}) initialized with agent (Socket timeout: 1800s, Memory limit: 2GB)")


def init_worker_with_id(config, model_name, counter):
    """Initialize the worker and assign a unique worker ID."""
    global _worker_id
    _worker_id = counter.value
    counter.value += 1
    init_worker(config, model_name, worker_id=_worker_id)


def load_config(config_path: str) -> Optional[Dict]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None


def init_agent(config: Dict, model_name: str = None, question_id: str = '', use_api_key_rotation: bool = True) -> FnCallAgent:
    """Initialize FnCallAgent with code_interpreter tool from config.

    Args:
        config: Configuration dictionary
        model_name: Model name for organizing output
        question_id: Question ID for tool output organization
        use_api_key_rotation: Whether to use API key rotation (for parallel processing)
    """
    # Get LLM config
    llm_config = config.get('llm', {})
    
    model_type = llm_config.get('model_type', 'qwenvl_dashscope')
    llm_cfg = {
        'model_type': model_type,
        'model': llm_config.get('model', 'qwen-vl-max-0809'),
        'generate_cfg': llm_config.get('generate_cfg', {'max_retries': 10, 'fncall_prompt_type': 'qwen'}),
        'timeout': 3600,
        'extra_config': {
            'timeout': 1200, 
            'connect_timeout': 3600
        }
    }
    
    # Handle api_key based on model_type
    if model_type in ['qwenvl_oai', 'oai', 'qwenvl_dashscope']:
        # For qwenvl_oai and oai, use model_server (api_base) from config
        llm_cfg['model_server'] = llm_config.get('model_server')

        # Use API key rotation if enabled, otherwise use default from config
        if use_api_key_rotation:
            api_key = get_next_api_key()
            if api_key:
                llm_cfg['api_key'] = api_key
            else:
                llm_cfg['api_key'] = llm_config.get('api_key', 'EMPTY')
        else:
            llm_cfg['api_key'] = llm_config.get('api_key', 'EMPTY')
    else:
        # For qwenvl_dashscope, use DASHSCOPE_API_KEY
        llm_cfg['api_key'] = llm_config.get('api_key') or os.getenv('DASHSCOPE_API_KEY')
    
    # Get tools with model_name and question_id for organizing output
    tools_config = config.get('tools', {})
    tool_names = tools_config.get('enabled', ['code_interpreter'])
    
    # Build tool list with configuration
    tools = []
    for tool_name in tool_names:
        if tool_name == 'code_interpreter':
            # Pass model_name and question_id to code_interpreter for directory organization
            tool_cfg = {
                'name': 'code_interpreter',
                'model_name': model_name,
                'question_id': question_id
            }
            tools.append(tool_cfg)
        else:
            tools.append(tool_name)

    # Get agent config
    agent_config = config.get('agent', {})
    system_prompt = agent_config.get('system_prompt', 'You are a helpful assistant.')
    
    bot = FnCallAgent(
        llm=llm_cfg,
        name=agent_config.get('name', 'VTC_Bench Agent'),
        description=agent_config.get('description', 'Agent for answering visual questions on VTC_Bench'),
        function_list=tools,
        system_message=system_prompt,
    )
    
    return bot


def get_image_size(image_path: str) -> tuple:
    """Get image size from file path."""
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error getting image size for {image_path}: {e}")
        return (0, 0)


def make_json_serializable(obj: Any, depth: int = 0) -> Any:
    """
    Recursively convert objects to JSON serializable format.
    Handle None, Pydantic objects, dictionaries, lists, etc.
    """
    import json
    from qwen_agent.llm.schema import Message, ContentItem, FunctionCall

    if depth > 50:
        return str(obj)

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if hasattr(obj, 'model_dump'):
        try:
            return make_json_serializable(obj.model_dump(), depth + 1)
        except Exception as e:
            print(f"Warning: Failed to serialize Pydantic model {type(obj)}: {e}")
            return str(obj)

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            try:
                key_str = str(key) if not isinstance(key, str) else key
                result[key_str] = make_json_serializable(value, depth + 1)
            except Exception as e:
                print(f"Warning: Failed to process dict key '{key}': {e}")
                result[key_str] = None
        return result

    if isinstance(obj, (list, tuple)):
        result = []
        for i, item in enumerate(obj):
            try:
                result.append(make_json_serializable(item, depth + 1))
            except Exception as e:
                print(f"Warning: Failed to process list item at index {i}: {e}")
                result.append(None)
        return result

    if hasattr(obj, '__dict__'):
        try:
            return make_json_serializable(obj.__dict__, depth + 1)
        except Exception as e:
            print(f"Warning: Failed to serialize object {type(obj)}: {e}")
            return str(obj)

    return str(obj)


def save_response_list_raw(response_list: List, output_file: str) -> None:
    """
    Save the complete response_list to a JSON file.
    
    Args:
        response_list: the complete list returned by agent.run()
        output_file: the output file path
    """
    import json
    from datetime import datetime
    
    # Convert Message objects to serializable format
    serializable_response = []
    
    for round_idx, round_response in enumerate(response_list):
        round_data = []
        for msg in round_response:
            # Use the generic serialization function
            msg_dict = make_json_serializable(msg)
            round_data.append(msg_dict)
        
        serializable_response.append(round_data)

    # Save to file
    result = {
        'timestamp': datetime.now().isoformat(),
        'response_list': serializable_response
    }
    
    # Ensure the final result is also fully serialized
    result = make_json_serializable(result)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Response List saved to: {output_file}")


def load_valid_results_from_jsonl(jsonl_path: str) -> Dict:
    """
    Load valid results from JSONL file.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        Dictionary mapping row_index to result data
    """
    valid_results = {}

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        row_index = data.get('row_index')

                        # Only keep records with successful and valid answers
                        if data.get('status') == 'success':
                            agent_answer = data.get('agent_answer')
                            if not should_retry_answer(agent_answer):
                                valid_results[row_index] = data
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"⚠️  Error loading results: {e}")

    return valid_results


def load_processed_indices(results_dir: str, model_name: str) -> Tuple[set, Optional[str]]:
    """
    Load already processed row indices from existing JSONL results.
    
    When resuming, screen all answers:
    - Only keep records with valid answers
    - Mark rows with invalid answers as needing reprocessing

    Args:
        results_dir: Base output directory
        model_name: Model name (for organizing results)
    
    Returns:
        Tuple of (set of processed row indices with valid answers, path to latest JSONL file or None)
    """
    processed = set()
    invalid_answer_count = 0
    model_dir = os.path.join(results_dir, model_name)
    
    if not os.path.exists(model_dir):
        return processed, None
    
    jsonl_files = [f for f in os.listdir(model_dir) if f.startswith('results_') and f.endswith('.jsonl')]
    if not jsonl_files:
        return processed, None
    

    print("model_dir:", model_dir)
    latest_jsonl = os.path.join(model_dir, sorted(jsonl_files)[-1])
    
    try:
        with open(latest_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        row_index = data.get('row_index')

                        # Check answer quality
                        if data.get('status') == 'success':
                            agent_answer = data.get('agent_answer')
                            # If the answer is valid, add to the processed list
                            if not should_retry_answer(agent_answer):
                                processed.add(row_index)
                            else:
                                # Invalid answers are not added and will be reprocessed
                                invalid_answer_count += 1
                        else:
                            # Records with error status are also added as processed (no reprocessing of error items)
                            processed.add(row_index)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

        total_previous = len(processed) + invalid_answer_count
        print(f"✅ Found {len(processed)} items with valid answers from {os.path.basename(latest_jsonl)}")
        if invalid_answer_count > 0:
            print(f"⚠️  Found {invalid_answer_count} items with invalid answers (will be reprocessed)")
        print(f"   Total previous results: {total_previous}")
        return processed, latest_jsonl
    except Exception as e:
        print(f"⚠️  Error loading processed indices: {e}")
        return processed, None

def load_tsv_data(tsv_path: str, start_idx: int = 0, end_idx: Optional[int] = None, 
                  skip_processed: bool = False, processed_indices: Optional[set] = None) -> List[Tuple[int, Dict]]:
    """
    Load TSV data and return as list of (row_idx, row_dict) tuples.
    
    Args:
        tsv_path: Path to TSV file
        start_idx: Start index (0-based)
        end_idx: End index (exclusive)
        skip_processed: Whether to skip already processed items
        processed_indices: Set of processed row indices (required if skip_processed=True)
    """
    data = []
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row_idx, row in enumerate(reader):
                # Apply start and end indices
                if row_idx < start_idx:
                    continue
                if end_idx is not None and row_idx >= end_idx:
                    break
                
                # Skip already processed items if requested
                if skip_processed and processed_indices and row_idx in processed_indices:
                    continue
                
                data.append((row_idx, row))
        
        print(f"✅ Loaded {len(data)} items from TSV")
        return data
    except Exception as e:
        print(f"❌ Error loading TSV data: {e}")
        raise


def is_retryable_error(error_str: str, error_type: str = "", attempt: int = 0, max_retries: int = 3) -> bool:
    """
    Check if the error is retryable.
    
    Args:
        error_str: Error message string
        error_type: Exception type name (e.g., 'ReadTimeout', 'ConnectionError')
        attempt: Current retry count
        max_retries: Maximum retry count
    """
    # Intelligent handling of task timeout
    if 'TaskTimeoutError' in error_str or 'TaskTimeoutError' in error_type:
        if attempt >= max_retries - 1:
            return False
        return True

    # Add exception type check
    retryable_types = [
        'ReadTimeout',
        'ConnectTimeout', 
        'ConnectionError',
        'HTTPError',
        'Timeout',
        'RemoteDisconnected',
    ]
    
    if any(t in error_type for t in retryable_types):
        return True

    # Original string keyword check
    retryable_keywords = [
        '429', '500', '503',
        'timeout', 'Timeout',
        'connection', 'Connection',
        'temporarily unavailable',
        'rate limit', 'Rate limit',
        'TPM limit',
        'request queue',
        "'NoneType' object",
        "is not iterable",
        'None',
        'remote_failed',
        'service','time'
    ]
    return any(keyword in error_str for keyword in retryable_keywords)


def should_retry_answer(agent_answer: Optional[str]) -> bool:
    """
    Check if agent_answer needs to be retried.

    Cases that need to be retried:
    - agent_answer is empty or None
    - agent_answer contains unacceptable keywords: 'unable', 'Your final answer here', 'cannot', 'indiscernible'

    Args:
        agent_answer: The answer extracted from the model response

    Returns:
        True if needs to be retried, False if the answer is valid
    """
    # If the answer is empty or None, it needs to be retried
    if not agent_answer or (isinstance(agent_answer, str) and not agent_answer.strip()):
        return True

    # Check unacceptable keywords (case-insensitive)
    if isinstance(agent_answer, str):
        answer_lower = agent_answer.lower()
        unacceptable_keywords = [
            'unable',
            'your final answer here',
            'cannot',
            'indiscernible',
            'insufficient',
            'unreadable'
        ]

        for keyword in unacceptable_keywords:
            if keyword in answer_lower:
                return True

    # The answer is valid, no need to retry
    return False


def process_single_item(args: Tuple) -> Dict:
    """
    Process a single item for multiprocessing with retry logic.

    Args:
        args: Tuple of (config, item_data, output_dir, prompt_template, skip_errors, model_name)
                where item_data is (row_idx, row_dict)

    Returns:
        Result dictionary or error dict
    """
    import gc

    config, item_data, output_dir, prompt_template, skip_errors, model_name = args
    row_idx, row = item_data

    # Timeout and retry configuration
    MAX_RETRIES = 3
    INITIAL_WAIT = 5  # Initial wait time (seconds)
    BACKOFF_FACTOR = 2  # Exponential backoff factor
    TASK_TIMEOUT = 3600  # Maximum task timeout time (seconds) = 30 minutes
    
    try:
        # Extract data from TSV first to get item_id
        item_idx = row.get('index', str(row_idx))
        item_id = row.get('id', f"item_{row_idx}")

        global _worker_agent, _worker_config
        # Create a new agent for each task to ensure getting the next rotated API key
        agent = init_agent(_worker_config, model_name=model_name, question_id=item_id, use_api_key_rotation=True)
        category = row.get('category', '')
        image_path = row.get('image', '')
        question = row.get('question', '')
        gt_answer = row.get('answer', '')
        
        # Optional fields
        option_a = row.get('A', '')
        option_b = row.get('B', '')
        option_c = row.get('C', '')
        option_d = row.get('D', '')
        option_e = row.get('E', '')
        
        # Validate required fields
        if not image_path or not question:
            return {
                'status': 'error',
                'row_index': row_idx,
                'error': 'Missing image_path or question'
            }
        
        # Check if image exists
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'row_index': row_idx,
                'error': f'Image not found at {image_path}'
            }
        
        # Get image size
        image_size = get_image_size(image_path)

        options = []
        for opt_key in ['A', 'B', 'C', 'D', 'E']:
            opt_value = row.get(opt_key, '')
            if opt_value and opt_value.strip():  # Only add non-empty options
                options.append(f"{opt_key}. {opt_value}\n")
        
        if options:
            question += "\n\nOptions:\n" + "\n".join(options)

        
        # Format user prompt
        if "reference_trajectory" in prompt_template:
            model_tools_gt = row.get("model_tools_gt")
            user_prompt = format_user_prompt(question, image_path, image_size, model_tools_gt, prompt_template)
        else:
            user_prompt = format_user_prompt(question, image_path, image_size, None, prompt_template)
        
        # print("User_Prompt:", user_prompt)
        
        # Prepare messages for agent
        # Use ContentItem objects to ensure proper handling of multimodal content
        messages = [Message(
            role='user',
            content=[
                ContentItem(image=image_path),
                ContentItem(text=user_prompt)
            ]
        )]
        
        # Run agent with retry logic and timeout control
        response_list = None
        last_error = None
        agent_answer = None
        
        for attempt in range(MAX_RETRIES):
            try:
                with task_timeout(seconds=TASK_TIMEOUT, task_id=item_id):
                    response_list = agent.run_nonstream(messages=messages, question_id=item_id)
                
                # Validate that response_list is not None
                if response_list is None:
                    raise ValueError("Agent returned None response_list - model service may have failed")

                # Validate that response_list is iterable
                if not isinstance(response_list, (list, tuple)):
                    raise ValueError(f"Expected response_list to be list/tuple, got {type(response_list).__name__}")

                # Extract answer from response
                response_text = None
                reasoning_content = None

                if response_list:
                    # response_list is a list of Message objects, iterate backwards to find last assistant message
                    for msg in reversed(response_list):
                        # Handle Message object with content attribute
                        if hasattr(msg, 'content'):
                            content = msg.content
                            reasoning_content = getattr(msg, 'reasoning_content', None)
                            # Skip empty content or tool messages
                            if content and isinstance(content, str) and content.strip():
                                response_text = content
                                break
                            # Handle list of ContentItem
                            elif isinstance(content, list):
                                for item in reversed(content):
                                    if hasattr(item, 'text') and item.text:
                                        response_text = item.text
                                        break
                            if response_text:
                                break
                        # Handle dict response
                        elif isinstance(msg, dict) and 'content' in msg:
                            content = msg['content']
                            reasoning_content = msg.get('reasoning_content', None)
                            if content and isinstance(content, str) and content.strip():
                                response_text = content
                                break

                # Extract answer - Strategy 1: Extract from response_text
                if response_text and '<answer>' in response_text and '</answer>' in response_text:
                    start = response_text.find('<answer>') + len('<answer>')
                    end = response_text.find('</answer>')
                    agent_answer = response_text[start:end].strip()
                # Internvl <|answer|>
                if response_text and response_text.count('<|answer|>') >= 2:
                    # Find the position of the first tag and skip its length
                    start = response_text.find('<|answer|>') + len('<|answer|>')
                    # Start from the start position and find the second tag
                    end = response_text.find('<|answer|>', start)
                    # Extract the content between the two tags
                    agent_answer = response_text[start:end].strip()

                # Strategy 2: Extract from reasoning_content
                if not agent_answer and reasoning_content and '<answer>' in reasoning_content and '</answer>' in reasoning_content:
                    start = reasoning_content.find('<answer>') + len('<answer>')
                    end = reasoning_content.find('</answer>')
                    agent_answer = reasoning_content[start:end].strip()

                # Check if the answer needs to be retried
                if should_retry_answer(agent_answer):
                    if attempt < MAX_RETRIES - 1:
                        print(f"[Row {row_idx}] ⚠️  Invalid answer detected (attempt {attempt + 1}/{MAX_RETRIES}): {agent_answer if agent_answer else '(empty)'}")
                        # Raise an exception to trigger the retry mechanism
                        raise ValueError(f"Invalid agent answer: {agent_answer if agent_answer else '(empty) - answer contains invalid keywords or is empty'}")
                    else:
                        # Last attempt, even if the answer is invalid, accept it
                        print(f"[Row {row_idx}] ⚠️  Last attempt - accepting invalid answer: {agent_answer if agent_answer else '(empty)'}")

                break  # Successfully obtained a valid answer, exit the retry loop

            except Exception as e:
                last_error = str(e)
                error_type = type(e).__name__
                
                # Check if the error is retryable or the answer is invalid, pass in the current retry count and maximum retry count
                is_invalid_answer = 'Invalid agent answer' in last_error
                is_retryable = is_retryable_error(
                    last_error, 
                    error_type=error_type,  # Pass exception type
                    attempt=attempt, 
                    max_retries=MAX_RETRIES
                )
                if is_retryable or is_invalid_answer:
                    if attempt < MAX_RETRIES - 1:
                        wait_time = INITIAL_WAIT * (BACKOFF_FACTOR ** attempt)
                        # More detailed logging
                        if error_type in ['ReadTimeout', 'ConnectTimeout', 'ConnectionError']:
                            print(f"[Row {row_idx}] 🔌 Network timeout ({error_type}) - retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                        elif 'TaskTimeoutError' in error_type:
                            print(f"[Row {row_idx}] ⏳ Task timeout - retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                        elif is_invalid_answer:
                            print(f"[Row {row_idx}] ⚠️  Invalid answer - retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                        else:
                            print(f"[Row {row_idx}] ⏳ Retryable error ({error_type}) - retrying (attempt {attempt + 1}/{MAX_RETRIES})")
                        
                        print(f"[Row {row_idx}] ⏰ Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Last attempt failed
                        if 'TaskTimeoutError' in error_type:
                            print(f"[Row {row_idx}] ⏱️  Task timeout (all retries exhausted): {last_error}")
                        elif is_invalid_answer:
                            print(f"[Row {row_idx}] ⚠️  Invalid answer - all retries exhausted: {last_error}")
                        else:
                            print(f"[Row {row_idx}] ❌ Max retries reached for: {last_error}")
                        raise
                else:
                    # Non-retryable error
                    print(f"[Row {row_idx}] ❌ Non-retryable error ({error_type}): {last_error}")
                    raise
        
        # Save the complete response_list
        response_file = os.path.join(output_dir, f"response_list_{item_id}.json")
        save_response_list_raw([response_list], response_file)
        
        # Build the complete conversation record
        full_conversation = []  # Store complete conversation
        
        # Record initial system message
        full_conversation.append({
            'role': 'system',
            'content': agent.system_message if hasattr(agent, 'system_message') else 'System message not available'
        })
        
        # Record user message (without image for JSON serialization)
        user_msg_record = {
            'role': 'user',
            'content': [
                {'image': image_path},  # Keep image path reference
                {'text': user_prompt}
            ]
        }
        full_conversation.append(user_msg_record)
        
        # Record assistant message if we found response text
        if response_text:
            full_conversation.append({
                'role': 'assistant',
                'content': response_text
            })
            
        # Also record reasoning_content if present
        if reasoning_content:
            full_conversation.append({
                'role': 'assistant',
                'reasoning_content': reasoning_content
            })
        
        # Return result
        result = {
            'status': 'success',
            'row_index': row_idx,
            'item_index': item_idx,
            'item_id': item_id,
            'category': category,
            'image_path': image_path,
            'image_size': image_size,
            'question': question,
            'ground_truth': gt_answer,
            'agent_answer': agent_answer,
            'full_response': response_text,
            'options': {
                'A': option_a,
                'B': option_b,
                'C': option_c,
                'D': option_d,
                'E': option_e
            },
            'conversation': full_conversation,
            'timestamp': datetime.now().isoformat()
        }
        return result
        
    except Exception as e:
        error_msg = str(e)
        if skip_errors:
            # Skip errors, record and continue
            return {
                'status': 'error',
                'row_index': row_idx,
                'error': error_msg,
                'retried': True
            }
        else:
            # Do not skip, return error
            return {
                'status': 'error',
                'row_index': row_idx,
                'error': error_msg,
                'retried': False
            }
    finally:
        # Explicitly clean up memory and resources
        # Eliminate references to local variables to help garbage collection
        try:
            del row, messages, response_list, result
        except:
            pass
        # Force garbage collection (after processing 10 items)
        if row_idx % 10 == 0:
            gc.collect()


def format_user_prompt(question: str, image_path: str, image_size: tuple, model_tools_gt: str = None, template: str = None) -> str:
    """Format user prompt according to the specified format."""
    size_str = f"{image_size[0]}x{image_size[1]}" if image_size else "Unknown"
    
    # Use custom template if provided, otherwise use default
    if model_tools_gt:
        prompt = template.format(
            question=question,
            image_path=image_path,
            image_size=size_str,
            reference_trajectory=model_tools_gt
        )
    else:
        prompt = template.format(
            question=question,
            image_path=image_path,
            image_size=size_str
        )

    return prompt


def process_vtc_bench_data_parallel(
    tsv_path: str,
    output_dir: str,
    config: Dict,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    skip_errors: bool = True,
    prompt_template: str = None,
    num_workers: int = 10,
    model_name: str = None,
    resume: bool = True
) -> Optional[str]:
    """
    Process VTC_Bench.tsv data in parallel using multiprocessing with resume capability.
    
    Args:
        tsv_path: Path to VTC_Bench.tsv file
        output_dir: Output directory for results (base directory)
        config: Configuration dictionary
        start_idx: Start index (0-based)
        end_idx: End index (exclusive), None means process all
        skip_errors: Whether to skip errors and continue
        prompt_template: Custom prompt template
        num_workers: Number of parallel workers (default: 10)
        model_name: Model name for organizing results
        resume: Whether to resume from previous run (default: True)
    """
    
    # If model_name is provided, create model-specific output directory
    if model_name:
        output_dir = os.path.join(output_dir, model_name)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load previously processed indices if resume is enabled
    processed_indices = set()
    latest_results_file = None
    if resume:
        print(f"🔍 Checking for previous results and filtering for answer quality...")
        processed_indices, latest_results_file = load_processed_indices(output_dir, "")
        
        if processed_indices:
            print(f"📋 Resume mode enabled: Will skip {len(processed_indices)} items with valid answers")
            print(f"   Previous results: {os.path.basename(latest_results_file) if latest_results_file else 'N/A'}")
    
    # Prepare output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_jsonl = os.path.join(output_dir, f"results_{timestamp}.jsonl")
    error_log = os.path.join(output_dir, f"errors_{timestamp}.txt")
    
    # If resume mode is enabled and there are previous results, write valid results to a new jsonl file
    valid_results_count = 0
    if resume and latest_results_file:
        print(f"\n📝 Loading valid results from previous session...")
        valid_results = load_valid_results_from_jsonl(latest_results_file)

        if valid_results:
            print(f"✅ Loaded {len(valid_results)} valid results from previous session")
            print(f"📝 Writing valid results to new JSONL file...")

            try:
                with open(results_jsonl, 'w', encoding='utf-8') as f:
                    for row_index, result_data in sorted(valid_results.items()):
                        f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
                valid_results_count = len(valid_results)
                print(f"✅ Successfully wrote {valid_results_count} valid results to new file")
            except Exception as e:
                print(f"⚠️  Error writing valid results: {e}")

    try:
        # Load TSV data (with option to skip already processed items)
        print(f"📂 Loading TSV data from {tsv_path}...")
        data = load_tsv_data(tsv_path, start_idx, end_idx, skip_processed=resume, processed_indices=processed_indices)
        total_items = len(data)
        
        if total_items == 0:
            print("⚠️  No items to process!")
            return
        
        print(f"\n🔄 Starting parallel processing with {num_workers} workers...")
        print(f"📊 Total items to process: {total_items}\n")
        
        args_list = [
            (config, item_data, output_dir, prompt_template, skip_errors, model_name)
            for item_data in data
        ]
        
        # Process in parallel
        processed_count = 0
        error_count = 0
        
        # Open result files once for the entire batch to reduce file handle churn
        # This significantly reduces the number of open file handles in parallel processing
        results_file = open(results_jsonl, 'a', encoding='utf-8', buffering=1)  # Line buffering
        error_file = open(error_log, 'a', encoding='utf-8', buffering=1)
                    
        try:
            # Use initializer to create agent once per worker
            if num_workers <= 4:
                maxtasksperchild = 50  
                chunksize = 5
            else:
                maxtasksperchild = 20  
                chunksize = 2

            print(f"🔧 Pool Configuration:")
            print(f"   - Workers: {num_workers}")
            print(f"   - Max tasks per worker: {maxtasksperchild}")
            print(f"   - Chunk size: {chunksize}")
            print(f"   - Estimated process lifecycle: ~{maxtasksperchild * 5} min (assuming 5 min/task)\n")

            with Manager() as manager:
                worker_counter = manager.Value('i', 0)

                with Pool(processes=num_workers, initializer=init_worker_with_id, initargs=(config, model_name, worker_counter), maxtasksperchild=maxtasksperchild) as pool:
                    for idx, result in enumerate(pool.imap_unordered(process_single_item, args_list, chunksize=chunksize), 1):
                        if result['status'] == 'success':
                            # Write successful result to JSONL
                            results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                            processed_count += 1
                    
                            row_idx = result['row_index']
                            item_id = result['item_id']
                            agent_answer = result['agent_answer']
        
                            print(f"[{idx}/{total_items}] ✅ Row {row_idx}: {item_id}")
                            answer_preview = agent_answer if agent_answer else "(empty)"
                            if isinstance(answer_preview, str) and len(answer_preview) > 80:
                                answer_preview = answer_preview[:80] + "..."
                            print(f"         Answer: {answer_preview}")

                        else:
                            # Error occurred
                            row_idx = result['row_index']
                            error_msg_content = result['error']
                            retried = result.get('retried', False)

                            # Write error to log
                            error_log_line = f"Row {row_idx}: {error_msg_content}"
                            error_file.write(f"{error_log_line}\n")
                            error_count += 1

                            # Display error with retry info
                            if retried:
                                print(f"[{idx}/{total_items}] ⏭️  Row {row_idx}: SKIPPED (after retries)")
                            else:
                                print(f"[{idx}/{total_items}] ❌ Row {row_idx}: FAILED")
                            error_preview = error_msg_content if error_msg_content else "(unknown error)"
                            if len(error_preview) > 100:
                                error_preview = error_preview[:100] + "..."
                            print(f"         Error: {error_preview}")
        finally:
            # Ensure files are properly closed
            results_file.close()
            error_file.close()

        # Summary
        total_processed_ever = valid_results_count + processed_count
        print(f"\n{'='*60}")
        print(f"🎉 Parallel Processing Complete!")
        print(f"{'='*60}")
        print(f"📊 Processing Statistics:")
        print(f"  ✅ This Session Successful: {processed_count}/{total_items}")
        print(f"  ❌ This Session Failed: {error_count}/{total_items}")
        if total_items > 0:
            print(f"  📈 This Session Success Rate: {(processed_count/total_items*100):.1f}%")
        if valid_results_count > 0:
            print(f"\n  📊 Overall Statistics (including previous sessions):")
            print(f"  ✅ Valid Results from Previous: {valid_results_count}")
            print(f"  ✅ New Results This Session: {processed_count}")
            print(f"  ✅ Total Valid Results: {total_processed_ever}")
        if len(processed_indices) > 0:
            print(f"  ⏭️  Previously Processed: {len(processed_indices)}")
            print(f"  📊 This Session Processed: {total_items}")
        print(f"\n📁 Output Files:")
        print(f"  📄 Results: {results_jsonl}")
        if error_count > 0:
            print(f"  📋 Errors: {error_log}")
        print(f"{'='*60}")
        print(f"\n💡 Error Handling & Timeout Strategy:")
        print(f"  ⏱️  Task Timeout: {600} seconds (10 minutes) per item")
        print(f"  🔄 Retryable errors (AUTO RETRY - up to 3 attempts with exponential backoff):")
        print(f"     - Network errors: 429, 500, 503, connection timeout")
        print(f"     - Model errors: 'NoneType' object, is not iterable")
        print(f"     - Service errors: temporarily unavailable, rate limit, remote failed")
        print(f"  ❌ Non-retryable errors: Skipped based on skip_errors setting")
        print(f"  📋 Configuration:")
        print(f"     - skip_errors={skip_errors}")
        print(f"     - Resume mode: {resume}")
        print(f"  🖥️  Timeout Mechanism:")
        print(f"     - Unix/Linux: Signal-based (SIGALRM) for reliable timeout")
        print(f"     - Windows: Threading-based (less reliable)")
        
        # Return the path to the latest JSONL file
        return results_jsonl

    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        with open(error_log, 'a', encoding='utf-8') as ef:
            ef.write(f"Fatal error: {str(e)}\n")
        raise


def extract_predictions_by_id(jsonl_path: str, field_name: str = 'agent_answer') -> Dict:
    """
    Extract predictions or agent_answer from JSONL file, using ID as key.

    Args:
        jsonl_path: Path to the prediction JSONL file
        field_name: Field name to extract ('prediction' or 'agent_answer')

    Returns:
        Dictionary mapping ID to prediction/agent_answer
    """
    prediction_dict = {}

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    item_id = data.get('item_id', data.get('id', ''))  # Support both item_id and id
                    prediction = data.get(field_name, '')
                    if prediction is None:
                        prediction = ''

                    if item_id:
                        prediction_dict[item_id] = prediction
                except json.JSONDecodeError as e:
                    continue
        print(f"✅ Extracted {len(prediction_dict)} {field_name} values from JSONL")
    except Exception as e:
        print(f"⚠️  Error extracting predictions: {e}")

    return prediction_dict


def convert_jsonl_to_xlsx(jsonl_path: str, tsv_path: str, output_xlsx_path: str) -> bool:
    """
    Convert JSONL results to Excel format using VTC_Bench.tsv as base.

    Args:
        jsonl_path: Path to the results JSONL file
        tsv_path: Path to VTC_Bench.tsv
        output_xlsx_path: Output path for Excel file

    Returns:
        True if successful, False otherwise
    """
    if not HAS_EVAL_SUPPORT:
        print("❌ pandas or VLMEvalKit not available. Skipping JSONL to Excel conversion.")
        return False

    try:
        print(f"\n📝 Converting JSONL to Excel...")
        print(f"   JSONL file: {jsonl_path}")

        # Load base data from TSV
        print(f"📂 Loading base data from TSV: {tsv_path}")
        df_base = pd.read_csv(tsv_path, sep='\t')

        # Select required columns (dynamically handle option columns)
        required_cols = ['index', 'id', 'category', 'question', 'answer']
        select_cols = required_cols.copy()

        # Add option columns that exist in the TSV
        option_cols = []
        for col in ['A', 'B', 'C', 'D', 'E']:
            if col in df_base.columns:
                select_cols.append(col)
                option_cols.append(col)

        df_base = df_base[select_cols]

        # Add prediction column if missing
        if 'prediction' not in df_base.columns:
            df_base['prediction'] = None

        # Reorder columns: required + options + prediction
        reorder_cols = required_cols + option_cols + ['prediction']
        df_base = df_base[reorder_cols]

        print(f"✅ Loaded {len(df_base)} rows from TSV")
        print(f"   Columns: {', '.join(reorder_cols)}")

        # Extract predictions from JSONL
        prediction_dict = extract_predictions_by_id(jsonl_path, field_name='agent_answer')

        # Update predictions in DataFrame
        matched_count = 0
        for idx, row in df_base.iterrows():
            item_id = row['id']
            if item_id in prediction_dict:
                df_base.at[idx, 'prediction'] = prediction_dict[item_id]
                matched_count += 1

        print(f"✅ Matched and updated {matched_count}/{len(df_base)} records")

        # Create output directory
        Path(output_xlsx_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to Excel
        dump(df_base, output_xlsx_path)
        print(f"✅ Excel file saved: {output_xlsx_path}")

        return True

    except Exception as e:
        print(f"❌ Error converting JSONL to Excel: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_eval_output_folder(model_name: str, benchmark_name: str = 'VTC_Bench',
                              config_file: str = None, use_raw_api: bool = False) -> str:
    """
    Create evaluation output folder following the naming convention.
    Format: Qwen-Agent-{CodeType}-{APIType}-{ModelType}-{Model-Name}

    Args:
        model_name: Model name (e.g., 'Qwen3-VL-32B-Instruct', 'Qwen3-VL-32B-Thinking', etc.)
        benchmark_name: Benchmark name (default: 'VTC_Bench')
        config_file: Config file path for determining code/interface type
        use_raw_api: Whether using raw API (from config)

    Returns:
        Path to the created folder
    """
    # Step 1: Determine CodeType (code or interface) from config_file name
    if config_file:
        config_name = os.path.basename(config_file).lower()
        if 'interface' in config_name:
            code_type = 'Interface'
        else:
            code_type = 'Code'
    else:
        code_type = 'Code'  # Default to Code

    # Step 2: Determine APIType (nous or rawapi)
    api_type = 'RawAPI' if use_raw_api else 'Nous'

    # Step 3: Determine ModelType (thinking or instruct) for Qwen models only
    model_type = ''
    if 'qwen' in model_name.lower() or 'Qwen' in model_name:
        if 'Thinking' in model_name or 'thinking' in model_name:
            model_type = 'Thinking'
        else:
            model_type = 'Instruct'

    # Build folder name
    if model_type:
        folder_name = f"Qwen-Agent-{code_type}-{api_type}-{model_type}-{model_name}"
    else:
        folder_name = f"Qwen-Agent-{code_type}-{api_type}-{model_name}"

    # Base path
    base_eval_path = 'eval/VLMEvalKit/outputs'
    
    eval_folder = os.path.join(base_eval_path, benchmark_name, folder_name)

    # Create folder
    Path(eval_folder).mkdir(parents=True, exist_ok=True)

    print(f"✅ Created evaluation output folder: {eval_folder}")
    return eval_folder


def run_evaluation(xlsx_path: str, eval_folder: str, method: str = 'llm',
                   model: str = 'gpt-4o-mini', api_key: Optional[str] = None,
                   api_base: Optional[str] = None, nproc: int = 8,
                   data_dir: Optional[str] = None) -> bool:
    """
    Run evaluation on the converted Excel file.

    Args:
        xlsx_path: Path to the Excel file
        eval_folder: Output folder for evaluation results
        method: Evaluation method ('heuristic' or 'llm')
        model: Model for LLM evaluation
        api_key: API key for LLM
        api_base: API base URL for LLM
        nproc: Number of processes for evaluation
        data_dir: Path to VTC_Bench data directory (optional, will set LMUData env var)

    Returns:
        True if successful, False otherwise
    """
    if not HAS_EVAL_SUPPORT:
        print("❌ VLMEvalKit not available. Skipping evaluation.")
        return False

    try:
        print(f"\n🔄 Starting evaluation...")
        print(f"   Method: {method}")
        print(f"   Excel file: {xlsx_path}")
        print(f"   Output folder: {eval_folder}")

        # Create VTC_BenchDataset instance
        dataset = VTC_BenchDataset('VTC_Bench_680')

        if method == 'heuristic':
            print(f"\n{'='*60}")
            print(f"Method: Heuristic evaluation (exact matching)")
            print(f"{'='*60}")

            result = dataset.evaluate(
                xlsx_path,
                model='exact_matching',
                nproc=nproc
            )

            print(f"\n✅ Heuristic evaluation completed")
            print(f"Result:\n{result}")

        else:  # llm
            print(f"\n{'='*60}")
            print(f"Method: LLM evaluation")
            print(f"Model: {model}")
            print(f"{'='*60}")

            # Setup judge kwargs
            judge_kwargs = {
                'model': model,
                'nproc': nproc,
            }

            if api_key:
                judge_kwargs['key'] = api_key
                judge_kwargs['api_key'] = api_key
                print(f"✅ Using provided API key")

            if api_base:
                judge_kwargs['api_base'] = api_base
                print(f"✅ Using custom API base URL: {api_base}")

            result = dataset.evaluate(xlsx_path, **judge_kwargs)

            print(f"\n✅ LLM evaluation completed")
            print(f"Result:\n{result}")

        print(f"\n✅ Evaluation completed")
        print(f"📂 Result directory: {eval_folder}")

        return True

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def post_process_results(
    jsonl_path: str,
    tsv_path: str,
    model_name: str,
    benchmark_name: str = 'VTC_Bench',
    eval_method: str = 'llm',
    eval_model: str = 'gpt-4o-mini',
    eval_api_key: Optional[str] = '',
    eval_api_base: Optional[str] = '',
    nproc: int = 8,
    data_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    use_raw_api: bool = False
) -> bool:
    """
    Post-process results: convert JSONL to Excel and run evaluation.

    Args:
        jsonl_path: Path to the JSONL results file
        tsv_path: Path to VTC_Bench TSV file
        model_name: Model name for output folder naming
        benchmark_name: Benchmark name (default: 'VTC_Bench')
        eval_method: Evaluation method ('heuristic' or 'llm')
        eval_model: Model for LLM evaluation
        eval_api_key: API key for evaluation
        eval_api_base: API base URL for evaluation
        nproc: Number of processes for evaluation
        data_dir: Path to VTC_Bench data directory (optional, will set LMUData env var)
        config_file: Config file path for determining code/interface type
        use_raw_api: Whether using raw API (from config)

    Returns:
        True if all steps successful, False otherwise
    """

    if not os.path.exists(jsonl_path):
        print(f"❌ JSONL file not found: {jsonl_path}")
        return False

    if not os.path.exists(tsv_path):
        print(f"❌ TSV file not found: {tsv_path}")
        return False

    # Create evaluation output folder with proper classification
    eval_folder = create_eval_output_folder(model_name, benchmark_name, config_file, use_raw_api)

    # Prepare xlsx file path
    xlsx_filename = f"{model_name}_{benchmark_name}.xlsx"
    xlsx_path = os.path.join(eval_folder, xlsx_filename)

    # Convert JSONL to Excel
    if not convert_jsonl_to_xlsx(jsonl_path, tsv_path, xlsx_path):
        return False

    # Run evaluation
    if not run_evaluation(
        xlsx_path,
        eval_folder,
        method=eval_method,
        model=eval_model,
        api_key=eval_api_key,
        api_base=eval_api_base,
        nproc=nproc,
        data_dir=data_dir
    ):
        return False

    print(f"\n{'='*60}")
    print(f"✅ Post-processing completed")
    print(f"{'='*60}")
    print(f"📊 Summary:")
    print(f"  📝 JSONL file: {jsonl_path}")
    print(f"  📄 Excel file: {xlsx_path}")
    print(f"  📁 Evaluation results: {eval_folder}")
    print(f"{'='*60}")

    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process VTC_Bench data in parallel using multiprocessing with resume capability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config_vtc_bench.yaml in script directory (with resume enabled by default)
  python run_vtc_bench_parallel_models.py
  
  # Use custom config file (positional argument, resume enabled by default)
  python run_vtc_bench_parallel_models.py /path/to/config.yaml
  
  # Use custom config file (with -c flag, resume enabled by default)
  python run_vtc_bench_parallel_models.py -c /path/to/config.yaml
  
  # Use custom config file from current directory
  python run_vtc_bench_parallel_models.py eval_config/config.yaml
  
  # Resume from previous run (if results already exist, will skip processed items)
  python run_vtc_bench_parallel_models.py eval_config/config.yaml
  
  # Force reprocessing all items (disable resume mode)
  python run_vtc_bench_parallel_models.py eval_config/config.yaml --no-resume
        """
    )
    
    parser.add_argument(
        'config_file',
        nargs='?',
        help='Path to configuration YAML file (default: config_vtc_bench.yaml in script directory)'
    )
    
    parser.add_argument(
        '-c', '--config',
        dest='config_path',
        help='Path to configuration YAML file (alternative way to specify)'
    )
    
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        default=True,
        help='Disable resume mode and reprocess all items (default: resume mode enabled)'
    )
    
    parser.add_argument(
        '--eval-method',
        type=str,
        dest='eval_method',
        default='llm',
        choices=['heuristic', 'llm'],
        help='Evaluation method (default: llm)'
    )

    parser.add_argument(
        '--eval-model',
        type=str,
        dest='eval_model',
        default='gpt-4o-mini',
        help='Model for LLM evaluation (default: gpt-4o-mini)'
    )

    parser.add_argument(
        '--eval-api-key',
        type=str,
        dest='eval_api_key',
        default='',
        help='API key for evaluation'
    )

    parser.add_argument(
        '--eval-api-base',
        type=str,
        dest='eval_api_base',
        default='',
        help='API base URL for evaluation'
    )

    parser.add_argument(
        '--skip-eval',
        dest='skip_eval',
        action='store_true',
        default=False,
        help='Skip post-processing and evaluation (default: run post-processing)'
    )

    return parser.parse_args()


def main():
    """Main function - Parallel processing with 10 workers."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine config file path
    if args.config_path:
        # -c flag has priority
        config_path = args.config_path
    elif args.config_file:
        # Positional argument
        config_path = args.config_file
    else:
        # Default to config_vtc_bench.yaml in script directory
        config_path = os.path.join(script_dir, 'config_vtc_bench.yaml')
    
    # Make path absolute if it's relative
    if not os.path.isabs(config_path):
        # If relative path, resolve it relative to current working directory or script directory
        if os.path.exists(config_path):
            config_path = os.path.abspath(config_path)
        else:
            # Try script directory
            alt_path = os.path.join(script_dir, config_path)
            if os.path.exists(alt_path):
                config_path = alt_path
            else:
                config_path = os.path.abspath(config_path)
    
    # Load config
    print("🔧 Loading configuration...")
    print(f"   Config file: {config_path}")
    config = load_config(config_path)
    
    if not config:
        print("❌ Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # Get configuration values
    input_config = config.get('input', {})
    output_config = config.get('output', {})
    processing_config = config.get('processing', {})
    llm_config = config.get('llm', {})
    
    tsv_path = input_config.get('tsv_path')
    output_dir = output_config.get('results_dir')

    # Get data_dir: priority is explicit config > directory containing tsv_path
    data_dir = input_config.get('data_dir', None)
    if not data_dir and tsv_path:
        # Auto-detect data directory from tsv_path
        data_dir = os.path.dirname(tsv_path)

    start_idx = input_config.get('start_idx', 0)
    end_idx = input_config.get('end_idx')
    skip_errors = processing_config.get('skip_errors', True)
    prompt_template = config.get('prompt_template')
    num_workers = processing_config.get('num_workers', 4)  # Get from config or default to 4 (was 10)
    model_name_full = llm_config.get('model', 'qwen-vl-max-0809')  # Use model name from config (keep original for API)
    
    # Extract model name after '/' if it contains one for folder naming (e.g., "Qwen/Qwen3-VL-235B" -> "Qwen3-VL-235B")
    model_name_for_folder = model_name_full
    if '/' in model_name_for_folder:
        model_name_for_folder = model_name_for_folder.split('/')[-1]
    
    # Initialize API key rotator for parallel processing (before multiprocessing starts)
    api_keys = llm_config.get('api_keys', [])
    if api_keys and len(api_keys) > 0:
        print(f"🔑 API Key Rotation: Initialized with {len(api_keys)} keys")
        initialize_api_key_rotator(api_keys)
    else:
        print(f"🔑 API Key Rotation: Disabled (using single API key)")

    # Check if DASHSCOPE_API_KEY is set (only for dashscope models)
    model_type = llm_config.get('model_type', 'qwenvl_dashscope')
    if model_type == 'qwenvl_dashscope' and not os.getenv('DASHSCOPE_API_KEY'):
        api_key = llm_config.get('api_key')
        if not api_key:
            print("❌ Error: DASHSCOPE_API_KEY environment variable not set and no api_key in config!")
            print("Please set it with: export DASHSCOPE_API_KEY='your_api_key'")
            sys.exit(1)
    
    # Check if TSV file exists
    if not os.path.exists(tsv_path):
        print(f"❌ Error: TSV file not found at {tsv_path}")
        sys.exit(1)
    
    print(f"\n📋 Configuration Summary:")
    print(f"  TSV Path: {tsv_path}")
    print(f"  Base Output Dir: {output_dir}")
    print(f"  Model Name (API): {model_name_full}")
    print(f"  Model Name (Folder): {model_name_for_folder}")
    print(f"  Model Results Dir: {os.path.join(output_dir, model_name_for_folder)}")
    print(f"  Model Images Dir: /Users/xuanyuzhu/benchmark/code/Qwen-Agent/logs/tmp_img/{model_name_for_folder}")
    print(f"  Processing Range: {start_idx} to {end_idx or 'all'}")
    print(f"  Skip Errors: {skip_errors}")
    print(f"  🔀 Parallel Workers (Concurrent): {num_workers}")
    print(f"  🔑 API Keys: {len(api_keys)} keys configured" if api_keys else f"  🔑 API Keys: Single key mode")
    print(f"  📋 Resume Mode: {'✅ Enabled' if args.resume else '❌ Disabled'}")
    print(f"  ⚡ Parallelism: Multi-process with {num_workers} concurrent workers (actual API requests scale with worker count)\n")
    
    # Process data in parallel
    results_jsonl = process_vtc_bench_data_parallel(
        tsv_path=tsv_path,
        output_dir=output_dir,
        config=config,
        start_idx=start_idx,
        end_idx=end_idx,
        skip_errors=skip_errors,
        prompt_template=prompt_template,
        num_workers=num_workers,
        model_name=model_name_for_folder,
        resume=args.resume
    )

    # Post-process results if not skipped
    if results_jsonl and not args.skip_eval:
        print(f"\n{'='*60}")
        print(f"🔄 Starting Post-Processing (JSONL → Excel → Evaluation)")
        print(f"{'='*60}")

        # Get evaluation settings from arguments or use defaults
        eval_method = args.eval_method
        eval_model = args.eval_model
        eval_api_key = args.eval_api_key
        eval_api_base = args.eval_api_base

        # Get use_raw_api setting from config
        use_raw_api = llm_config.get('generate_cfg', {}).get('use_raw_api', False)

        if "ablation" in output_config.get('results_dir', ""):
            if "strong" in output_config.get('results_dir', ""):
                # Run post-processing
                success = post_process_results(
                    jsonl_path=results_jsonl,
                    tsv_path=tsv_path,
                    model_name=model_name_for_folder,
                    benchmark_name='VTC_Bench_Ablation_Strong',
                    eval_method=eval_method,
                    eval_model=eval_model,
                    eval_api_key=eval_api_key,
                    eval_api_base=eval_api_base,
                    nproc=num_workers,
                    data_dir=data_dir,
                    config_file=config_path,
                    use_raw_api=use_raw_api
                )
            else:
                # Run post-processing
                success = post_process_results(
                    jsonl_path=results_jsonl,
                    tsv_path=tsv_path,
                    model_name=model_name_for_folder,
                    benchmark_name='VTC_Bench_Ablation_Weak',
                    eval_method=eval_method,
                    eval_model=eval_model,
                    eval_api_key=eval_api_key,
                    eval_api_base=eval_api_base,
                    nproc=num_workers,
                    data_dir=data_dir,
                    config_file=config_path,
                    use_raw_api=use_raw_api
                )
        else:
            # Run post-processing
            success = post_process_results(
                jsonl_path=results_jsonl,
                tsv_path=tsv_path,
                model_name=model_name_for_folder,
                benchmark_name='VTC_Bench',
                eval_method=eval_method,
                eval_model=eval_model,
                eval_api_key=eval_api_key,
                eval_api_base=eval_api_base,
                nproc=num_workers,
                data_dir=data_dir,
                config_file=config_path,
                use_raw_api=use_raw_api
            )

        if success:
            print(f"\n✅ Complete pipeline finished successfully!")
        else:
            print(f"\n⚠️  Post-processing encountered some issues. Check logs above for details.")
    elif results_jsonl and args.skip_eval:
        print(f"\n{'='*60}")
        print(f"⏭️  Post-processing skipped (--skip-eval flag enabled)")
        print(f"{'='*60}")
        print(f"JSONL results saved at: {results_jsonl}")
        print(f"To run post-processing later, use:")
        print(f"  python run_vtc_bench_parallel_models.py config.yaml --skip-eval=false")
    else:
        print(f"\n❌ No results generated. Skipping post-processing.")


if __name__ == '__main__':
    main()
