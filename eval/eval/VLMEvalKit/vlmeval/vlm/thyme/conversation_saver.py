"""
Module for saving conversation history and intermediate images from Thyme model inference.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConversationSaver:
    """
    Saves conversation history and intermediate images generated during model inference.
    
    Structure:
    - conversation_history.json: Full conversation including text and image references
    - intermediate_images/: Directory containing all intermediate images
    - metadata.json: Metadata about the conversation (timestamps, iterations, etc.)
    """
    
    def __init__(self, save_dir: str, sample_id: Optional[int] = None, sample_name: Optional[str] = None, verbose: bool = False):
        """
        Initialize the ConversationSaver.
        
        Args:
            save_dir: Base directory to save conversations
            sample_id: Optional ID for the sample being processed
            sample_name: Optional name for the sample (used if sample_id is not provided)
            verbose: Whether to print debug information
        """
        self.save_dir = Path(save_dir)
        self.sample_id = sample_id
        self.sample_name = sample_name
        self.verbose = verbose
        
        # Create sample-specific directory based on sample_id or sample_name
        if sample_id is not None:
            self.sample_dir = self.save_dir / f"sample_{sample_id}"
        elif sample_name is not None:
            # Use sample_name to create unique directory
            self.sample_dir = self.save_dir / sample_name
        else:
            self.sample_dir = self.save_dir
        
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.sample_dir / "intermediate_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversation_history = []
        self.api_calls = []  # Track each API call with its messages and response
        self.image_counter = 0
        self.metadata = {
            'sample_id': sample_id,
            'total_iterations': 0,
            'total_images': 0,
            'generation_attempts': 0,
            'total_api_calls': 0,
        }
        
        if self.verbose:
            logger.info(f"ConversationSaver initialized at {self.sample_dir}")
    
    def add_user_message(self, text: str, image_paths: Optional[List[str]] = None) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            text: User's text message
            image_paths: Optional list of image paths provided by the user
        """
        message = {
            'role': 'user',
            'type': 'text',
            'content': text,
            'images': []
        }
        
        if image_paths:
            for img_path in image_paths:
                if os.path.exists(img_path):
                    message['images'].append(os.path.basename(img_path))
        
        self.conversation_history.append(message)
        if self.verbose:
            logger.info(f"Added user message: {text[:100]}...")
    
    def add_assistant_message(
        self, 
        text: str, 
        code_blocks: Optional[List[str]] = None,
        intermediate_images: Optional[List[str]] = None
    ) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            text: Assistant's text response
            code_blocks: Optional list of code blocks generated
            intermediate_images: Optional list of intermediate image paths
        """
        message = {
            'role': 'assistant',
            'type': 'text',
            'content': text,
            'code_blocks': code_blocks or [],
            'images': []
        }
        
        # Copy and register intermediate images
        if intermediate_images:
            if self.verbose:
                logger.info(f"Processing {len(intermediate_images)} intermediate images: {intermediate_images}")
            
            for img_path in intermediate_images:
                if self.verbose:
                    logger.info(f"Checking image path: {img_path}, exists: {os.path.exists(img_path)}")
                
                if os.path.exists(img_path):
                    saved_name = self._save_intermediate_image(img_path)
                    if saved_name:
                        message['images'].append(saved_name)
                        if self.verbose:
                            logger.info(f"Successfully saved image: {saved_name}")
                else:
                    if self.verbose:
                        logger.warning(f"Image path does not exist: {img_path}")
        
        self.conversation_history.append(message)
        self.metadata['total_images'] = self.image_counter
        
        if self.verbose:
            logger.info(f"Added assistant message with {len(message['images'])} images")
    
    def _save_intermediate_image(self, src_path: str) -> str:
        """
        Copy an intermediate image to the conversation directory.
        
        Args:
            src_path: Source path of the image
            
        Returns:
            Filename of the saved image
        """
        if not src_path:
            return ""
        
        # Check if file exists, if not, wait a bit and try again
        import time
        max_retries = 5
        retry_count = 0
        
        while not os.path.exists(src_path) and retry_count < max_retries:
            time.sleep(0.1)
            retry_count += 1
        
        if not os.path.exists(src_path):
            logger.warning(f"Image path does not exist after retries: {src_path}")
            return ""
        
        # Generate unique filename
        file_ext = os.path.splitext(src_path)[1]
        filename = f"image_{self.image_counter:04d}{file_ext}"
        dst_path = self.images_dir / filename
        
        try:
            # Ensure destination directory exists
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_path, dst_path)
            self.image_counter += 1
            
            if self.verbose:
                logger.info(f"Saved intermediate image: {filename} from {src_path}")
            
            return filename
        except Exception as e:
            logger.error(f"Failed to save image {src_path} to {dst_path}: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def add_generation_attempt(self, attempt_num: int, success: bool, reason: str = "") -> None:
        """
        Record information about a generation attempt.
        
        Args:
            attempt_num: Attempt number (1-indexed)
            success: Whether the attempt succeeded
            reason: Optional reason for failure
        """
        self.metadata['generation_attempts'] = max(
            self.metadata['generation_attempts'], 
            attempt_num
        )
        if self.verbose:
            status = "SUCCESS" if success else f"FAILED ({reason})"
            logger.info(f"Generation attempt {attempt_num}: {status}")
    
    def add_iteration(self, iteration_num: int) -> None:
        """
        Record an iteration.
        
        Args:
            iteration_num: Iteration number (1-indexed)
        """
        self.metadata['total_iterations'] = max(
            self.metadata['total_iterations'],
            iteration_num
        )
    
    def add_api_call(self, messages: List[Dict], response: str, iteration_num: int = 0) -> None:
        """
        Record an API call with its input messages and response.
        
        Args:
            messages: The messages sent to the API
            response: The response from the API
            iteration_num: Optional iteration number
        """
        api_call = {
            'api_call_num': len(self.api_calls) + 1,
            'iteration': iteration_num,
            'messages': messages,
            'response': response,
        }
        self.api_calls.append(api_call)
        self.metadata['total_api_calls'] = len(self.api_calls)
        
        if self.verbose:
            logger.info(f"Recorded API call #{api_call['api_call_num']} with {len(messages)} messages")
    
    def save_conversation(self) -> str:
        """
        Save the conversation history to a JSON file.
        
        Returns:
            Path to the saved conversation file
        """
        conv_file = self.sample_dir / "conversation_history.json"
        
        try:
            with open(conv_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            
            if self.verbose:
                logger.info(f"Conversation saved to {conv_file}")
            
            return str(conv_file)
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return ""
    
    def save_metadata(self) -> str:
        """
        Save metadata about the conversation.
        
        Returns:
            Path to the saved metadata file
        """
        meta_file = self.sample_dir / "metadata.json"
        
        try:
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            
            if self.verbose:
                logger.info(f"Metadata saved to {meta_file}")
            
            return str(meta_file)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return ""
    
    def save_api_calls(self) -> str:
        """
        Save the API call history to a JSON file.
        
        Returns:
            Path to the saved API calls file
        """
        api_calls_file = self.sample_dir / "api_calls.json"
        
        try:
            with open(api_calls_file, 'w', encoding='utf-8') as f:
                json.dump(self.api_calls, f, indent=2, ensure_ascii=False)
            
            if self.verbose:
                logger.info(f"API calls saved to {api_calls_file}")
            
            return str(api_calls_file)
        except Exception as e:
            logger.error(f"Failed to save API calls: {e}")
            return ""
    
    def finalize(self) -> Dict[str, str]:
        """
        Finalize the conversation saving by writing all files.
        
        Returns:
            Dictionary with paths to saved files
        """
        result = {
            'sample_dir': str(self.sample_dir),
            'conversation_file': self.save_conversation(),
            'metadata_file': self.save_metadata(),
            'api_calls_file': self.save_api_calls() if self.api_calls else None,
            'images_dir': str(self.images_dir),
            'total_images': self.image_counter,
        }
        
        if self.verbose:
            logger.info(f"Conversation finalized. Total images: {self.image_counter}, Total API calls: {len(self.api_calls)}")
        
        return result
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Dictionary containing conversation summary
        """
        return {
            'total_messages': len(self.conversation_history),
            'total_images': self.image_counter,
            'total_iterations': self.metadata['total_iterations'],
            'generation_attempts': self.metadata['generation_attempts'],
            'sample_id': self.sample_id,
            'sample_dir': str(self.sample_dir),
        }


def create_conversation_saver(
    work_dir: str,
    model_name: str,
    dataset_name: str,
    sample_id: Optional[int] = None,
    image_path: Optional[str] = None,
    verbose: bool = False
) -> ConversationSaver:
    """
    Factory function to create a ConversationSaver with standard directory structure.
    
    Args:
        work_dir: Base work directory
        model_name: Name of the model
        dataset_name: Name of the dataset
        sample_id: Optional sample ID
        image_path: Optional path to the input image (used to generate unique sample name if sample_id is not provided)
        verbose: Whether to print debug information
        
    Returns:
        ConversationSaver instance
    """
    save_dir = os.path.join(
        work_dir,
        f"{model_name}_{dataset_name}_conversations"
    )
    
    # Generate sample_name if sample_id is not provided
    sample_name = None
    if sample_id is None and image_path is not None:
        # Extract image name without extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        # Use timestamp to make it unique
        import time
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        sample_name = f"{image_name}_{timestamp}"
    
    return ConversationSaver(
        save_dir, 
        sample_id=sample_id, 
        sample_name=sample_name,
        verbose=verbose
    )