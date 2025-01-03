import time
import os
import json
from pathlib import Path
from openai import AzureOpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import logging
import sys
import select
import math
from termcolor import colored

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            if "User input" in record.msg:
                record.msg = colored(record.msg, 'green')
            elif "Response" in record.msg:
                record.msg = colored(record.msg, 'blue')
            elif "Self-reflection" in record.msg:
                record.msg = colored(record.msg, 'yellow')
        elif record.levelno == logging.ERROR:
            record.msg = colored(record.msg, 'red')
        return super().format(record)

# Setup logging with colors
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(message)s', 
                                   datefmt='%H:%M:%S'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = [handler]

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
print(env_path)
load_dotenv(env_path)

# Validate required environment variables
required_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_DEPLOYMENT_NAME"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

MAX_TOKENS = 4096
MAX_RETRIES = 3
MIN_PAUSE = 1
MAX_PAIRS = 10

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-07-01-preview",
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

class LLM:
    def __init__(self, initial_state="", save_path="conversation_history.json"):
        self._internal_state = "initialized"  # Private attribute
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.save_path = Path(save_path)
        self.last_request_time = 0
        self.conversation_history = self.load_history() or []
        self._validate_and_init_state(initial_state)
        
    @property
    def internal_state(self):
        """Get the internal state of the LLM."""
        return self._internal_state

    @internal_state.setter
    def internal_state(self, value):
        """Set the internal state of the LLM."""
        logger.debug(f"State change: {self._internal_state} -> {value}")
        self._internal_state = value

    def process_message(self, message):
        self._internal_state = "processing"
        # ...existing processing code...
        self._internal_state = "completed"

    def handle_error(self, error):
        self._internal_state = "error"
        logger.error(f"Error: {str(error)}")
        # ...existing error handling...

    def _validate_and_init_state(self, initial_state):
        if initial_state and not self.conversation_history:
            if not isinstance(initial_state, str):
                logger.error("Invalid initial state type")
                raise ValueError("Initial state must be a string")
            self.conversation_history.append({
                "role": "system",
                "content": initial_state
            })

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=MIN_PAUSE))
    def generate_response(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Invalid prompt")
            
        self._rate_limit()
        self._maintain_history_size()
        
        if self._count_tokens(self.conversation_history) > MAX_TOKENS:
            self._summarize_history()
            
        try:
            self.conversation_history.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=deployment_name,
                messages=self.conversation_history,
                max_tokens=150,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            self.save_history()
            return response_text
            
        except (APIError, RateLimitError, APITimeoutError) as e:
            print(f"API Error: {str(e)}")
            raise
            
    def _rate_limit(self):
        """Ensure 1 second between API calls"""
        now = time.time()
        if now - self.last_request_time < 1:
            time.sleep(1 - (now - self.last_request_time))
        self.last_request_time = time.time()
        
    def _maintain_history_size(self):
        """Keep only recent conversation pairs plus system message"""
        if len(self.conversation_history) > (MAX_PAIRS * 2 + 1):
            system_message = None
            if self.conversation_history[0]["role"] == "system":
                system_message = self.conversation_history.pop(0)
            self.conversation_history = self.conversation_history[-(MAX_PAIRS * 2):]
            if system_message:
                self.conversation_history.insert(0, system_message)
                
    def _summarize_history(self):
        """Summarize conversation when it gets too long"""
        try:
            summary_prompt = "Summarize the key points of this conversation:"
            for msg in self.conversation_history[1:]:  # Skip system message
                summary_prompt += f"\n{msg['role']}: {msg['content']}"
                
            response = client.chat.completions.create(
                model=deployment_name,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=150
            )
            
            summary = response.choices[0].message.content.strip()
            self.conversation_history = [
                self.conversation_history[0],  # Keep system message
                {"role": "system", "content": f"Previous conversation summary: {summary}"}
            ]
        except Exception as e:
            print(f"Error summarizing conversation: {e}")

    def summarize_conversation(self):
        try:
            response = self._get_summary()
            summary = response.choices[0].message.content.strip()
            logger.info("Conversation summarized: %s", summary)
            self.conversation_history = [
                self.conversation_history[0],  # Keep system message
                {"role": "system", "content": f"Previous conversation summary: {summary}"}
            ]
        except Exception as e:
            logger.error("Error summarizing conversation: %s", str(e))

    def save_history(self):
        """Save conversation history to JSON file"""
        try:
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
                logger.info("Conversation history saved to %s", self.save_path)
        except Exception as e:
            logger.error("Error saving conversation history: %s", str(e))

    def load_history(self):
        """Load conversation history from JSON file"""
        try:
            if self.save_path.exists():
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    logger.info("Loaded conversation history from %s", self.save_path)
                    return history
        except Exception as e:
            logger.error("Error loading conversation history: %s", str(e))
            return None

    def _count_tokens(self, conversation_history):
        """Estimate token count - adjust based on your tokenizer"""
        return sum(len(msg['content'].split()) * 1.3 for msg in conversation_history)

    def get_conversation_history(self):
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history and saved file"""
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []
        self.save_history()

class ConversationManager:
    def __init__(self, llm, initial_reflection_rate=1):
        self.llm = llm
        self.running = True
        self.last_reflection_time = 0
        self.last_user_interaction = time.time()
        self.initial_reflection_rate = initial_reflection_rate
        self.max_interval = 3600  # 1 hour in seconds

    def start_internal_reflection(self):
        while self.running:
            current_time = time.time()
            
            # Check stdin for user input
            if select.select([sys.stdin], [], [], 0.0)[0]:
                user_input = sys.stdin.readline().strip()
                if user_input:
                    self.last_user_interaction = current_time
                    logger.info("User input: %s", user_input)
                    response = self.llm.generate_response(user_input)
                    logger.info("Response: %s", response)
                    continue

            # Calculate dynamic reflection rate
            time_since_interaction = current_time - self.last_user_interaction
            interval = min(
                self.max_interval,
                (1.0 / self.initial_reflection_rate) * math.pow(2, time_since_interaction / 60)
            )
            reflection_rate = 1.0 / interval

            # Self-reflection with backoff
            if current_time - self.last_reflection_time >= interval:
                internal_prompt = f"Reflect on your purpose and the user input: {self.llm.internal_state}"
                response = self.llm.generate_response(internal_prompt)
                logger.info("Self-reflection (%.1f sec interval): %s", interval, response)
                self.last_reflection_time = current_time

            time.sleep(0.1)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    try:
        my_llm = LLM(initial_state="Ready!")
        conversation_manager = ConversationManager(my_llm, initial_reflection_rate=1)
        
        logger.info("Starting continuous reflection loop. Press CTRL+C to exit.")
        conversation_manager.start_internal_reflection()
    
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        conversation_manager.running = False
    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        raise
