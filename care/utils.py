


"""
Utility functions for experiment_v2 pipeline.

Includes LLM client, I/O helpers, and common utilities.
"""

import json
import logging
import os
import re
import threading
import time
import requests as _requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from openai import AzureOpenAI, OpenAI


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(name: str, level: str = 'INFO') -> logging.Logger:
    """Set up logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """
    Wrapper for Azure OpenAI and OpenAI-compatible API calls with retry logic.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        max_retries: int = 3,
        timeout: int = 300,
        temperature: float = 1.0,
        max_tokens: int = 16384,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize LLM client for Azure OpenAI or OpenAI-compatible endpoints.

        Args:
            model: Model name (e.g., 'gpt-5', 'llama-3.3-70b-instruct')
            api_key: SecureGPT API key
            api_version: Azure API version (unused for Llama)
            azure_endpoint: API endpoint URL
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            rate_limit_delay: Delay between API calls
        """
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit_delay = rate_limit_delay
        self.logger = setup_logging(f'LLMClient.{model}')

        # Determine client type based on endpoint/model
        self.is_claude = 'claude' in model.lower()
        self.is_gemini = 'gemini' in model.lower()
        self.is_azure_openai = 'openai-eastus2' in azure_endpoint

        if self.is_claude:
            # Claude uses Stanford APIM prompt_text endpoint (raw REST)
            self.claude_endpoint = azure_endpoint
            self.claude_headers = {
                'Ocp-Apim-Subscription-Key': api_key,
                'Content-Type': 'application/json',
            }
            self.client = None
        
        elif self.is_gemini:
            self.gemini_endpoint = azure_endpoint
            self.gemini_headers = {
                'Ocp-Apim-Subscription-Key': api_key,
                'Content-Type': 'application/json',
            }
            self.client = None

        elif self.is_azure_openai:
            # Initialize Azure OpenAI client (for GPT-5)
            headers = {
                'Ocp-Apim-Subscription-Key': api_key,
                'Content-Type': 'application/json',
            }

            self.client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_deployment=model,
                default_headers=headers,
                azure_ad_token=api_key,
                timeout=timeout,
            )
        else:
            # Initialize standard OpenAI client (for Llama)
            # Use a placeholder api_key and put real key in headers
            headers = {
                'Ocp-Apim-Subscription-Key': api_key,
            }

            self.client = OpenAI(
                base_url=azure_endpoint.replace('/chat/completions', ''),
                api_key='placeholder',  # Required by OpenAI client but not used
                default_headers=headers,
                timeout=timeout,
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate completion from LLM with retry logic.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Override default temperature (not used for GPT-5)
            max_tokens: Override default max_tokens

        Returns:
            Generated text response

        Raises:
            Exception: If all retries fail
        """
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.is_claude:
            return self._generate_claude(system_prompt, user_prompt, temperature, max_tokens)

        if self.is_gemini:
            return self._generate_gemini(system_prompt, user_prompt, temperature, max_tokens)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]

        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)

                # Build API call parameters
                api_params = {
                    'model': self.model,
                    'messages': messages,
                }

                # GPT-5 uses max_completion_tokens, Llama uses max_tokens
                if 'gpt-5' in self.model.lower():
                    api_params['max_completion_tokens'] = max_tokens
                else:
                    api_params['max_tokens'] = max_tokens
                    if temperature is not None:
                        api_params['temperature'] = temperature
                    if top_p is not None:
                        api_params['top_p'] = top_p

                # API call
                response = self.client.chat.completions.create(**api_params)

                # Debug: Log response details
                content = response.choices[0].message.content
                self.logger.debug(f"Raw API response content: {repr(content)}")
                self.logger.debug(f"Response finish_reason: {response.choices[0].finish_reason}")

                if content is None or not content.strip():
                    finish_reason = response.choices[0].finish_reason
                    self.logger.warning(
                        f"Empty/None content from API (attempt {attempt + 1}/{self.max_retries}); "
                        f"finish_reason={finish_reason}"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(self.rate_limit_delay)
                        continue
                    return ""

                return content.strip()

            except Exception as e:
                error_msg = str(e)
                if 'rate' in error_msg.lower():
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(1)

        raise Exception(f"Failed after {self.max_retries} attempts")

    def _generate_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion from Claude via Stanford APIM prompt_text endpoint."""
        # Stanford APIM only accepts prompt_text — combine system + user
        prompt_text = f"{system_prompt}\n\n{user_prompt}"

        payload = {
            "prompt_text": prompt_text,
            "max_tokens": max_tokens or 8000,
        }
        temp = temperature if temperature is not None else self.temperature
        if temp is not None:
            payload["temperature"] = temp

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)

                resp = _requests.post(
                    self.claude_endpoint,
                    headers=self.claude_headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()

                data = resp.json()
                # Anthropic Messages API response format
                content = data["content"][0]["text"]
                self.logger.debug(f"Raw Claude response: {repr(content[:200])}")

                if content is None:
                    self.logger.warning("Claude returned None content!")
                    return ""

                return content.strip()

            except Exception as e:
                error_msg = str(e)
                if 'rate' in error_msg.lower() or (hasattr(e, 'response') and getattr(e, 'response', None) is not None and e.response.status_code == 429):
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Claude error (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(1)

        raise Exception(f"Claude failed after {self.max_retries} attempts")

    def _generate_gemini(
    self,
    system_prompt: str,
    user_prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion from Gemini via Stanford APIM endpoint."""
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            "generation_config": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxOutputTokens": max_tokens or self.max_tokens,
            },
        }

        for attempt in range(self.max_retries):
            try:
                time.sleep(self.rate_limit_delay)

                resp = _requests.post(
                    self.gemini_endpoint,
                    headers=self.gemini_headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()

                # Response is a streaming array of chunks — concatenate text parts
                chunks = resp.json()
                content = "".join(
                    chunk["candidates"][0]["content"]["parts"][0].get("text", "")
                    for chunk in chunks
                    if chunk.get("candidates")
                    and chunk["candidates"][0].get("content")
                    and chunk["candidates"][0]["content"].get("parts")
                )

                self.logger.debug(f"Raw Gemini response: {repr(content[:200])}")
                return content.strip()

            except Exception as e:
                error_msg = str(e)
                if 'rate' in error_msg.lower() or (
                    hasattr(e, 'response') and e.response is not None
                    and e.response.status_code == 429
                ):
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Rate limit (attempt {attempt+1}/{self.max_retries}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Gemini error (attempt {attempt+1}/{self.max_retries}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(1)

        raise Exception(f"Gemini failed after {self.max_retries} attempts")
        

    def generate_batch(
        self,
        prompts: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> List[str]:
        """Sequential fallback for API models — one request per prompt."""
        return [self.generate(sys_p, usr_p, temperature=temperature, max_tokens=max_tokens, top_p=top_p)
                for sys_p, usr_p in prompts]

    def batch_generate(
        self,
        prompts: List[Tuple[str, str]],
        show_progress: bool = True,
    ) -> List[str]:
        return self.generate_batch(prompts)


# ============================================================================
# Local Model Client (HuggingFace transformers)
# ============================================================================

class LocalLLMClient:
    """
    Judge client for a locally downloaded HuggingFace model.

    Loads the model once (process-level singleton) and serializes GPU inference
    with a lock.  Supports the same .generate() interface as LLMClient.

    Usage:
        client = LocalLLMClient("/path/to/model")
        response = client.generate(system_prompt, user_prompt)
    """

    _model = None
    _tokenizer = None
    _loaded_path: Optional[str] = None
    _load_lock = threading.Lock()
    _infer_lock = threading.Lock()

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        rate_limit_delay: float = 0.0,
        max_retries: int = 3,
    ):
        self.model_path = str(model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.logger = setup_logging(f'LocalLLMClient.{Path(model_path).name}')
        self._ensure_loaded()

    def _ensure_loaded(self):
        with LocalLLMClient._load_lock:
            if LocalLLMClient._loaded_path == self.model_path:
                return
            self.logger.info(f"Loading local model from {self.model_path} ...")
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            LocalLLMClient._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            LocalLLMClient._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            LocalLLMClient._model.eval()
            LocalLLMClient._loaded_path = self.model_path
            self.logger.info("Local model loaded.")

    @staticmethod
    def _build_prefix_fn(tokenizer, schema: dict):
        """Build a lm-format-enforcer prefix_allowed_tokens_fn for the given JSON schema."""
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        parser = JsonSchemaParser(schema)
        return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        import torch

        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        tokenizer = LocalLLMClient._tokenizer
        model = LocalLLMClient._model

        # Disable thinking mode (Qwen3 series) so output is clean JSON
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        prefix_fn = self._build_prefix_fn(tokenizer, response_format) if response_format else None

        for attempt in range(self.max_retries):
            try:
                if self.rate_limit_delay:
                    time.sleep(self.rate_limit_delay)

                with LocalLLMClient._infer_lock:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=temp > 0,
                            temperature=temp if temp > 0 else None,
                            top_p=top_p if (temp > 0 and top_p is not None) else None,
                            pad_token_id=tokenizer.eos_token_id,
                            prefix_allowed_tokens_fn=prefix_fn,
                        )

                new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                return raw

            except Exception as e:
                self.logger.error(f"Local inference error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)

        raise Exception(f"LocalLLMClient failed after {self.max_retries} attempts")

    def _apply_template(self, system_prompt: str, user_prompt: str) -> str:
        tokenizer = LocalLLMClient._tokenizer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def generate_batch(
        self,
        prompts: List[Tuple[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        response_format: Optional[dict] = None,
    ) -> List[str]:
        """
        Run multiple (system, user) prompt pairs as a single batched GPU call.
        Left-pads inputs so all sequences share one model.generate() invocation.
        """
        import torch

        if not prompts:
            return []

        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temp = temperature if temperature is not None else self.temperature

        tokenizer = LocalLLMClient._tokenizer
        model = LocalLLMClient._model

        texts = [self._apply_template(sys_p, usr_p) for sys_p, usr_p in prompts]

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        prefix_fn = self._build_prefix_fn(tokenizer, response_format) if response_format else None

        for attempt in range(self.max_retries):
            try:
                with LocalLLMClient._infer_lock:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=temp > 0,
                            temperature=temp if temp > 0 else None,
                            top_p=top_p if (temp > 0 and top_p is not None) else None,
                            pad_token_id=tokenizer.eos_token_id,
                            prefix_allowed_tokens_fn=prefix_fn,
                        )
                results = []
                for output in outputs:
                    raw = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
                    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                    results.append(raw)
                return results
            except Exception as e:
                self.logger.error(f"Batch inference error (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(1)

        raise Exception(f"LocalLLMClient.generate_batch failed after {self.max_retries} attempts")

    def batch_generate(
        self,
        prompts: List[Tuple[str, str]],
        show_progress: bool = True,
    ) -> List[str]:
        return self.generate_batch(prompts)
    
    def batch_forward_token_probs(
        self,
        prompts: List[Tuple[str, str]],
        label_tokens: List[str],
        temperature: float = 1.0,
        batch_size: int = 16,
    ) -> List[Dict[str, float]]:
        """
        Batched forward pass → softmax over label first-tokens.

        For each prompt, runs the chat template up through the assistant turn
        opener and reads the next-token logits. The logits are sliced down to
        the first token id of each label, divided by `temperature`, and
        softmaxed to give a normalized distribution over the label set.

        Notes:
        - `temperature=1.0` returns raw probabilities. CRC is post-hoc, so any
          monotone transformation (incl. T != 1) yields the same threshold
          structure but changes the resolution of the score grid.
        - Multi-piece labels (e.g. "SUPPORTED" → ["SUPP", "ORTED"]) are scored
          on their FIRST piece only. Single-letter labels ("A"/"B"/"C") are
          recommended to avoid prefix collisions.
        """
        import torch

        if not prompts:
            return []

        tokenizer = LocalLLMClient._tokenizer
        model = LocalLLMClient._model

        label_first_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in label_tokens]
        if len(set(label_first_ids)) != len(label_first_ids):
            collisions = {l: i for l, i in zip(label_tokens, label_first_ids)}
            raise ValueError(
                f"Label first-token ids are not unique under this tokenizer: {collisions}. "
                f"Pick labels whose first BPE piece differs (e.g., single letters A/B/C)."
            )

        texts = [self._apply_template(sys_p, usr_p) for sys_p, usr_p in prompts]

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        results = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            inputs = tokenizer(chunk, return_tensors="pt", padding=True)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with LocalLLMClient._infer_lock:
                with torch.no_grad():
                    logits = model(**inputs).logits[:, -1, :]
                    if temperature > 0:
                        logits = logits / temperature
                    target_logits = logits[:, label_first_ids]
                    probs = torch.softmax(target_logits, dim=-1)

            for i in range(len(chunk)):
                results.append({
                    label: probs[i, j].item()
                    for j, label in enumerate(label_tokens)
                })

        return results



def create_llm_client(
    model: str,
    api_key: str = "",
    api_version: str = "",
    azure_endpoint: str = "",
    max_tokens: int = 8000,
    rate_limit_delay: float = 0.5,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> "LLMClient | LocalLLMClient":
    """
    Factory that returns a LocalLLMClient for local paths or an LLMClient for API models.

    Local model convention: model name starts with 'local:' followed by the path
    """
    if model.startswith("local:"):
        model_path = model[len("local:"):]
        return LocalLLMClient(
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
        )
    return LLMClient(
        model=model,
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        max_tokens=max_tokens,
        rate_limit_delay=rate_limit_delay,
        max_retries=max_retries,
        temperature=temperature,
    )


# ============================================================================
# I/O Utilities
# ============================================================================

def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: Path):
    """Save list of dicts to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def save_json(data: Any, filepath: Path, indent: int = 2):
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(filepath: Path) -> Any:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


# ============================================================================
# Sentence Splitting
# ============================================================================

def is_formatting_artifact(sentence: str) -> bool:
    """
    Check if a sentence is a formatting artifact with no real content.

    Formatting artifacts are sentences that contain ONLY:
    - Numbers with optional punctuation: "1", "1.", "2", "2)", etc.
    - Bullet markers: "*", "-", "•"
    - Section markers: "**Follow-up:**", "**Assessment:**"
    - Empty or whitespace only

    Args:
        sentence: Sentence to check

    Returns:
        True if sentence is a formatting artifact, False otherwise
    """
    import re

    # Strip whitespace
    s = sentence.strip()

    # Empty or whitespace only
    if not s:
        return True

    # Just numbers (with or without punctuation): "1", "1.", "2", "2)", "3.", etc.
    # This catches both "1." and "1" (after period splitting)
    if re.fullmatch(r'\*{0,3}\s*\d+[\.\)]?\s*\*{0,3}\s*', s):
        return True

    # Just bullet markers: "*", "-", "•", etc.
    if re.fullmatch(r'[\*\-\•\◦\▪\▸]+\s*', s):
        return True

    # Just markdown bold/italic markers: "**", "***", etc.
    if re.fullmatch(r'\*+\s*', s):
        return True

    # Section headers with no content: "**Follow-up:**", "**Assessment:**", etc.
    # These have bold markers and colon but no other content
    if re.fullmatch(r'\*{2,}[A-Za-z\s\-]+\*{0,2}:\*{0,2}\s*', s):
        return True

    return False


def split_summary_into_sentences(summary: str) -> List[str]:
    """
    Split generated summary into sentences and filter out formatting artifacts. 
    Uses simple period-based splitting with some handling for section headers and special cases. 
    Filters out empty formatting artifacts like "1.", "2.", "*", etc.
    Args: 
        summary: Generated summary text 
    Returns: 
        List of summary sentences (excluding formatting artifacts)
    """
    import re

    summary = summary.strip()
    if not summary:
        return []

    sentences = []

    # Split by newlines first to preserve paragraph structure
    paragraphs = summary.split('\n')

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Skip entire paragraph if it's a formatting artifact
        if is_formatting_artifact(para):
            continue

        # Keep headers as their own "sentence"
        if para.isupper() or para.endswith(':'):
            sentences.append(para)
            continue

        # ---- Protect abbreviations so we don't split on their periods ----
        # common short abbreviations
        para = para.replace("e.g.", "e<DOT>g<DOT>")
        para = para.replace("i.e.", "i<DOT>e<DOT>")
        para = para.replace("vs.", "vs<DOT>")

        # multi-initial abbreviations like M.M.S. / U.S. / A.B.C.
        para = re.sub(r'\b(?:[A-Za-z]\.){2,}', lambda m: m.group(0).replace(".", "<DOT>"), para)

        # genus/initial like "P. acnes", "P. avidum, and P. granulosum"
        para = re.sub(r'\b([A-Za-z])\.(?=\s+[a-z])', r'\1<DOT>', para)

        # ---- Protect common titles so we don't split on their periods ----
        para = re.sub(
            r'\b(Dr|Mr|Mrs|Ms|Prof|St)\.(?=\s)',
            r'\1<DOT>',
            para,
            flags=re.IGNORECASE
        )

        # ---- Protect numbered list prefixes: "1. ", "**1. ", "**12. " ----
        para = re.sub(r'^(\*{0,2}\d+)\.(?=\s|$)', r'\1<DOT>', para)

        # ---- Split on sentence boundaries: ., ?, ! followed by whitespace/end ----
        parts = re.split(r'(?<=[.!?])\s+', para)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # restore dots
            part = part.replace("<DOT>", ".")

            # Skip formatting artifacts after restoration
            if is_formatting_artifact(part):
                continue

            # Do NOT force-add a period. Just ensure it ends with punctuation if needed.
            if part and part[-1] not in ".?!:":
                part = part + "."

            sentences.append(part)

    return sentences



# ============================================================================
# Response Parsing
# ============================================================================

def parse_yes_no_response(response: str) -> bool:
    """
    Parse yes/no response from oracle.

    Args:
        response: Raw LLM response

    Returns:
        True if yes, False if no

    Raises:
        ValueError: If response cannot be parsed
    """
    response = response.strip().lower()

    if 'yes' in response:
        return True
    elif 'no' in response:
        return False
    else:
        raise ValueError(f"Could not parse yes/no from response: {response}")


def parse_score_response(response: str) -> int:
    """
    Parse numeric score (0-10) from judge response.

    Args:
        response: Raw LLM response

    Returns:
        Integer score from 0 to 10

    Raises:
        ValueError: If response cannot be parsed
    """
    response = response.strip()

    # Try direct int conversion first
    try:
        score = int(response)
        if 0 <= score <= 10:
            return score
    except:
        pass

    # Try extracting first number
    import re
    match = re.search(r'\d+', response)
    if match:
        score = int(match.group())
        if 0 <= score <= 10:
            return score

    raise ValueError(f"Could not parse score from response: {response}")


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")

    # Test logging
    logger = setup_logging('test', 'INFO')
    logger.info("Logging works!")

    # Test sentence splitting
    test_summary = "ASSESSMENT\n\nPatient has hypertension. Will start medication.\n\nPLAN\n\nFollow up in 2 weeks."
    sentences = split_summary_into_sentences(test_summary)
    print(f"\nSentence splitting: {len(sentences)} sentences")
    for i, s in enumerate(sentences):
        print(f"  {i+1}. {s}")

    # Test response parsing
    assert parse_yes_no_response("yes") == True
    assert parse_yes_no_response("no") == False
    assert parse_score_response("7") == 7
    assert parse_score_response("The score is 8.") == 8
    print("\n✓ Response parsing works!")

    print("\n✓ All utilities working!")
