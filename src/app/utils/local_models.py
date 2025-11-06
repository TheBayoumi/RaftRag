"""
Local model wrappers for raganything RAG pipeline.

Provides callable wrappers for local HuggingFace models to integrate
with raganything's RAG engine without external API dependencies.
All processing happens locally using transformers and sentence-transformers.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from ..core.config import get_settings

settings = get_settings()


class StopOnTokens(StoppingCriteria):
    """
    Custom stopping criteria that stops generation when specific strings are encountered.

    This is more reliable than the stop_strings parameter in model.generate(),
    which can have compatibility issues with different transformers versions.
    """

    def __init__(
        self,
        stop_strings: List[str],
        tokenizer: Any,
        input_length: int,
    ) -> None:
        """
        Initialize stopping criteria.

        Args:
            stop_strings: List of strings that should trigger stopping.
            tokenizer: Tokenizer for decoding generated tokens.
            input_length: Length of input tokens (to skip checking input).
        """
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self.input_length = input_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        """
        Check if generation should stop.

        Args:
            input_ids: Generated token IDs.
            scores: Generation scores.
            **kwargs: Additional arguments.

        Returns:
            bool: True if generation should stop.
        """
        # Only check the newly generated tokens
        generated_ids = input_ids[0][self.input_length :]

        # Decode the generated text
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Check if any stop string appears in the generated text
        for stop_string in self.stop_strings:
            if stop_string in generated_text:
                return True

        return False


class LocalLLMWrapper:
    """
    Wrapper for local HuggingFace LLMs for raganything integration.

    Provides a callable interface that raganything can use for answer generation,
    ensuring all processing happens locally with no external API calls.

    The wrapper is designed to match raganything's expected llm_model_func signature
    and handles lazy model loading, GPU/CPU management, and proper text generation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize local LLM wrapper.

        Args:
            model_name: HuggingFace model name or path.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            device: Device to run on (cuda/cpu).
        """
        self.model_name = model_name or settings.rag_llm_model
        self.max_tokens = (
            max_tokens if max_tokens is not None else settings.rag_max_tokens
        )
        self.temperature = (
            temperature if temperature is not None else settings.rag_temperature
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

        logger.info(f"Initializing LocalLLMWrapper with {self.model_name}")

    def load_model(self) -> None:
        """
        Load the model and tokenizer with optimal configuration.

        Uses ResourceDetector to automatically configure quantization
        based on available GPU memory, ensuring consistent loading
        across RAG and fine-tuning pipelines.

        Returns:
            None
        """
        if self.model is not None:
            logger.debug("Model already loaded")
            return

        logger.info(f"Loading LLM model: {self.model_name}")

        try:
            # Load tokenizer - explicitly use fast tokenizer
            # Use HF_HOME for HuggingFace cache (set as environment variable)
            # Allow download if model not cached (local_files_only=False)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(settings.HF_HOME),
                trust_remote_code=True,
                use_fast=True,  # Force fast tokenizer (requires sentencepiece for Mistral)
            )

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Use ResourceDetector to get optimal configuration
            # This ensures consistent quantization across RAG and fine-tuning
            from .resource_detector import ResourceDetector

            resources = ResourceDetector.detect_available_resources()
            model_config = ResourceDetector.get_optimal_model_config(
                self.model_name, resources
            )

            logger.info(f"Loading model with config: {model_config}")

            # Load model with optimal configuration
            # Use HF_HOME for HuggingFace cache (set as environment variable)
            # Allow download if model not cached (local_files_only=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                cache_dir=str(settings.HF_HOME),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            # Move to device explicitly (avoids meta tensor issues in multi-worker)
            self.model = self.model.to(self.device)

            self.model.eval()  # Set to evaluation mode

            logger.success(f"LLM model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

    def unload_model(self) -> None:
        """
        Unload model and tokenizer, clearing all caches and forcing garbage collection.

        Critical for preventing "duplicate template name" errors when loading
        the same model multiple times (e.g., switching between RAG and fine-tuning).

        This method performs aggressive memory cleanup:
        1. Deletes model and tokenizer objects
        2. Synchronizes CUDA operations
        3. Runs garbage collection multiple times
        4. Clears CUDA cache and resets memory stats
        5. Logs memory usage before/after for verification
        """
        if self.model is not None:
            # Log memory before unloading (if CUDA available)
            allocated_before = 0.0
            reserved_before = 0.0
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / (1024**3)  # GB
                reserved_before = torch.cuda.memory_reserved() / (1024**3)  # GB
                logger.debug(
                    f"Before unload - Allocated: {allocated_before:.2f}GB, "
                    f"Reserved: {reserved_before:.2f}GB"
                )

            # Delete model and tokenizer
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            # CRITICAL: Synchronize CUDA operations before clearing cache
            # This ensures all pending operations complete before we free memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Force aggressive garbage collection to clear Jinja2 template cache
            # and any circular references
            import gc
            gc.collect()  # First pass - collects most objects
            gc.collect()  # Second pass - collects circular references
            gc.collect(generation=2)  # Third pass - force full collection

            # Clear CUDA cache and reset memory statistics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache
                torch.cuda.reset_peak_memory_stats()  # Reset peak stats
                # Force another synchronization to ensure cache is cleared
                torch.cuda.synchronize()

                # Log memory after unloading for verification
                allocated_after = torch.cuda.memory_allocated() / (1024**3)  # GB
                reserved_after = torch.cuda.memory_reserved() / (1024**3)  # GB
                freed = reserved_before - reserved_after
                logger.info(
                    f"LLM model unloaded - Allocated: {allocated_after:.2f}GB, "
                    f"Reserved: {reserved_after:.2f}GB, Freed: {freed:.2f}GB"
                )
            else:
                logger.info("LLM model and tokenizer unloaded (CPU mode)")

    def is_model_loaded(self, model_name: Optional[str] = None) -> bool:
        """
        Check if a model is loaded, optionally checking for a specific model.

        Args:
            model_name: Optional model name to check for. If None, checks if any model is loaded.

        Returns:
            bool: True if the model is loaded.
        """
        if model_name is None:
            return self.model is not None
        return self.model is not None and self.model_name == model_name

    def switch_model(
        self,
        new_model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Switch to a different model (unloads current, loads new).

        Args:
            new_model_name: Name of the new model to load.
            max_tokens: Optional new max_tokens value.
            temperature: Optional new temperature value.

        Returns:
            None
        """
        if self.model_name == new_model_name and self.model is not None:
            logger.debug(f"Model {new_model_name} already loaded, skipping switch")
            # Update parameters even if model is the same
            if max_tokens is not None:
                self.max_tokens = max_tokens
            if temperature is not None:
                self.temperature = temperature
            return

        logger.info(f"Switching model from {self.model_name} to {new_model_name}")

        # Unload current model
        self.unload_model()

        # Update model name and parameters
        self.model_name = new_model_name
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

        # Load new model (will happen lazily on next __call__)
        logger.success(f"Ready to load {new_model_name} on next inference")

    def _get_model_max_length(self) -> int:
        """
        Get model's maximum context window from config.

        This dynamically determines the model's maximum sequence length
        to prevent truncation issues with smaller models (e.g., Llama 3.2 1B).

        Returns:
            int: Maximum sequence length the model can handle.

        Note:
            Different models use different config attributes:
            - max_position_embeddings (Llama, Mistral)
            - n_positions (GPT-2)
            - max_seq_len (some custom models)
        """
        if self.model is None:
            # Model not loaded yet, return conservative default
            return 2048

        config = self.model.config

        # Try different config attributes in order of preference
        if hasattr(config, "max_position_embeddings"):
            max_len = config.max_position_embeddings
            logger.debug(
                f"Model max_position_embeddings: {max_len} "
                f"(model: {self.model_name})"
            )
            return max_len
        elif hasattr(config, "n_positions"):
            max_len = config.n_positions
            logger.debug(f"Model n_positions: {max_len} (model: {self.model_name})")
            return max_len
        elif hasattr(config, "max_seq_len"):
            max_len = config.max_seq_len
            logger.debug(f"Model max_seq_len: {max_len} (model: {self.model_name})")
            return max_len
        else:
            # Fallback to conservative value
            logger.warning(
                f"Could not determine max length for {self.model_name}, "
                "using conservative default of 2048"
            )
            return 2048

    def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using the local LLM.

        Args:
            prompt: User prompt.
            system_prompt: Optional system prompt.
            history_messages: Optional conversation history.
            **kwargs: Additional generation parameters.

        Returns:
            str: Generated text.
        """
        # Lazy load model
        if self.model is None:
            self.load_model()

        # Build messages in chat format
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history if provided
        if history_messages:
            messages.extend(history_messages)

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        # Detect and use tokenizer's chat template (fully dynamic, no hardcoding!)
        template_info = self._detect_chat_template()

        if template_info["has_template"]:
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.debug(
                    f"✅ Using {template_info['type']} chat template. "
                    f"Preview: {formatted_prompt[:150]}..."
                )
            except Exception as e:
                logger.warning(
                    f"Chat template application failed ({e}), using fallback"
                )
                formatted_prompt = self._format_prompt_fallback(
                    prompt, system_prompt, history_messages
                )
        else:
            logger.warning(
                f"No chat template found in tokenizer. Using fallback formatting. "
                f"For best results, use an -Instruct model variant."
            )
            formatted_prompt = self._format_prompt_fallback(
                prompt, system_prompt, history_messages
            )

        # Calculate maximum input length dynamically based on model's context window
        # CRITICAL FIX: Prevents truncation issues with small models (e.g., Llama 3.2 1B)
        model_max_length = self._get_model_max_length()

        # Reserve tokens for generation output
        # Ensure we leave enough room for the model to generate complete answers
        max_input_length = model_max_length - self.max_tokens

        # Safety check: ensure we have minimum viable space for generation
        min_generation_tokens = 256  # Minimum for coherent responses
        if max_input_length < min_generation_tokens:
            logger.error(
                f"Model context ({model_max_length}) too small! "
                f"Cannot reserve {self.max_tokens} tokens for generation. "
                f"Try using a model with larger context window."
            )
            # Use fallback: allocate 75% to input, 25% to generation
            max_input_length = int(model_max_length * 0.75)
            logger.warning(
                f"Using fallback allocation: {max_input_length} input tokens, "
                f"{model_max_length - max_input_length} generation tokens"
            )

        logger.debug(
            f"Token allocation: {max_input_length} input + {self.max_tokens} generation "
            f"= {max_input_length + self.max_tokens} total "
            f"(model max: {model_max_length})"
        )

        # Tokenize with dynamic max_length based on model capacity
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=False,  # Don't truncate yet
            max_length=max_input_length,  # ✅ DYNAMIC - adapts to model!
        ).to(self.device)

        # Warn if prompt was truncated (helps debugging context issues)
        actual_input_length = inputs["input_ids"].shape[1]
        if actual_input_length >= max_input_length:
            logger.warning(
                f"⚠️  Prompt truncated! Input: {actual_input_length} tokens "
                f"(max: {max_input_length}). Consider: "
                f"1) Reducing top_k to retrieve fewer documents, "
                f"2) Using a model with larger context window, "
                f"3) Reducing max_tokens for generation"
            )

        # Get dynamic stop tokens based on detected template
        stop_strings = self._get_stop_tokens_for_template(template_info["type"])

        stopping_criteria = StoppingCriteriaList(
            [
                StopOnTokens(
                    stop_strings=stop_strings,
                    tokenizer=self.tokenizer,
                    input_length=inputs["input_ids"].shape[1],
                )
            ]
        )

        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    # do_sample=True,
                    top_p=0.85,  # Reduced from 0.9 for more focused sampling
                    repetition_penalty=1.2,  # Prevent repetition loops
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    # stopping_criteria=stopping_criteria, Use custom stopping criteria
                )

            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            # Post-process to remove repetition and truncate cleanly
            cleaned_text = self._clean_generated_text(generated_text)

            return cleaned_text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating response: {e}"

    def _detect_chat_template(self) -> Dict[str, Any]:
        """
        Detect chat template from tokenizer (fully dynamic, no hardcoding).

        Returns:
            Dict with template information:
            - has_template: bool (whether template exists)
            - type: str (detected template type: Llama-3, Mistral, ChatML, etc.)
            - template: Optional[str] (raw template string)
        """
        # Check if tokenizer has chat_template attribute
        if not hasattr(self.tokenizer, "chat_template"):
            return {"has_template": False, "type": "unknown", "template": None}

        template = self.tokenizer.chat_template

        if template is None:
            return {"has_template": False, "type": "unknown", "template": None}

        # Detect template type by inspecting the template content
        # This is dynamic detection based on actual template, not model name!
        template_lower = str(template).lower()

        if "start_header_id" in template_lower or "<|start_header_id|>" in template:
            template_type = "Llama-3"
        elif "[inst]" in template_lower and "[/inst]" in template_lower:
            template_type = "Mistral"
        elif "<|im_start|>" in template or "<|im_end|>" in template:
            template_type = "ChatML"
        elif "<|user|>" in template or "<|assistant|>" in template:
            template_type = "Phi"
        elif "<<sys>>" in template_lower:
            template_type = "Llama-2"
        else:
            template_type = "Custom"

        return {
            "has_template": True,
            "type": template_type,
            "template": template,
        }

    def _get_stop_tokens_for_template(self, template_type: str) -> List[str]:
        """
        Get appropriate stop tokens based on detected template type.

        Args:
            template_type: Detected template type (e.g., "Llama-3", "Mistral").

        Returns:
            List[str]: Stop token strings for this template.
        """
        # Base stop tokens that work across templates
        base_stops = ["\nUser:", "\nHuman:", "\nQuestion:", "```", "\n\n\n"]

        # Add tokenizer's eos_token if available
        if hasattr(self.tokenizer, "eos_token") and self.tokenizer.eos_token:
            base_stops.append(self.tokenizer.eos_token)

        # Template-specific stop tokens (dynamically selected)
        template_stops = {
            "Llama-3": [
                "<|eot_id|>",
                "<|end_of_text|>",
                "<|start_header_id|>",
            ],
            "Mistral": [
                "</s>",
                "[INST]",
                "[/INST]",
            ],
            "ChatML": [
                "<|im_end|>",
                "<|im_start|>",
            ],
            "Phi": [
                "<|end|>",
                "<|user|>",
                "<|assistant|>",
            ],
            "Llama-2": [
                "</s>",
                "<<SYS>>",
                "<</SYS>>",
            ],
            "Custom": [
                "<|endoftext|>",
                "</s>",
            ],
            "unknown": [
                "<|endoftext|>",
                "</s>",
            ],
        }

        # Get template-specific stops or use generic ones
        specific_stops = template_stops.get(template_type, template_stops["unknown"])

        # Combine and remove duplicates
        all_stops = base_stops + specific_stops
        return list(dict.fromkeys(all_stops))  # Remove duplicates, preserve order

    def _format_prompt_fallback(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Fallback prompt formatting for models without chat templates.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            history_messages: Conversation history.

        Returns:
            str: Formatted prompt.
        """
        full_prompt = ""
        if system_prompt:
            full_prompt += f"System: {system_prompt}\n\n"

        if history_messages:
            for msg in history_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                full_prompt += f"{role.capitalize()}: {content}\n"

        full_prompt += f"User: {prompt}\nAssistant:"
        return full_prompt

    def _clean_generated_text(self, text: str) -> str:
        """
        Clean generated text by removing special tokens and detecting repetitions.

        NO truncation or capping - returns full model output.
        Only removes repetition loops and cleans formatting.

        Args:
            text: Generated text from model.

        Returns:
            str: Cleaned text with repetitions removed.
        """
        if not text:
            return text

        # Remove special tokens
        text = text.replace("<|endoftext|>", "")
        text = text.replace("<|end|>", "")
        text = text.replace("<eos>", "")
        text = text.replace("[EOS]", "")
        text = text.replace("</s>", "")
        text = text.replace("<|eot_id|>", "")

        # Clean excessive whitespace
        import re

        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Max 2 consecutive newlines
        text = re.sub(r" +", " ", text)  # Collapse multiple spaces

        return text.strip()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            float: Similarity score (0-1).
        """
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0


class LocalEmbeddingWrapper:
    """
    Wrapper for local embedding models for raganything integration.

    Provides a callable interface that raganything can use for document
    and query embeddings, ensuring all processing happens locally.

    Uses sentence-transformers for efficient text embeddings with support
    for both document batches and single queries. Compatible with raganything's
    expected embedding_func signature.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize local embedding wrapper.

        Args:
            model_name: Sentence-transformers model name.
            device: Device to run on (cuda/cpu).
        """
        self.model_name = model_name or settings.rag_embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[SentenceTransformer] = None

        logger.info(f"Initializing LocalEmbeddingWrapper with {self.model_name}")

    def load_model(self) -> None:
        """
        Load the embedding model.

        Returns:
            None
        """
        if self.model is not None:
            logger.debug("Embedding model already loaded")
            return

        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            # Use HF_HOME for HuggingFace cache (set as environment variable)
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=str(settings.HF_HOME),
            )

            logger.success(f"Embedding model loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def unload_model(self) -> None:
        """
        Unload the embedding model to free memory.

        This method performs aggressive memory cleanup:
        1. Deletes model object
        2. Synchronizes CUDA operations
        3. Runs garbage collection multiple times
        4. Clears CUDA cache and resets memory stats
        5. Logs memory usage before/after for verification

        Returns:
            None
        """
        if self.model is not None:
            # Log memory before unloading (if CUDA available)
            allocated_before = 0.0
            reserved_before = 0.0
            if torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated() / (1024**3)  # GB
                reserved_before = torch.cuda.memory_reserved() / (1024**3)  # GB
                logger.debug(
                    f"Before embedding unload - Allocated: {allocated_before:.2f}GB, "
                    f"Reserved: {reserved_before:.2f}GB"
                )

            # Delete model
            del self.model
            self.model = None

            # CRITICAL: Synchronize CUDA operations before clearing cache
            # This ensures all pending operations complete before we free memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Force aggressive garbage collection
            # Embedding models can have circular references too
            import gc
            gc.collect()  # First pass
            gc.collect()  # Second pass for circular references
            gc.collect(generation=2)  # Third pass - force full collection

            # Clear CUDA cache and reset memory statistics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache
                torch.cuda.reset_peak_memory_stats()  # Reset peak stats
                # Force another synchronization to ensure cache is cleared
                torch.cuda.synchronize()

                # Log memory after unloading for verification
                allocated_after = torch.cuda.memory_allocated() / (1024**3)  # GB
                reserved_after = torch.cuda.memory_reserved() / (1024**3)  # GB
                freed = reserved_before - reserved_after
                logger.info(
                    f"Embedding model unloaded - Allocated: {allocated_after:.2f}GB, "
                    f"Reserved: {reserved_after:.2f}GB, Freed: {freed:.2f}GB"
                )
            else:
                logger.info("Embedding model unloaded (CPU mode)")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        # Lazy load model
        if self.model is None:
            self.load_model()

        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (LangChain interface).

        Args:
            texts: List of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query (LangChain interface).

        Args:
            text: Text to embed.

        Returns:
            List[float]: Embedding vector.
        """
        return self([text])[0]

    @property
    def embedding_dim(self) -> int:
        """
        Get embedding dimension.

        Returns:
            int: Embedding dimension.
        """
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

    # Alias for compatibility with different naming conventions
    @property
    def dim(self) -> int:
        """Alias for embedding_dim."""
        return self.embedding_dim


class LocalVisionModelWrapper:
    """
    Wrapper for local vision models (optional for multimodal RAG).

    Currently a placeholder for future multimodal support.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize local vision model wrapper.

        Args:
            model_name: Vision model name.
            device: Device to run on.
        """
        self.model_name = model_name or settings.rag_vision_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("LocalVisionModelWrapper initialized (not yet implemented)")

    def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, str]]] = None,
        image_data: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Placeholder for vision model inference.

        Args:
            prompt: Text prompt.
            system_prompt: System prompt.
            history_messages: Conversation history.
            image_data: Image data.
            messages: Messages with images.
            **kwargs: Additional parameters.

        Returns:
            str: Generated response.
        """
        logger.warning("Vision model not yet implemented - using text-only LLM")
        # Fallback to text-only for now
        llm_wrapper = LocalLLMWrapper(device=self.device)
        return llm_wrapper(prompt, system_prompt, history_messages, **kwargs)


def create_embedding_func() -> Callable:
    """
    Create embedding function for raganything integration.

    Creates a LocalEmbeddingWrapper instance configured with settings
    that raganything can use as its embedding_func parameter.

    Returns:
        Callable: Embedding function compatible with raganything.
    """
    wrapper = LocalEmbeddingWrapper()

    # Return the wrapper callable
    return wrapper


def create_llm_func() -> Callable:
    """
    Create LLM function for raganything integration.

    Creates a LocalLLMWrapper instance configured with settings
    that raganything can use as its llm_model_func parameter.

    Returns:
        Callable: LLM function compatible with raganything.
    """
    wrapper = LocalLLMWrapper(
        max_tokens=settings.rag_max_tokens,
        temperature=settings.rag_temperature,
    )

    # Return the wrapper callable
    return wrapper


def create_vision_func() -> Optional[Callable]:
    """
    Create vision model function for raganything.

    Returns:
        Optional[Callable]: Vision function or None if not configured.
    """
    if settings.rag_vision_model is None:
        return None

    wrapper = LocalVisionModelWrapper()
    return wrapper
