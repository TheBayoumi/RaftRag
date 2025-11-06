"""
Resource Detection Utility.

Intelligently detects available GPU/CPU resources and provides
recommendations for optimal model loading and training.
"""

from typing import Any, Dict, List, Optional

import torch
from loguru import logger


class ResourceDetector:
    """
    Intelligent GPU/CPU resource detection for optimal performance.

    Detects:
    - Available CUDA devices
    - GPU memory (total and available)
    - CPU cores and RAM
    - Recommends optimal settings for model loading
    """

    @staticmethod
    def detect_available_resources() -> Dict[str, Any]:
        """
        Detect and return comprehensive resource information.

        Returns:
            Dict with resource details:
            - device: "cuda" or "cpu"
            - gpu_count: Number of GPUs
            - gpu_names: List of GPU names
            - total_vram_gb: Total VRAM per GPU
            - available_vram_gb: Available VRAM
            - recommended_batch_size: Optimal batch size
            - recommended_model: Suggested model size
            - use_4bit_quantization: Whether to use 4-bit
            - max_model_memory: Memory allocation dict
        """
        resources = {
            "device": "cpu",
            "gpu_count": 0,
            "gpu_names": [],
            "total_vram_gb": 0.0,
            "available_vram_gb": 0.0,
            "recommended_batch_size": 1,
            "recommended_model": "mistralai/Mistral-7B-Instruct-v0.2",
            "use_4bit_quantization": True,
            "use_8bit_quantization": False,
            "max_model_memory": {},
            "cpu_offload_recommended": False,
        }

        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("No CUDA devices available, using CPU mode")
            resources["device"] = "cpu"
            resources["recommended_batch_size"] = 1
            resources["recommended_model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            return resources

        # GPU detected
        resources["device"] = "cuda"
        resources["gpu_count"] = torch.cuda.device_count()

        # Get GPU details
        for i in range(resources["gpu_count"]):
            try:
                gpu_name = torch.cuda.get_device_name(i)
                resources["gpu_names"].append(gpu_name)

                # Get memory info (in GB)
                props = torch.cuda.get_device_properties(i)
                total_memory_gb = props.total_memory / (1024**3)
                resources["total_vram_gb"] = total_memory_gb

                # Get available memory
                torch.cuda.reset_peak_memory_stats(i)
                torch.cuda.empty_cache()
                available_memory_gb = (
                    torch.cuda.get_device_properties(i).total_memory
                    - torch.cuda.memory_allocated(i)
                ) / (1024**3)
                resources["available_vram_gb"] = available_memory_gb

                logger.info(
                    f"GPU {i}: {gpu_name} - "
                    f"Total: {total_memory_gb:.1f}GB, "
                    f"Available: {available_memory_gb:.1f}GB"
                )

            except Exception as e:
                logger.warning(f"Could not get GPU {i} details: {e}")

        # Make recommendations based on available VRAM
        vram = resources["available_vram_gb"]

        if vram >= 20:
            # High-end GPU (RTX 3090, A100, etc.)
            resources["recommended_batch_size"] = 8
            resources["recommended_model"] = "meta-llama/Llama-3-8B"
            resources["use_4bit_quantization"] = False
            resources["max_model_memory"] = {0: f"{vram-4:.1f}GB", "cpu": "20GB"}
            logger.info("High-end GPU detected - can handle large models")

        elif vram >= 12:
            # Mid-range GPU (RTX 3080, RTX 4070, etc.)
            resources["recommended_batch_size"] = 4
            resources["recommended_model"] = "mistralai/Mistral-7B-Instruct-v0.2"
            resources["use_4bit_quantization"] = True
            resources["max_model_memory"] = {0: f"{vram-2:.1f}GB", "cpu": "16GB"}
            logger.info("Mid-range GPU detected - use 4-bit quantization")

        elif vram >= 8:
            # Entry-level GPU (RTX 3060, etc.)
            resources["recommended_batch_size"] = 2
            resources["recommended_model"] = "mistralai/Mistral-7B-Instruct-v0.2"
            resources["use_4bit_quantization"] = True
            resources["cpu_offload_recommended"] = True
            resources["max_model_memory"] = {0: f"{vram-1:.1f}GB", "cpu": "12GB"}
            logger.warning("Limited VRAM - use 4-bit + CPU offload")

        elif vram >= 4:
            # Very limited GPU (user's 6GB GTX 1060 falls here)
            resources["recommended_batch_size"] = 1
            resources["recommended_model"] = "meta-llama/Llama-3.2-3B"
            resources["use_4bit_quantization"] = True
            resources["use_8bit_quantization"] = False
            resources["cpu_offload_recommended"] = True
            resources["max_model_memory"] = {0: f"{vram-0.5:.1f}GB", "cpu": "8GB"}
            logger.warning("Very limited VRAM - use smaller model (3B)")

        else:
            # Insufficient VRAM for training
            resources["recommended_batch_size"] = 1
            resources["recommended_model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            resources["use_4bit_quantization"] = True
            resources["cpu_offload_recommended"] = True
            resources["max_model_memory"] = {"cpu": "8GB"}
            logger.error(
                f"Insufficient VRAM ({vram:.1f}GB) - use tiny model or CPU mode"
            )

        return resources

    @staticmethod
    def get_optimal_model_config(
        model_name: str, resources: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get optimal model loading configuration for given resources.

        Args:
            model_name: Model to load.
            resources: Resource dict from detect_available_resources().

        Returns:
            Dict with optimal loading config.
        """
        if resources is None:
            resources = ResourceDetector.detect_available_resources()

        config = {
            "device_map": "auto",
            "dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "max_memory": resources["max_model_memory"],
        }

        # Add quantization if recommended
        if resources["use_4bit_quantization"] and resources["device"] == "cuda":
            from transformers import BitsAndBytesConfig

            config["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            config["device_map"] = "auto"
            logger.info("4-bit quantization enabled for model loading")

        elif resources["use_8bit_quantization"] and resources["device"] == "cuda":
            from transformers import BitsAndBytesConfig

            config["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            logger.info("8-bit quantization enabled for model loading")

        # CPU-only mode
        if resources["device"] == "cpu":
            config["device_map"] = None
            config["dtype"] = torch.float32  # CPU needs float32
            del config["max_memory"]
            logger.warning("CPU-only mode - training will be very slow")

        return config

    @staticmethod
    def validate_model_fits(
        model: Any, resources: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Validate that loaded model fits in available memory.

        Args:
            model: Loaded model to validate.
            resources: Resource dict.

        Returns:
            bool: True if model fits, False otherwise.
        """
        if resources is None:
            resources = ResourceDetector.detect_available_resources()

        # Check for meta tensors (model didn't load properly)
        if hasattr(model, "hf_device_map"):
            meta_modules = [k for k, v in model.hf_device_map.items() if v == "meta"]
            if meta_modules:
                logger.error(
                    f"Model has {len(meta_modules)} modules on 'meta' device - "
                    "model is too large for available VRAM!"
                )
                logger.error(f"Meta modules: {meta_modules[:5]}...")
                return False

        # Check GPU memory usage
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
            available_gb = resources.get("available_vram_gb", 0)

            logger.info(
                f"GPU memory: Allocated={allocated_gb:.2f}GB, "
                f"Reserved={reserved_gb:.2f}GB, "
                f"Available={available_gb:.2f}GB"
            )

            if allocated_gb > available_gb * 0.95:
                logger.warning(
                    "Model using >95% of available VRAM - may cause OOM errors"
                )
                return False

        logger.success("Model validation passed - fits in available memory")
        return True

    @staticmethod
    def get_training_recommendations(
        resources: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get training hyperparameter recommendations.

        Args:
            resources: Resource dict.

        Returns:
            Dict with recommended training settings.
        """
        if resources is None:
            resources = ResourceDetector.detect_available_resources()

        recommendations = {
            "batch_size": resources["recommended_batch_size"],
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "warmup_steps": 100,
            "max_grad_norm": 0.3,
            "weight_decay": 0.001,
            "fp16": resources["device"] == "cuda",
            "dataloader_num_workers": 2 if resources["device"] == "cuda" else 0,
            "dataloader_pin_memory": resources["device"] == "cuda",
            "gradient_checkpointing": False,  # Disabled for performance
        }

        # Adjust based on VRAM
        vram = resources["available_vram_gb"]

        if vram < 6:
            recommendations["gradient_accumulation_steps"] = 8  # More accumulation
            recommendations["batch_size"] = 1
            recommendations["dataloader_num_workers"] = 0
            logger.warning("Low VRAM - increased gradient accumulation")

        elif vram >= 20:
            recommendations["batch_size"] = 8
            recommendations["gradient_accumulation_steps"] = 2
            logger.info("High VRAM - increased batch size")

        return recommendations

    @staticmethod
    def print_resource_summary(resources: Optional[Dict[str, Any]] = None) -> None:
        """
        Print formatted resource summary.

        Args:
            resources: Resource dict to print.
        """
        if resources is None:
            resources = ResourceDetector.detect_available_resources()

        print("\n" + "=" * 60)
        print("[INFO] SYSTEM RESOURCES DETECTED")
        print("=" * 60)
        print(f"Device: {resources['device'].upper()}")

        if resources["device"] == "cuda":
            print(f"GPU Count: {resources['gpu_count']}")
            for i, name in enumerate(resources["gpu_names"]):
                print(f"  GPU {i}: {name}")
            print(f"Total VRAM: {resources['total_vram_gb']:.1f} GB")
            print(f"Available VRAM: {resources['available_vram_gb']:.1f} GB")
            print(
                f"4-bit Quantization: "
                f"{'[ENABLED]' if resources['use_4bit_quantization'] else '[DISABLED]'}"
            )
            print(
                f"CPU Offload: "
                f"{'[RECOMMENDED]' if resources['cpu_offload_recommended'] else '[NOT NEEDED]'}"
            )

        print(f"\n[INFO] RECOMMENDATIONS")
        print(f"Recommended Model: {resources['recommended_model']}")
        print(f"Recommended Batch Size: {resources['recommended_batch_size']}")
        print("=" * 60 + "\n")
