"""Multi-provider AI abstraction layer with intelligent fallback support."""

import os
import asyncio
import json
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
from loguru import logger

# Provider options
class AIProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"


@dataclass
class AIConfig:
    """Configuration for AI providers."""
    primary_provider: AIProvider = AIProvider.OPENAI
    fallback_chain: List[AIProvider] = None
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 2.0
    
    def __post_init__(self):
        if self.fallback_chain is None:
            # Default fallback chain
            self.fallback_chain = [
                AIProvider.OPENAI,
                AIProvider.ANTHROPIC,
                AIProvider.GOOGLE,
                AIProvider.DEEPSEEK,
            ]


@dataclass
class ProviderCredentials:
    """Store provider API credentials."""
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    google_key: Optional[str] = None
    deepseek_key: Optional[str] = None
    
    @classmethod
    def from_environment(cls) -> 'ProviderCredentials':
        """Load credentials from environment variables."""
        return cls(
            openai_key=os.getenv("OPENAI_API_KEY"),
            anthropic_key=os.getenv("ANTHROPIC_API_KEY"),
            google_key=os.getenv("GOOGLE_API_KEY"),
            deepseek_key=os.getenv("DEEPSEEK_API_KEY"),
        )
    
    def get_key(self, provider: AIProvider) -> Optional[str]:
        """Get API key for specific provider."""
        keys = {
            AIProvider.OPENAI: self.openai_key,
            AIProvider.ANTHROPIC: self.anthropic_key,
            AIProvider.GOOGLE: self.google_key,
            AIProvider.DEEPSEEK: self.deepseek_key,
        }
        return keys.get(provider)


class AIProviderClient:
    """Base class for AI provider implementations."""
    
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def complete(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
    
    async def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError


class OpenAIClient(AIProviderClient):
    """OpenAI GPT-4 implementation."""
    
    async def complete(self, prompt: str, model: str = "gpt-4-mini", temperature: float = 0.7, **kwargs) -> str:
        """Generate completion using OpenAI API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": kwargs.get("max_tokens", 1024),
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"OpenAI API error: {error_data}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise
    
    async def embeddings(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Generate embeddings using OpenAI."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model,
                "input": text,
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"OpenAI API error: {error_data}")
                
                data = await response.json()
                return data["data"][0]["embedding"]
        
        except Exception as e:
            logger.error(f"OpenAI embeddings failed: {e}")
            raise


class AnthropicClient(AIProviderClient):
    """Anthropic Claude implementation."""
    
    async def complete(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.7, **kwargs) -> str:
        """Generate completion using Anthropic API."""
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model,
                "max_tokens": kwargs.get("max_tokens", 1024),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"Anthropic API error: {error_data}")
                
                data = await response.json()
                return data["content"][0]["text"]
        
        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise


class GoogleClient(AIProviderClient):
    """Google Gemini implementation."""
    
    async def complete(self, prompt: str, model: str = "gemini-2.0-flash", temperature: float = 0.7, **kwargs) -> str:
        """Generate completion using Google Gemini API."""
        try:
            headers = {
                "Content-Type": "application/json",
            }
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": kwargs.get("max_tokens", 1024),
                },
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"Google API error: {error_data}")
                
                data = await response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        
        except Exception as e:
            logger.error(f"Google completion failed: {e}")
            raise


class DeepSeekClient(AIProviderClient):
    """DeepSeek implementation (cost-effective fallback)."""
    
    async def complete(self, prompt: str, model: str = "deepseek-chat", temperature: float = 0.7, **kwargs) -> str:
        """Generate completion using DeepSeek API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": kwargs.get("max_tokens", 1024),
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"DeepSeek API error: {error_data}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"DeepSeek completion failed: {e}")
            raise


class AIOrchestrator:
    """Main orchestrator handling multi-provider AI with intelligent fallback."""
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or AIConfig()
        self.credentials = ProviderCredentials.from_environment()
        self.provider_map = {
            AIProvider.OPENAI: OpenAIClient,
            AIProvider.ANTHROPIC: AnthropicClient,
            AIProvider.GOOGLE: GoogleClient,
            AIProvider.DEEPSEEK: DeepSeekClient,
        }
        self.provider_health: Dict[AIProvider, Dict[str, Any]] = {}
        self._initialize_health()
    
    def _initialize_health(self):
        """Initialize provider health tracking."""
        for provider in AIProvider:
            self.provider_health[provider] = {
                "available": self.credentials.get_key(provider) is not None,
                "failures": 0,
                "last_failure": None,
                "success_count": 0,
            }
    
    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion with intelligent fallback."""
        
        # Try each provider in fallback chain
        for provider in self.config.fallback_chain:
            api_key = self.credentials.get_key(provider)
            
            if not api_key:
                logger.debug(f"Skipping {provider.value} - no API key")
                continue
            
            if not self.provider_health[provider]["available"]:
                logger.debug(f"Skipping {provider.value} - marked unavailable")
                continue
            
            try:
                logger.info(f"Attempting completion with {provider.value}")
                
                # Select appropriate client and model
                client_class = self.provider_map[provider]
                model = model or self._get_default_model(provider)
                
                client = client_class(api_key, timeout=self.config.timeout)
                async with client:
                    result = await client.complete(prompt, model=model, temperature=temperature, **kwargs)
                
                # Record success
                self.provider_health[provider]["failures"] = 0
                self.provider_health[provider]["success_count"] += 1
                
                logger.info(f"Completion successful with {provider.value}")
                return result
            
            except Exception as e:
                logger.warning(f"Completion failed with {provider.value}: {e}")
                self.provider_health[provider]["failures"] += 1
                self.provider_health[provider]["last_failure"] = datetime.now()
                
                # Mark unavailable if too many failures
                if self.provider_health[provider]["failures"] >= self.config.max_retries:
                    self.provider_health[provider]["available"] = False
                    logger.error(f"Marking {provider.value} as unavailable after {self.config.max_retries} failures")
                
                continue
        
        # All providers failed
        logger.error("All AI providers failed - returning fallback response")
        raise Exception("All AI providers exhausted - unable to generate completion")
    
    async def embeddings(self, text: str, **kwargs) -> List[float]:
        """Generate embeddings - prefer OpenAI for consistency."""
        api_key = self.credentials.get_key(AIProvider.OPENAI)
        
        if not api_key:
            logger.error("OpenAI API key not available for embeddings")
            raise Exception("OpenAI API key required for embeddings")
        
        try:
            client = OpenAIClient(api_key, timeout=self.config.timeout)
            async with client:
                return await client.embeddings(text, **kwargs)
        except Exception as e:
            logger.error(f"Embeddings generation failed: {e}")
            raise
    
    def _get_default_model(self, provider: AIProvider) -> str:
        """Get default model for provider."""
        models = {
            AIProvider.OPENAI: "gpt-4-mini",
            AIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            AIProvider.GOOGLE: "gemini-2.0-flash",
            AIProvider.DEEPSEEK: "deepseek-chat",
        }
        return models.get(provider, "unknown")
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get current status of all providers."""
        return {
            provider.value: {
                "available": health["available"],
                "failures": health["failures"],
                "success_count": health["success_count"],
                "last_failure": health["last_failure"].isoformat() if health["last_failure"] else None,
            }
            for provider, health in self.provider_health.items()
        }


# Global orchestrator instance
_orchestrator: Optional[AIOrchestrator] = None


async def get_orchestrator() -> AIOrchestrator:
    """Get or create global AI orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AIOrchestrator()
    return _orchestrator


async def complete(prompt: str, **kwargs) -> str:
    """Convenience function for text completion."""
    orchestrator = await get_orchestrator()
    return await orchestrator.complete(prompt, **kwargs)


async def embeddings(text: str, **kwargs) -> List[float]:
    """Convenience function for embeddings."""
    orchestrator = await get_orchestrator()
    return await orchestrator.embeddings(text, **kwargs)
