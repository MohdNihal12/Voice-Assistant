import json
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel

class ServerConfig(BaseModel):
    port: int = 8004
    host: str = "0.0.0.0"
    cors_origins: list = ["*"]

class DeepgramConfig(BaseModel):
    model: str = "nova-3"
    sample_rate: int = 16000
    language: Optional[str] = None
    enable_multilingual: bool = True
    verbose: bool = False
    chunk_size: int = 1024
    endpointing: int = 100
    utterance_end_ms: int = 3000

class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    timeout: int = 15
    max_tokens: int = 500
    temperature: float = 0.7

class TTSConfig(BaseModel):
    voice_id: str = "iP95p4xoKVk53GoZ742B"
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "mp3_44100_128"
    timeout: int = 10

class ProductSearchConfig(BaseModel):
    product_file: str = "data/product.json"
    top_k: int = 3
    similarity_threshold: float = 0.3

class DatabaseConfig(BaseModel):
    enabled: bool = True
    max_connections: int = 10

class WebSocketConfig(BaseModel):
    ping_interval: int = 30
    receive_timeout: int = 30
    audio_chunk_delay: float = 0.01

class FeaturesConfig(BaseModel):
    enable_tts: bool = True
    enable_product_search: bool = True
    enable_database: bool = True
    enable_multilingual: bool = True

class AppConfig(BaseModel):
    server: ServerConfig
    deepgram: DeepgramConfig
    llm: LLMConfig
    tts: TTSConfig
    product_search: ProductSearchConfig
    database: DatabaseConfig
    websocket: WebSocketConfig
    features: FeaturesConfig

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config: Optional[AppConfig] = None
        self.load_config()
    
    def load_config(self) -> AppConfig:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                # Create default config if file doesn't exist
                config_data = self._get_default_config()
                self.save_config(config_data)
            
            # Convert to Pydantic model
            self.config = AppConfig(**config_data)
            print(f"✅ Configuration loaded from {self.config_file}")
            return self.config
            
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            # Fallback to default config
            self.config = AppConfig(**self._get_default_config())
            return self.config
    
    def save_config(self, config_data: Optional[Dict] = None) -> bool:
        """Save configuration to JSON file"""
        try:
            if config_data is None and self.config:
                config_data = self.config.dict()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        if self.config is None:
            self.load_config()
        return self.config
    
    def update_config(self, new_config: Dict) -> bool:
        """Update configuration with new values"""
        try:
            current_config = self.get_config().dict()
            
            # Deep merge the configurations
            merged_config = self._deep_merge(current_config, new_config)
            
            # Validate the merged config
            self.config = AppConfig(**merged_config)
            
            # Save to file
            return self.save_config(merged_config)
            
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            return False
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "server": {
                "port": 8004,
                "host": "0.0.0.0",
                "cors_origins": ["*"]
            },
            "deepgram": {
                "model": "nova-3",
                "sample_rate": 16000,
                "language": None,
                "enable_multilingual": True,
                "verbose": False,
                "chunk_size": 1024,
                "endpointing": 100,
                "utterance_end_ms": 3000
            },
            "llm": {
                "model": "gpt-4o-mini",
                "timeout": 15,
                "max_tokens": 500,
                "temperature": 0.7
            },
            "tts": {
                "voice_id": "iP95p4xoKVk53GoZ742B",
                "model_id": "eleven_multilingual_v2",
                "output_format": "mp3_44100_128",
                "timeout": 10
            },
            "product_search": {
                "product_file": "data/product.json",
                "top_k": 3,
                "similarity_threshold": 0.3
            },
            "database": {
                "enabled": True,
                "max_connections": 10
            },
            "websocket": {
                "ping_interval": 30,
                "receive_timeout": 30,
                "audio_chunk_delay": 0.01
            },
            "features": {
                "enable_tts": True,
                "enable_product_search": True,
                "enable_database": True,
                "enable_multilingual": True
            }
        }
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in update.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

# Global config instance
config_manager = ConfigManager()