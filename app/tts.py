# app/tts.py
import os
import asyncio
import queue
import threading
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import io
import logging

# Load environment variables
load_dotenv()

class ElevenLabsTTS:
    """
    ElevenLabs Text-to-Speech integration for voice assistant responses
    """
    
    def __init__(self, 
                 voice_id: str = "A9ATTqUUQ6GHu0coCz8t",
                 model_id: str = "eleven_multilingual_v2",
                 output_format: str = "mp3_44100_128"):
        
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.client = None
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stop_playback = False
        self.playback_thread = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the ElevenLabs client"""
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                print("❌ ELEVENLABS_API_KEY not found in environment variables")
                raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
            
            self.client = ElevenLabs(api_key=api_key)
            print(f"✅ ElevenLabs TTS initialized")
            print(f"   Voice ID: {self.voice_id}")
            print(f"   Model: {self.model_id}")
            
            # Test the connection with a short text
            test_text = "Hello"
            try:
                audio = self.client.text_to_speech.convert(
                    text=test_text,
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    output_format=self.output_format,
                )
                print("✅ ElevenLabs connection test successful")
            except Exception as e:
                print(f"❌ ElevenLabs connection test failed: {e}")
                raise
                
        except Exception as e:
            print(f"❌ Failed to initialize ElevenLabs TTS: {e}")
            self.client = None
    
    async def text_to_speech(self, text: str, play_audio: bool = True) -> Optional[bytes]:
        """
        Convert text to speech using ElevenLabs
        
        Args:
            text: Text to convert to speech
            play_audio: Whether to play the audio immediately
            
        Returns:
            Audio data as bytes if successful, None otherwise
        """
        if not self.client:
            print("❌ ElevenLabs client not initialized")
            return None
        
        if not text or not text.strip():
            print("⚠️ Empty text provided for TTS")
            return None
        
        try:
            print(f"🔊 Converting text to speech: '{text[:50]}...'")
            
            # Convert text to speech - FIXED: Use proper async handling
            audio = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=self.output_format,
            )
            
            # Collect all audio data
            audio_data = b""
            for chunk in audio:
                if chunk:
                    audio_data += chunk
            
            print(f"✅ TTS conversion successful - {len(audio_data)} bytes")
            
            if play_audio:
                await self.play_audio(audio_data)
            
            return audio_data
            
        except Exception as e:
            print(f"❌ TTS conversion failed: {e}")
            return None
    
    async def play_audio(self, audio_data: bytes):
        """
        Play audio data
        
        Args:
            audio_data: Audio data to play
        """
        try:
            print("🔊 Playing audio...")
            play(audio_data)
            print("✅ Audio playback completed")
        except Exception as e:
            print(f"❌ Audio playback failed: {e}")
    
    def start_playback_thread(self):
        """Start the background playback thread"""
        if self.playback_thread and self.playback_thread.is_alive():
            return
        
        self.stop_playback = False
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        print("✅ Playback thread started")
    
    def stop_playback_thread(self):
        """Stop the background playback thread"""
        self.stop_playback = True
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=5)
        print("🛑 Playback thread stopped")
    
    def _playback_worker(self):
        """Background worker for playing audio from queue"""
        while not self.stop_playback:
            try:
                # Get audio data from queue with timeout
                audio_data = self.audio_queue.get(timeout=1.0)
                if audio_data is None:  # Sentinel value to stop
                    break
                
                self.is_playing = True
                play(audio_data)
                self.is_playing = False
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Playback worker error: {e}")
                self.is_playing = False
    
    def queue_audio(self, audio_data: bytes):
        """
        Queue audio data for playback in background thread
        
        Args:
            audio_data: Audio data to queue for playback
        """
        if audio_data:
            self.audio_queue.put(audio_data)
            print("✅ Audio queued for playback")
    
    def queue_text(self, text: str):
        """
        Convert text to speech and queue it for playback
        
        Args:
            text: Text to convert and queue
        """
        if not text or not text.strip():
            return
        
        # Run TTS in a thread and queue the result
        def tts_and_queue():
            try:
                audio = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    output_format=self.output_format,
                )
                # Collect all chunks
                audio_data = b""
                for chunk in audio:
                    if chunk:
                        audio_data += chunk
                self.queue_audio(audio_data)
            except Exception as e:
                print(f"❌ TTS queuing failed: {e}")
        
        threading.Thread(target=tts_and_queue, daemon=True).start()
    
    async def generate_and_speak(self, llm_response: Dict[str, Any]):
        """
        Generate speech from LLM response text
        
        Args:
            llm_response: Response dictionary from OllamaLLM
        """
        if not llm_response or not llm_response.get('success'):
            print("⚠️ Invalid LLM response for TTS")
            return
        
        text = llm_response.get('text', '').strip()
        if not text:
            print("⚠️ Empty text in LLM response")
            return
        
        await self.text_to_speech(text)
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about the current voice configuration"""
        return {
            'voice_id': self.voice_id,
            'model_id': self.model_id,
            'output_format': self.output_format,
            'queue_size': self.audio_queue.qsize(),
            'is_playing': self.is_playing,
            'client_initialized': self.client is not None
        }
    
    def clear_queue(self):
        """Clear the audio queue"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        print("🗑️ Audio queue cleared")

# Pre-configured voices for different use cases
VOICE_PRESETS = {
    "professional_male": "JBFqnCBsd6RMkjVDRZzb",  # Your current voice
    "professional_female": "Rachel",
    "friendly_male": "Adam",
    "friendly_female": "Dorothy",
    "multilingual": "Arnold"  # Good for multiple languages
}

def create_tts_with_preset(voice_preset: str = "professional_male") -> ElevenLabsTTS:
    """
    Create a TTS instance with a preset voice
    
    Args:
        voice_preset: One of the preset voice names
        
    Returns:
        Configured ElevenLabsTTS instance
    """
    voice_id = VOICE_PRESETS.get(voice_preset, VOICE_PRESETS["professional_male"])
    return ElevenLabsTTS(voice_id=voice_id)

# Example usage and test function
async def test_tts():
    """Test function for TTS functionality"""
    print("🧪 Testing ElevenLabs TTS...")
    
    tts = ElevenLabsTTS()
    
    if not tts.client:
        print("❌ TTS client not available - skipping test")
        return
    
    # Test direct TTS
    test_texts = [
        "Hello! This is the Hidayath Group sales assistant. How can I help you today?",
        "I can help you with product information, pricing, and availability.",
        "Thank you for contacting Hidayath Group!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔊 Test {i}: '{text}'")
        audio_data = await tts.text_to_speech(text, play_audio=False)
        if audio_data:
            print(f"✅ Test {i} successful - generated {len(audio_data)} bytes")
        else:
            print(f"❌ Test {i} failed")
        await asyncio.sleep(1)  # Brief pause between tests
    
    print("\n✅ TTS test completed successfully")

if __name__ == "__main__":
    # Run test if script is executed directly
    asyncio.run(test_tts())