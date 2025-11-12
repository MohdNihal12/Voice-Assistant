# app/tts.py - IMPROVED VERSION WITH BETTER INTERRUPTION CONTROL AND CONFIG INTEGRATION
import os
import asyncio
import threading
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import pyaudio
import wave
import tempfile
import io
from pydub import AudioSegment
import time
import logging

# NEW: Import config manager
from app.config_manager import config_manager

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    """
    ElevenLabs Text-to-Speech with robust audio interruption control and config integration
    """
    
    def __init__(self, 
                 voice_id: str = None,  # UPDATED: Use config by default
                 model_id: str = None,  # UPDATED: Use config by default
                 output_format: str = None):  # UPDATED: Use config by default
        
        # Get config values
        config = config_manager.get_config()
        
        # UPDATED: Use config values with fallbacks
        self.voice_id = voice_id or config.tts.voice_id
        self.model_id = model_id or config.tts.model_id
        self.output_format = output_format or config.tts.output_format
        self.timeout = config.tts.timeout
        self.client = None
        
        # Audio playback control with thread-safe operations
        self._is_playing = False
        self._stop_playback = False
        self._current_stream = None
        self._pyaudio = None
        self._playback_lock = threading.RLock()  # Use RLock for reentrant locks
        self._playback_thread = None
        
        # Statistics
        self.stats = {
            'tts_requests': 0,
            'audio_generated': 0,
            'playback_completed': 0,
            'playback_interrupted': 0,
            'errors': 0,
            'total_audio_duration': 0.0
        }
        
        self._initialize_client_and_audio()
    
    def _initialize_client_and_audio(self):
        """Initialize the ElevenLabs client and PyAudio"""
        try:
            # Check if TTS is enabled
            config = config_manager.get_config()
            if not config.features.enable_tts:
                logger.info("⏭️ TTS disabled in configuration")
                return
            
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                logger.error("ELEVENLABS_API_KEY not found in environment variables")
                raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
            
            self.client = ElevenLabs(api_key=api_key)
            
            # Initialize PyAudio
            self._pyaudio = pyaudio.PyAudio()
            
            logger.info(f"✅ ElevenLabs TTS initialized with PyAudio control")
            logger.info(f"⚙️ Config: Voice: {self.voice_id}, Model: {self.model_id}, Format: {self.output_format}")
            logger.info(f"⚙️ Timeout: {self.timeout}s, TTS Enabled: {config.features.enable_tts}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize ElevenLabs TTS: {e}")
            self.client = None
    
    @property
    def is_playing(self):
        """Thread-safe access to playing state"""
        with self._playback_lock:
            return self._is_playing
    
    def stop_current_audio(self):
        """Force stop the currently playing audio - SYNCHRONOUS VERSION"""
        config = config_manager.get_config()
        if not config.features.enable_tts:
            logger.debug("⏭️ TTS disabled - ignoring stop request")
            return
            
        with self._playback_lock:
            if self._is_playing:
                logger.info("🛑 Force stopping current audio playback")
                self._stop_playback = True
                self.stats['playback_interrupted'] += 1
                
                # Stop the PyAudio stream immediately
                if self._current_stream:
                    try:
                        self._current_stream.stop_stream()
                        self._current_stream.close()
                        logger.debug("PyAudio stream closed")
                    except Exception as e:
                        logger.warning(f"Error closing stream: {e}")
                    finally:
                        self._current_stream = None
                
                self._is_playing = False
                
                # Wait for playback thread to finish if it exists
                if self._playback_thread and self._playback_thread.is_alive():
                    logger.debug("Waiting for playback thread to terminate...")
                    self._playback_thread.join(timeout=1.0)  # Wait up to 1 second
                    if self._playback_thread.is_alive():
                        logger.warning("Playback thread did not terminate cleanly")
                
                self._playback_thread = None
        
        # Small delay to ensure cleanup
        time.sleep(0.05)
        logger.info("✅ Audio playback stopped successfully")
    
    async def text_to_speech(self, text: str, play_audio: bool = True, interrupt_current: bool = True) -> Optional[bytes]:
        """
        Convert text to speech with robust interruption control
        
        Args:
            text: Text to convert to speech
            play_audio: Whether to play the audio immediately
            interrupt_current: Whether to interrupt currently playing audio
            
        Returns:
            Audio data as bytes if successful, None otherwise
        """
        # Check if TTS is enabled
        config = config_manager.get_config()
        if not config.features.enable_tts:
            logger.info("⏭️ TTS disabled - skipping audio generation")
            return None
            
        if not self.client:
            logger.error("ElevenLabs client not initialized")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        try:
            logger.info(f"Converting text to speech: '{text[:50]}...'")
            self.stats['tts_requests'] += 1
            
            # Stop current playback if requested - DO THIS FIRST
            if interrupt_current:
                # Call synchronous stop method
                await asyncio.get_event_loop().run_in_executor(None, self.stop_current_audio)
            
            # Generate audio data with timeout from config
            audio_data = await asyncio.wait_for(
                self._generate_audio_data(text),
                timeout=self.timeout
            )
            
            if audio_data:
                self.stats['audio_generated'] += 1
                if play_audio:
                    await self._play_audio_async(audio_data)
            
            return audio_data
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ TTS generation timed out after {self.timeout} seconds")
            self.stats['errors'] += 1
            return None
        except Exception as e:
            logger.error(f"❌ TTS conversion failed: {e}")
            self.stats['errors'] += 1
            import traceback
            traceback.print_exc()
            return None
    
    async def _generate_audio_data(self, text: str) -> Optional[bytes]:
        """Generate audio data from text using ElevenLabs API"""
        try:
            loop = asyncio.get_event_loop()
            audio_generator = await loop.run_in_executor(
                None,
                lambda: self.client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    output_format=self.output_format,
                )
            )
            
            # Collect all audio data
            audio_data = b""
            for chunk in audio_generator:
                if chunk:
                    audio_data += chunk
            
            logger.info(f"✅ TTS conversion successful - {len(audio_data)} bytes")
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Audio generation failed: {e}")
            return None
    
    async def _play_audio_async(self, audio_data: bytes):
        """Play audio data asynchronously with interruption support"""
        if not audio_data:
            return
        
        # Check if TTS is enabled
        config = config_manager.get_config()
        if not config.features.enable_tts:
            logger.debug("⏭️ TTS disabled - skipping audio playback")
            return
        
        try:
            logger.info("Playing audio with interruption control...")
            
            # Convert MP3 to WAV for PyAudio playback
            wav_data = await self._convert_mp3_to_wav(audio_data)
            if not wav_data:
                logger.error("Failed to convert audio to WAV format")
                return
            
            # Use thread pool executor for synchronous playback
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self._play_audio_sync(wav_data)
            )
                
        except Exception as e:
            logger.error(f"❌ Audio playback failed: {e}")
            self.stats['errors'] += 1
    
    def _play_audio_sync(self, wav_data: bytes):
        """Synchronous audio playback with real-time interruption checking"""
        # Check if TTS is enabled
        config = config_manager.get_config()
        if not config.features.enable_tts:
            return
            
        with self._playback_lock:
            if self._stop_playback:
                logger.info("Playback aborted before starting")
                return
                
            self._is_playing = True
            self._stop_playback = False
        
        try:
            # Write WAV data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(wav_data)
                tmp_file.flush()
                temp_filename = tmp_file.name
            
            # Open the WAV file
            with wave.open(temp_filename, 'rb') as wf:
                # Get audio parameters
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                frame_rate = wf.getframerate()
                frames_per_buffer = 1024
                
                # Calculate audio duration for stats
                n_frames = wf.getnframes()
                duration = n_frames / float(frame_rate)
                self.stats['total_audio_duration'] += duration
                
                # Open PyAudio stream
                stream = self._pyaudio.open(
                    format=self._pyaudio.get_format_from_width(sample_width),
                    channels=channels,
                    rate=frame_rate,
                    output=True,
                    frames_per_buffer=frames_per_buffer
                )
                
                with self._playback_lock:
                    self._current_stream = stream
                
                # Play audio in chunks with interruption checking
                data = wf.readframes(frames_per_buffer)
                while data and not self._stop_playback:
                    stream.write(data)
                    data = wf.readframes(frames_per_buffer)
                
                # Clean up stream
                stream.stop_stream()
                stream.close()
                
                with self._playback_lock:
                    self._current_stream = None
            
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")
            
            if self._stop_playback:
                logger.info("Playback was interrupted")
            else:
                logger.info("✅ Playback completed successfully")
                self.stats['playback_completed'] += 1
                
        except Exception as e:
            logger.error(f"❌ Audio playback error: {e}")
            self.stats['errors'] += 1
        finally:
            with self._playback_lock:
                self._is_playing = False
    
    async def _convert_mp3_to_wav(self, mp3_data: bytes) -> Optional[bytes]:
        """Convert MP3 data to WAV format for PyAudio playback"""
        try:
            # Create AudioSegment from MP3 data
            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Convert to WAV
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_data = wav_io.getvalue()
            
            return wav_data
            
        except Exception as e:
            logger.error(f"❌ MP3 to WAV conversion failed: {e}")
            return None
    
    async def generate_and_speak(self, llm_response: Dict[str, Any], interrupt_current: bool = True):
        """
        Generate speech from LLM response text
        
        Args:
            llm_response: Response dictionary from LLM
            interrupt_current: Whether to interrupt current playback
        """
        # Check if TTS is enabled
        config = config_manager.get_config()
        if not config.features.enable_tts:
            logger.debug("⏭️ TTS disabled - skipping speech generation")
            return
            
        if not llm_response or not llm_response.get('success'):
            logger.warning("Invalid LLM response for TTS")
            return
        
        text = llm_response.get('text', '').strip()
        if not text:
            logger.warning("Empty text in LLM response")
            return
        
        await self.text_to_speech(text, interrupt_current=interrupt_current)
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about the current voice configuration"""
        config = config_manager.get_config()
        
        return {
            'voice_id': self.voice_id,
            'model_id': self.model_id,
            'output_format': self.output_format,
            'timeout': self.timeout,
            'is_playing': self.is_playing,
            'client_initialized': self.client is not None,
            'pyaudio_initialized': self._pyaudio is not None,
            'tts_enabled': config.features.enable_tts,
            'config_source': 'config.json'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS system statistics"""
        config = config_manager.get_config()
        
        stats = {
            'tts_requests': self.stats['tts_requests'],
            'audio_generated': self.stats['audio_generated'],
            'playback_completed': self.stats['playback_completed'],
            'playback_interrupted': self.stats['playback_interrupted'],
            'errors': self.stats['errors'],
            'total_audio_duration': round(self.stats['total_audio_duration'], 2),
            'success_rate': f"{(self.stats['audio_generated'] / self.stats['tts_requests'] * 100) if self.stats['tts_requests'] > 0 else 0:.1f}%",
            'tts_enabled': config.features.enable_tts,
            'config': {
                'voice_id': self.voice_id,
                'model_id': self.model_id,
                'output_format': self.output_format,
                'timeout': self.timeout
            }
        }
        
        return stats
    
    async def test_connection(self) -> bool:
        """Test connection to ElevenLabs API"""
        config = config_manager.get_config()
        if not config.features.enable_tts:
            logger.info("⏭️ TTS disabled - connection test skipped")
            return False
            
        if not self.client:
            return False
        
        try:
            # Try to get voice info as a connection test
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.voices.get(self.voice_id)
            )
            return True
        except Exception as e:
            logger.warning(f"⚠️ ElevenLabs connection test failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources - call this when shutting down"""
        config = config_manager.get_config()
        if not config.features.enable_tts:
            return
            
        self.stop_current_audio()
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
                logger.info("✅ PyAudio terminated")
            except Exception as e:
                logger.warning(f"⚠️ Error terminating PyAudio: {e}")
    
    def __del__(self):
        """Cleanup resources"""
        self.cleanup()


# Example usage and testing
async def main():
    """Test the TTS system with configuration"""
    tts = ElevenLabsTTS()
    
    # Test connection
    print("Testing ElevenLabs connection...")
    if await tts.test_connection():
        print("✅ ElevenLabs connection successful!")
    else:
        print("❌ ElevenLabs connection failed!")
        return
    
    # Test TTS with different texts
    test_texts = [
        "Hello! This is a test of the text to speech system.",
        "How can I help you with steel and aluminum products today?",
        "We have stainless steel 304 and aluminum 5052 in stock."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🧪 Test {i}: '{text}'")
        audio_data = await tts.text_to_speech(text, play_audio=True, interrupt_current=True)
        
        if audio_data:
            print(f"✅ TTS successful - {len(audio_data)} bytes")
        else:
            print("❌ TTS failed")
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Print statistics
    print(f"\n📊 TTS Stats: {tts.get_stats()}")
    print(f"🔊 Voice Info: {tts.get_voice_info()}")
    
    # Cleanup
    tts.cleanup()


if __name__ == "__main__":
    asyncio.run(main())