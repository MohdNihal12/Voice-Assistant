# app/tts.py - IMPROVED VERSION WITH BETTER INTERRUPTION CONTROL
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

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    """
    ElevenLabs Text-to-Speech with robust audio interruption control
    """
    
    def __init__(self, 
                 voice_id: str = "iP95p4xoKVk53GoZ742B",
                 model_id: str = "eleven_multilingual_v2",
                 output_format: str = "mp3_44100_128"):
        
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.client = None
        
        # Audio playback control with thread-safe operations
        self._is_playing = False
        self._stop_playback = False
        self._current_stream = None
        self._pyaudio = None
        self._playback_lock = threading.RLock()  # Use RLock for reentrant locks
        self._playback_thread = None
        
        self._initialize_client_and_audio()
    
    def _initialize_client_and_audio(self):
        """Initialize the ElevenLabs client and PyAudio"""
        try:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if not api_key:
                logger.error("ELEVENLABS_API_KEY not found in environment variables")
                raise ValueError("ELEVENLABS_API_KEY not found in environment variables")
            
            self.client = ElevenLabs(api_key=api_key)
            
            # Initialize PyAudio
            self._pyaudio = pyaudio.PyAudio()
            
            logger.info(f"ElevenLabs TTS initialized with PyAudio control")
            logger.info(f"Voice ID: {self.voice_id}, Model: {self.model_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
            self.client = None
    
    @property
    def is_playing(self):
        """Thread-safe access to playing state"""
        with self._playback_lock:
            return self._is_playing
    
    def stop_current_audio(self):
        """Force stop the currently playing audio - SYNCHRONOUS VERSION"""
        with self._playback_lock:
            if self._is_playing:
                logger.info("🛑 Force stopping current audio playback")
                self._stop_playback = True
                
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
        if not self.client:
            logger.error("ElevenLabs client not initialized")
            return None
        
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        
        try:
            logger.info(f"Converting text to speech: '{text[:50]}...'")
            
            # Stop current playback if requested - DO THIS FIRST
            if interrupt_current:
                # Call synchronous stop method
                await asyncio.get_event_loop().run_in_executor(None, self.stop_current_audio)
            
            # Generate audio data
            audio_data = await self._generate_audio_data(text)
            
            if audio_data and play_audio:
                await self._play_audio_async(audio_data)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS conversion failed: {e}")
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
            
            logger.info(f"TTS conversion successful - {len(audio_data)} bytes")
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None
    
    async def _play_audio_async(self, audio_data: bytes):
        """Play audio data asynchronously with interruption support"""
        if not audio_data:
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
            logger.error(f"Audio playback failed: {e}")
    
    def _play_audio_sync(self, wav_data: bytes):
        """Synchronous audio playback with real-time interruption checking"""
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
                logger.info("Playback completed successfully")
                
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
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
            logger.error(f"MP3 to WAV conversion failed: {e}")
            return None
    
    async def generate_and_speak(self, llm_response: Dict[str, Any], interrupt_current: bool = True):
        """
        Generate speech from LLM response text
        
        Args:
            llm_response: Response dictionary from LLM
            interrupt_current: Whether to interrupt current playback
        """
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
        return {
            'voice_id': self.voice_id,
            'model_id': self.model_id,
            'output_format': self.output_format,
            'is_playing': self.is_playing,
            'client_initialized': self.client is not None,
            'pyaudio_initialized': self._pyaudio is not None
        }
    
    def cleanup(self):
        """Cleanup resources - call this when shutting down"""
        self.stop_current_audio()
        if self._pyaudio:
            self._pyaudio.terminate()
    
    def __del__(self):
        """Cleanup resources"""
        self.cleanup()