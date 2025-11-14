# app/azure_tts_streaming.py - Azure TTS with Streaming Support
import os
import asyncio
import threading
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import pyaudio
import wave
import tempfile
import time
import logging
import queue
import re

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logging.warning("Azure Speech SDK not installed")

from app.config_manager import config_manager

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureTTSStreaming:
    """
    Azure TTS with streaming text support - speaks as text chunks arrive
    """

    def __init__(self,
                 voice_name: str = None,
                 language: str = None,
                 speech_synthesis_output_format: str = None):

        config = config_manager.get_config()

        self.voice_name = voice_name or config.tts.azure_voice_name
        self.language = language or config.tts.azure_language
        self.output_format = speech_synthesis_output_format or config.tts.azure_output_format
        self.timeout = config.tts.timeout
        self.speech_config = None
        self.synthesizer = None

        # Audio playback control
        self._is_playing = False
        self._stop_playback = False
        self._current_stream = None
        self._pyaudio = None
        self._playback_lock = threading.RLock()
        self._playback_thread = None

        # Streaming support
        self._text_queue = queue.Queue()
        self._streaming_active = False
        self._streaming_thread = None
        self._sentence_buffer = ""
        self._playback_semaphore = threading.Semaphore(1)  # Only allow 1 concurrent playback

        # Statistics
        self.stats = {
            'tts_requests': 0,
            'audio_generated': 0,
            'playback_completed': 0,
            'playback_interrupted': 0,
            'errors': 0,
            'total_audio_duration': 0.0,
            'streaming_chunks_processed': 0
        }

        self._initialize_client_and_audio()

    def _initialize_client_and_audio(self):
        """Initialize Azure Speech SDK and PyAudio"""
        try:
            config = config_manager.get_config()
            if not config.features.enable_tts:
                logger.info("TTS disabled in configuration")
                return

            if not AZURE_AVAILABLE:
                logger.error("Azure Speech SDK not available")
                raise ImportError("Azure Speech SDK not installed")

            azure_key = os.getenv("AZURE_SPEECH_KEY")
            azure_region = os.getenv("AZURE_SPEECH_REGION")

            if not azure_key or not azure_region:
                logger.error("AZURE_SPEECH_KEY or AZURE_SPEECH_REGION not found")
                raise ValueError("Azure credentials not found")

            self.speech_config = speechsdk.SpeechConfig(
                subscription=azure_key,
                region=azure_region
            )

            self.speech_config.speech_synthesis_voice_name = self.voice_name

            format_mapping = {
                'audio-16khz-32kbitrate-mono-mp3': speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3,
                'audio-24khz-48kbitrate-mono-mp3': speechsdk.SpeechSynthesisOutputFormat.Audio24Khz48KBitRateMonoMp3,
                'riff-24khz-16bit-mono-pcm': speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm,
            }

            if self.output_format in format_mapping:
                self.speech_config.set_speech_synthesis_output_format(format_mapping[self.output_format])

            self.synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )

            self._pyaudio = pyaudio.PyAudio()

            logger.info(f"✅ Azure TTS Streaming initialized: {self.voice_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure TTS: {e}")
            self.speech_config = None
            self.synthesizer = None

    @property
    def is_playing(self):
        """Thread-safe access to playing state"""
        with self._playback_lock:
            return self._is_playing or self._streaming_active

    def stop_current_audio(self):
        """Force stop all audio playback and streaming"""
        config = config_manager.get_config()
        if not config.features.enable_tts:
            return

        logger.info("🛑 STOPPING ALL AUDIO - Clearing queue and stopping playback")

        # First, set stop flags
        self._stop_playback = True
        self._streaming_active = False

        # Clear the entire text queue IMMEDIATELY
        cleared_count = 0
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
                self._text_queue.task_done()
                cleared_count += 1
            except queue.Empty:
                break

        if cleared_count > 0:
            logger.info(f"🗑️ Cleared {cleared_count} pending sentences from queue")

        # Clear sentence buffer
        self._sentence_buffer = ""

        with self._playback_lock:
            # Stop current playback
            if self._is_playing:
                logger.info("🛑 Force stopping current audio playback")
                self.stats['playback_interrupted'] += 1

                if self._current_stream:
                    try:
                        self._current_stream.stop_stream()
                        self._current_stream.close()
                        logger.info("✅ Stream closed")
                    except Exception as e:
                        logger.warning(f"Error closing stream: {e}")
                    finally:
                        self._current_stream = None

                self._is_playing = False

                if self._playback_thread and self._playback_thread.is_alive():
                    self._playback_thread.join(timeout=1.0)
                self._playback_thread = None

        time.sleep(0.05)
        logger.info("✅ All audio stopped and queue cleared")

    def _split_into_sentences(self, text: str):
        """Split text into sentences for natural TTS streaming"""
        # Match sentence endings: . ! ? followed by space or end
        sentence_endings = re.compile(r'([.!?]+[\s]+|[.!?]+$)')
        sentences = sentence_endings.split(text)

        result = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            elif sentences[i].strip():
                result.append(sentences[i])

        return [s.strip() for s in result if s.strip()]

    async def add_text_chunk(self, text_chunk: str):
        """Add a text chunk to the streaming queue"""
        if not self._streaming_active:
            return

        self._sentence_buffer += text_chunk
        self.stats['streaming_chunks_processed'] += 1

        # Check if we have complete sentences
        sentences = self._split_into_sentences(self._sentence_buffer)

        if len(sentences) > 1:
            # Keep last incomplete sentence in buffer
            for sentence in sentences[:-1]:
                self._text_queue.put(sentence)
            self._sentence_buffer = sentences[-1]
        elif any(self._sentence_buffer.endswith(p) for p in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
            # Complete sentence
            self._text_queue.put(self._sentence_buffer.strip())
            self._sentence_buffer = ""

    async def start_streaming(self):
        """Start streaming TTS mode"""
        config = config_manager.get_config()
        if not config.features.enable_tts or not self.synthesizer:
            return

        logger.info("🎙️ Starting streaming TTS mode")
        self._streaming_active = True
        self._stop_playback = False  # Reset stop flag for new streaming session
        self._sentence_buffer = ""

        # Start background thread to process text queue
        self._streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self._streaming_thread.start()

    def _streaming_worker(self):
        """Background worker to process text queue and generate TTS"""
        while self._streaming_active:
            try:
                # Get text from queue with timeout
                text = self._text_queue.get(timeout=0.5)

                if text and not self._stop_playback:
                    # Wait if currently playing (should not happen but extra safety)
                    while self._is_playing and not self._stop_playback:
                        time.sleep(0.1)

                    if not self._stop_playback:
                        logger.info(f"🔊 Speaking: {text[:50]}...")
                        # Generate audio
                        audio_data = self._generate_audio_sync(text)
                        if audio_data and not self._stop_playback:
                            # Play audio and WAIT for it to complete before processing next sentence
                            self._play_audio_sync(audio_data)

                            # Small delay between sentences for natural speech
                            if not self._stop_playback:
                                time.sleep(0.15)

                self._text_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Streaming worker error: {e}")
                try:
                    self._text_queue.task_done()
                except:
                    pass

    async def finish_streaming(self):
        """Finish streaming mode and speak any remaining buffered text"""
        if not self._streaming_active:
            return

        # Add remaining buffer to queue
        if self._sentence_buffer.strip():
            self._text_queue.put(self._sentence_buffer.strip())
            self._sentence_buffer = ""

        # Wait for queue to be processed
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._text_queue.join()
            )
        except Exception as e:
            logger.error(f"Error finishing streaming: {e}")

        self._streaming_active = False
        logger.info("✅ Streaming TTS finished")

    def _generate_audio_sync(self, text: str) -> Optional[bytes]:
        """Synchronously generate audio from text"""
        try:
            result = self.synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            return None
        except Exception as e:
            logger.error(f"❌ Audio generation failed: {e}")
            return None

    async def _generate_audio_data(self, text: str) -> Optional[bytes]:
        """Generate audio data asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_audio_sync, text)

    def _play_audio_sync(self, audio_data: bytes):
        """Synchronously play audio data"""
        config = config_manager.get_config()
        if not config.features.enable_tts:
            return

        # Acquire semaphore to ensure only ONE audio plays at a time
        self._playback_semaphore.acquire()

        try:
            # Check stop flag BEFORE setting playing state
            if self._stop_playback:
                self._playback_semaphore.release()
                return

            with self._playback_lock:
                self._is_playing = True
                # Don't reset _stop_playback to False here - it should stay True if set during playback
            # Determine file format based on output format setting
            if 'mp3' in self.output_format.lower():
                # For MP3, we need to convert to WAV first using pydub
                try:
                    from pydub import AudioSegment
                    from pydub.playback import play
                    import io

                    # Convert MP3 bytes to AudioSegment
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")

                    # Export as WAV to temp file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                        audio_segment.export(tmp_file.name, format='wav')
                        temp_filename = tmp_file.name

                except ImportError:
                    logger.error("pydub not available for MP3 playback. Please use WAV format or install pydub.")
                    return
            else:
                # For WAV/PCM, write directly
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file.flush()
                    temp_filename = tmp_file.name

            # Play the WAV file
            with wave.open(temp_filename, 'rb') as wf:
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                frame_rate = wf.getframerate()
                frames_per_buffer = 1024

                stream = self._pyaudio.open(
                    format=self._pyaudio.get_format_from_width(sample_width),
                    channels=channels,
                    rate=frame_rate,
                    output=True,
                    frames_per_buffer=frames_per_buffer
                )

                with self._playback_lock:
                    self._current_stream = stream

                data = wf.readframes(frames_per_buffer)
                while data and not self._stop_playback:
                    stream.write(data)
                    data = wf.readframes(frames_per_buffer)

                stream.stop_stream()
                stream.close()

                with self._playback_lock:
                    self._current_stream = None

            try:
                os.unlink(temp_filename)
            except:
                pass

            if not self._stop_playback:
                self.stats['playback_completed'] += 1

        except Exception as e:
            logger.error(f"❌ Playback error: {e}")
            self.stats['errors'] += 1
            import traceback
            traceback.print_exc()
        finally:
            with self._playback_lock:
                self._is_playing = False
            # Release semaphore to allow next audio to play
            self._playback_semaphore.release()

    async def text_to_speech(self, text: str, play_audio: bool = True, interrupt_current: bool = True) -> Optional[bytes]:
        """Convert text to speech (non-streaming mode)"""
        config = config_manager.get_config()
        if not config.features.enable_tts or not self.synthesizer:
            return None

        if not text or not text.strip():
            return None

        try:
            logger.info(f"🔊 Converting text to speech: '{text[:50]}...'")
            self.stats['tts_requests'] += 1

            if interrupt_current:
                await asyncio.get_event_loop().run_in_executor(None, self.stop_current_audio)

            audio_data = await asyncio.wait_for(
                self._generate_audio_data(text),
                timeout=self.timeout
            )

            if audio_data:
                self.stats['audio_generated'] += 1
                if play_audio:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._play_audio_sync(audio_data)
                    )

            return audio_data

        except asyncio.TimeoutError:
            logger.error(f"⏰ TTS generation timed out")
            self.stats['errors'] += 1
            return None
        except Exception as e:
            logger.error(f"❌ TTS conversion failed: {e}")
            self.stats['errors'] += 1
            return None

    def get_voice_info(self) -> Dict[str, Any]:
        """Get voice configuration info"""
        config = config_manager.get_config()
        return {
            'provider': 'azure_streaming',
            'voice_name': self.voice_name,
            'language': self.language,
            'output_format': self.output_format,
            'is_playing': self.is_playing,
            'streaming_active': self._streaming_active,
            'tts_enabled': config.features.enable_tts
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics"""
        return {
            'provider': 'azure_streaming',
            'tts_requests': self.stats['tts_requests'],
            'audio_generated': self.stats['audio_generated'],
            'playback_completed': self.stats['playback_completed'],
            'playback_interrupted': self.stats['playback_interrupted'],
            'streaming_chunks_processed': self.stats['streaming_chunks_processed'],
            'errors': self.stats['errors']
        }

    def cleanup(self):
        """Cleanup resources"""
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
        """Cleanup on destruction"""
        self.cleanup()
