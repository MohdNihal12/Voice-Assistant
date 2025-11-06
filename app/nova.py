import asyncio
import json
import base64
from typing import Dict, Optional, Any, Callable
import aiohttp
from datetime import datetime
import numpy as np


class DeepgramNovaSTT:
    """
    Deepgram Nova 2 Speech-to-Text with live streaming capabilities
    """
    
    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        model: str = "nova-3",
        endpoint: str = "wss://api.deepgram.com/v1/listen",
        verbose: bool = False,
        enable_multilingual: bool = True  # NEW: Add multilingual support
    ):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.language = language
        self.model = model
        self.endpoint = endpoint
        self.verbose = verbose
        self.enable_multilingual = enable_multilingual  # NEW
        
        # Connection state management
        self._connection_lock = asyncio.Lock()
        self._is_connected = False
        self._current_connection = None
        
        # Audio streaming settings
        self.chunk_size = 1024  # Optimal chunk size for Deepgram
        self.sample_width = 2   # 16-bit audio
        
        # Background task for receiving
        self._receive_task = None
        
        # Statistics
        self.stats = {
            'sessions_processed': 0,
            'total_audio_duration': 0.0,
            'total_processing_time': 0.0,
            'errors': 0,
            'chunks_sent': 0,
            'utterances_received': 0
        }
        
        print(f"🎯 Deepgram Nova STT Initialized:")
        print(f"   Model: {self.model}")
        print(f"   Sample Rate: {self.sample_rate}Hz")
        print(f"   Language: {self.language or 'auto-detect'}")
        print(f"   Multilingual: {self.enable_multilingual}")  # NEW
    
    async def create_streaming_connection(self, websocket, callback: Callable):
        """Create a live streaming connection to Deepgram Nova"""
        async with self._connection_lock:
            if self._is_connected and self._current_connection:
                if self.verbose:
                    print("⚠️  Already connected to Deepgram, reusing connection")
                return self._current_connection
            
            try:
                # UPDATED Deepgram parameters for multilingual support
                params = {
                    'model': self.model,
                    'encoding': 'linear16',
                    'sample_rate': str(self.sample_rate),
                    'channels': '1',
                    'interim_results': 'true',
                    'punctuate': 'true',
                    'utterance_end_ms': '1000',
                    'endpointing': '100',  # CORRECTED: 100ms for code-switching as recommended
                }
                
                # UPDATED Language configuration
                if self.enable_multilingual:
                    # Enable multilingual/code-switching mode
                    params['language'] = 'multi'  # Use 'multi' for multilingual
                    if self.verbose:
                        print("🌍 MULTILINGUAL MODE ENABLED - Supports code-switching between languages")
                elif self.language:
                    # Specific language mode
                    params['language'] = self.language
                else:
                    # Auto-detect language mode
                    params['detectlanguage'] = 'true'
                
                if self.verbose:
                    print(f"🔗 Connecting to Deepgram Nova...")
                    print(f"   Endpoint: {self.endpoint}")
                    print(f"   Params: {params}")
                
                # Build URL with query parameters
                query_string = "&".join([f"{k}={v}" for k, v in params.items()])
                url = f"{self.endpoint}?{query_string}"
                
                if self.verbose:
                    print(f"   Full URL: {url.split('?')[0]}")  # Don't log full URL with API key
                
                # Create Deepgram WebSocket connection
                session = aiohttp.ClientSession()
                deepgram_ws = await session.ws_connect(
                    url,
                    headers={
                        'Authorization': f'Token {self.api_key}',
                        'User-Agent': 'Nova-STT-Client/1.0'
                    },
                    heartbeat=30,
                    timeout=30.0
                )
                
                if self.verbose:
                    print("✅ Connected to Deepgram Nova")
                
                self._is_connected = True
                
                async def send_audio(audio_data: bytes):
                    """Send audio data to Deepgram"""
                    try:
                        if len(audio_data) > 0:
                            if self.verbose:
                                print(f"📤 Sending {len(audio_data)} bytes to Deepgram")
                            
                            # Send raw bytes directly to Deepgram
                            await deepgram_ws.send_bytes(audio_data)
                            self.stats['chunks_sent'] += 1
                            
                    except Exception as e:
                        if self.verbose:
                            print(f"❌ Error sending audio to Deepgram: {e}")
                        self.stats['errors'] += 1
                        raise
                
                async def receive_transcriptions():
                    """Receive transcriptions from Deepgram"""
                    try:
                        utterance_count = 0
                        print("🎧 Starting to listen for Deepgram responses...")
                        
                        async for msg in deepgram_ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                
                                if self.verbose:
                                    print(f"📨 RAW DEEPGRAM MESSAGE: {json.dumps(data, indent=2)}")
                                
                                # Handle the message
                                result = await self._handle_deepgram_message(data, callback, websocket)
                                
                                if result and result.get('is_final'):
                                    utterance_count += 1
                                    self.stats['utterances_received'] += 1
                                    
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                print(f"❌ Deepgram WebSocket error: {msg.data}")
                                self.stats['errors'] += 1
                                break
                                
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                print("🔌 Deepgram WebSocket closed by server")
                                break
                        
                        print(f"📊 Receive loop ended: {utterance_count} utterances processed")
                            
                    except asyncio.CancelledError:
                        print("🛑 Receive task cancelled")
                        raise
                    except Exception as e:
                        print(f"❌ Error in receive loop: {e}")
                        import traceback
                        traceback.print_exc()
                        self.stats['errors'] += 1
                
                async def close_connection():
                    """Close Deepgram connection properly"""
                    async with self._connection_lock:
                        if self._is_connected:
                            try:
                                # Cancel receive task if running
                                if self._receive_task and not self._receive_task.done():
                                    self._receive_task.cancel()
                                    try:
                                        await self._receive_task
                                    except asyncio.CancelledError:
                                        pass
                                
                                # Send close frame to Deepgram
                                await deepgram_ws.close()
                                await session.close()
                                
                                self._is_connected = False
                                self._current_connection = None
                                self._receive_task = None
                                
                                if self.verbose:
                                    print("🔌 Deepgram connection closed gracefully")
                                    self._print_session_stats()
                                    
                            except Exception as e:
                                if self.verbose:
                                    print(f"❌ Error closing connection: {e}")
                
                # Store connection info
                connection_info = {
                    'send_audio': send_audio,
                    'receive_transcriptions': receive_transcriptions,
                    'close': close_connection,
                    'session': session,
                    'deepgram_ws': deepgram_ws
                }
                
                # START THE RECEIVE TASK - Only if not already running
                if not self._receive_task or self._receive_task.done():
                    self._receive_task = asyncio.create_task(receive_transcriptions())
                    print("✅ Receive task started")
                else:
                    print("ℹ️ Receive task already running")
                
                self._current_connection = connection_info
                return connection_info
                
            except aiohttp.ClientResponseError as e:
                if self.verbose:
                    print(f"❌ Deepgram API error {e.status}: {e.message}")
                self.stats['errors'] += 1
                raise
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to connect to Deepgram: {e}")
                    import traceback
                    traceback.print_exc()
                self.stats['errors'] += 1
                raise
    
    async def _handle_deepgram_message(self, data: Dict, callback: Callable, client_websocket):
        """
        Handle messages from Deepgram and forward to client
        """
        try:
            message_type = data.get('type', '')
            
            if message_type == 'Results':
                # This is a transcription result
                if self.verbose:
                    print(f"📥 Received Results message")
                
                channel = data.get('channel', {})
                alternatives = channel.get('alternatives', [])
                
                if alternatives:
                    transcript = alternatives[0].get('transcript', '').strip()
                    is_final = data.get('is_final', False)
                    confidence = alternatives[0].get('confidence', 0.0)
                    
                    # Extract language information - ENHANCED for multilingual
                    detected_language = data.get('language', 'en')
                    language_confidence = data.get('language_confidence', 0.0)
                    
                    # NEW: Handle multilingual word-level language detection
                    words = alternatives[0].get('words', [])
                    language_changes = []
                    
                    if words and self.enable_multilingual:
                        current_lang = None
                        for word in words:
                            word_lang = word.get('language', detected_language)
                            if word_lang != current_lang:
                                language_changes.append({
                                    'word': word.get('word', ''),
                                    'language': word_lang,
                                    'confidence': word.get('confidence', 0.0),
                                    'start': word.get('start', 0),
                                    'end': word.get('end', 0)
                                })
                                current_lang = word_lang
                    
                    if transcript:  # Only process non-empty transcripts
                        if self.verbose:
                            status = "FINAL" if is_final else "INTERIM"
                            lang_info = f" [{detected_language}]" if detected_language != 'en' else ""
                            print(f"📝 [{status}{lang_info}] '{transcript}' (confidence: {confidence:.2f})")
                            
                            # NEW: Log language changes in multilingual mode
                            if language_changes and self.enable_multilingual:
                                print(f"🌍 Language changes detected: {language_changes}")
                        
                        # Prepare transcription data with language info
                        transcription_data = {
                            'type': 'transcription',
                            'text': transcript,
                            'is_final': is_final,
                            'confidence': confidence,
                            'language': detected_language,
                            'language_confidence': language_confidence,
                            'duration': data.get('duration', 0),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # NEW: Add multilingual word-level data
                        if self.enable_multilingual and language_changes:
                            transcription_data['language_changes'] = language_changes
                            transcription_data['word_details'] = words
                        
                        # Send to client via callback
                        try:
                            await callback(transcription_data, client_websocket)
                        except Exception as e:
                            print(f"❌ Error in callback: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        return transcription_data
                else:
                    if self.verbose:
                        print("⚠️ No alternatives found in the results")
                    return None
            
            elif message_type == 'Metadata':
                # Connection metadata
                if self.verbose:
                    request_id = data.get('request_id', 'Unknown')
                    print(f"📊 Deepgram Metadata: {request_id}")
            
            elif message_type == 'SpeechStarted':
                # VAD detected speech start
                if self.verbose:
                    print("🎤 Deepgram: Speech started")
                
                try:
                    await client_websocket.send_json({
                        'type': 'speech_start',
                        'message': 'Speech detected',
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"⚠️  Could not send speech_start to client: {e}")
            
            elif message_type == 'UtteranceEnd':
                # Deepgram detected end of utterance
                if self.verbose:
                    print(f"⏹️  Deepgram: Utterance end")
                
                try:
                    await client_websocket.send_json({
                        'type': 'utterance_end',
                        'message': 'Utterance complete',
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"⚠️  Could not send utterance_end to client: {e}")
            
            elif message_type == 'Error':
                # Deepgram error
                error_msg = data.get('message', 'Unknown error')
                error_code = data.get('code', 'UNKNOWN')
                if self.verbose:
                    print(f"❌ Deepgram Error [{error_code}]: {error_msg}")
                self.stats['errors'] += 1
                
        except Exception as e:
            if self.verbose:
                print(f"❌ Error handling Deepgram message: {e}")
                import traceback
                traceback.print_exc()
            self.stats['errors'] += 1
        
        return None
    
    def _print_session_stats(self):
        """Print session statistics"""
        print("\n" + "="*80)
        print("🌊 DEEPGRAM STREAMING SESSION SUMMARY")
        print("="*80)
        print(f"   Chunks sent: {self.stats['chunks_sent']}")
        print(f"   Utterances processed: {self.stats['utterances_received']}")
        print(f"   Sessions processed: {self.stats['sessions_processed']}")
        print(f"   Total audio duration: {self.stats['total_audio_duration']:.2f}s")
        print(f"   Errors: {self.stats['errors']}")
        print(f"   Multilingual mode: {self.enable_multilingual}")
        
        if self.stats['sessions_processed'] > 0:
            avg_duration = self.stats['total_audio_duration'] / self.stats['sessions_processed']
            print(f"   Average duration: {avg_duration:.2f}s")
        print("="*80)
    
    async def transcribe_bytes(self, audio_data: bytes, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Transcribe audio bytes using Deepgram's REST API
        Useful for batch processing
        """
        try:
            url = "https://api.deepgram.com/v1/listen"
            headers = {
                'Authorization': f'Token {self.api_key}',
                'Content-Type': 'audio/wav'
            }
            
            params = {
                'model': self.model,
                'smart_format': 'true',
                'multichannel': 'false',
                'punctuate': 'true'
            }
            
            # UPDATED: Add multilingual support for batch processing
            if self.enable_multilingual:
                params['language'] = 'multi'
            elif self.language:
                params['language'] = self.language
            else:
                params['detectlanguage'] = 'true'
            
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{url}?{query_string}",
                    headers=headers,
                    data=audio_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_deepgram_response(result, return_confidence)
                    else:
                        error_text = await response.text()
                        raise Exception(f"Deepgram API error {response.status}: {error_text}")
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Batch transcription error: {e}")
            self.stats['errors'] += 1
            return {
                'text': '',
                'language': 'en',
                'duration': 0,
                'processing_time': 0,
                'confidence': 0.0,
                'rtf': 0.0,
                'error': str(e)
            }
    
    def _parse_deepgram_response(self, response: Dict, return_confidence: bool) -> Dict[str, Any]:
        """Parse Deepgram API response"""
        results = response.get('results', {})
        channels = results.get('channels', [])
        
        if not channels:
            return {
                'text': '',
                'language': 'en',
                'duration': response.get('metadata', {}).get('duration', 0),
                'processing_time': 0,
                'confidence': 0.0,
                'rtf': 0.0
            }
        
        channel = channels[0]
        alternatives = channel.get('alternatives', [])
        
        if not alternatives:
            return {
                'text': '',
                'language': 'en',
                'duration': response.get('metadata', {}).get('duration', 0),
                'processing_time': 0,
                'confidence': 0.0,
                'rtf': 0.0
            }
        
        alternative = alternatives[0]
        transcript = alternative.get('transcript', '').strip()
        confidence = alternative.get('confidence', 0.0)
        duration = response.get('metadata', {}).get('duration', 0)
        
        return {
            'text': transcript,
            'language': response.get('results', {}).get('channels', [{}])[0].get('detected_language', 'en'),
            'duration': duration,
            'processing_time': 0,
            'confidence': confidence if return_confidence else 0.0,
            'rtf': 0.0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about transcription sessions"""
        avg_duration = (self.stats['total_audio_duration'] / 
                       self.stats['sessions_processed'] if self.stats['sessions_processed'] > 0 else 0)
        
        return {
            'sessions_processed': self.stats['sessions_processed'],
            'total_audio_duration': self.stats['total_audio_duration'],
            'total_processing_time': self.stats['total_processing_time'],
            'errors': self.stats['errors'],
            'average_duration': avg_duration,
            'model': self.model,
            'sample_rate': self.sample_rate,
            'multilingual': self.enable_multilingual  # NEW
        }
    
    async def close(self):
        """Cleanup resources"""
        if self._current_connection:
            await self._current_connection['close']()