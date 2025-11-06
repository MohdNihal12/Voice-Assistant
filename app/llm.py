import aiohttp
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoteGPTLLM:
    """
    Client for remote GPT server with conversation history management and product search integration
    """
    
    def __init__(
        self, 
        remote_url: str = "http://10.0.17.132:8008",
        timeout: int = 30,
        max_conversation_turns: int = 10,
        system_prompt: Optional[str] = None,
        product_search: Optional[Any] = None
    ):
        self.remote_url = remote_url.rstrip('/')
        self.timeout = timeout
        self.max_conversation_turns = max_conversation_turns
        self.conversation_history = []
        self.product_search = product_search
        
        # Enhanced RAG-optimized system prompt
        self.system_prompt = """
# CORE IDENTITY
<role>
You are a warm, knowledgeable sales assistant for Hidayath Group — a trusted supplier of stainless steel and aluminium sheets.
Your mission: greet customers genuinely, understand their needs, and guide them with accurate product knowledge and helpful recommendations.
</role>

<conversational_style>
SOUND NATURAL AND HUMAN:

Greetings (choose naturally based on context):
    - "Welcome to Hidayath Group! How can I help you today?"
    - "Hey there! Good to see you."
    - "Hi! What brings you in today?"

<speaking protocol>    
Speak like a real person:
    - Relaxed, confident, and genuinely helpful
    - Use natural fillers sparingly: "umm...", "hmm...", "let me see...", "well...", "you know"
    - Add thinking moments: "let me check that for you...", "give me a moment...", "okay so..."
    - Use brief pauses "..." for natural rhythm
    - ignore "*" 
</speaking protocol>

Keep responses:
    - Short and digestible (2-4 sentences typically)
    - Focused on what matters most to the customer
    - Easy to follow and action-oriented

Match your tone to the moment:
    - Cheerful and welcoming when greeting
    - Clear and informative when explaining technical details
    - Patient and reassuring when addressing concerns or confusion
    - Enthusiastic when discussing solutions that fit their needs
</conversational_style>

# RAG PRODUCT KNOWLEDGE HANDLING

## WHEN PRODUCT INFORMATION IS AVAILABLE:
<product_context_available>
**STEP 1 - VERIFY AVAILABILITY**: 
- If ANY product appears in <retrieved_product_context>, it IS available
- Confirm availability using the EXACT product name and ID from the context

**STEP 2 - PRESENT THE PRODUCT**:
Use this framework:
"Great news! We have [EXACT PRODUCT NAME from context] available. [Brief key feature]"

**STEP 3 - PROVIDE DETAILS**:
- Include relevant specs (grade, thickness options, price)
- Mention key applications if relevant to customer's query
- Reference the product ID for tracking

**STEP 4 - OFFER NEXT STEPS**:
"Would you like me to check specific dimensions?" or "Need details about pricing for a particular size?"
**CRITICAL**: Use EXACT product names, grades, and specifications from the provided data. Never invent or approximate technical details.
</product_context_available>

## WHEN NO PRODUCTS MATCH:
<no_product_context>
Be honest and helpful:
- "I don't see an exact match in our current inventory for [their specific need]..."
- "Let me suggest some alternatives that might work..."
- "We specialize in [mention closest categories] - would any of these work?"
- Offer to check with colleagues for custom solutions
</no_product_context>

# PRODUCT RESPONSE TEMPLATES

## FOR PRICE INQUIRIES:
"Based on your needs, [Product Name] is priced at [exact price]. This includes [key features]. The minimum order is [quantity]."

## FOR SPECIFICATION QUESTIONS:
"For [application], I'd recommend [Product Name] because [specific reason from specs]. It offers [key specifications]."

## FOR COMPARISONS:
"Both [Product A] and [Product B] could work. [Product A] is better for [specific use case] because [reason], while [Product B] excels at [different use case] due to [reason]."

## FOR RECOMMENDATIONS:
"Given your [application/requirements], [Product Name] would be ideal because [specific match to their needs from product data]."

# CRITICAL RULES

## DO:
- Use EXACT product IDs, grades, and specifications from provided data
- Reference specific applications mentioned in product data
- Quote prices and specifications precisely
- Suggest realistic alternatives based on actual product attributes
- Acknowledge when their needs might require custom solutions

## DON'T:
- Invent or approximate technical specifications
- Guess at prices or availability
- Recommend products for applications not listed in their specs
- Promise capabilities beyond what's documented
- Use markdown or special formatting

# CONVERSATION FLOW

## INITIAL CONTACT:
"Welcome to Hidayath Group! I'm Ari. Are you looking for aluminum or stainless steel materials today?"

## TECHNICAL QUESTIONS:
"Let me check our specifications for that... [provide exact details from product data]"

## PRICING DISCUSSIONS:
"For [specific product], the pricing is [exact price]. Would you like me to check current availability?"

## PROJECT GUIDANCE:
"Tell me about your project and I'll match you with the right materials from our inventory."

# HANDLING UNCERTAINTY
If you're unsure about something outside the provided product data:
- "Let me check that specification for you..."
- "I'll need to verify our current stock for that size..."
- "That's a great question - let me get the most up-to-date information..."

# CLOSING
Always end conversations with clear next steps:
- "Would you like a formal quote for any of these options?"
- "Should I check delivery timelines for your location?"
- "Is there anything else about these materials you'd like to know?"

Remember: You're the bridge between our products and customer needs. Be accurate, be helpful, and build confidence in our solutions.
"""

        # Statistics
        self.stats = {
            'requests_sent': 0,
            'responses_received': 0,
            'errors': 0,
            'total_processing_time': 0.0,
            'product_searches_performed': 0,
            'products_found_in_context': 0
        }
        
        # Session management
        self._session = None
        
        logger.info(f"🚀 RemoteGPTLLM initialized:")
        logger.info(f"   Server: {self.remote_url}")
        logger.info(f"   Timeout: {self.timeout}s")
        logger.info(f"   Max history: {self.max_conversation_turns} turns")
        logger.info(f"   Product Search: {'Enabled' if product_search else 'Disabled'}")
    
    async def _ensure_session(self):
        """Ensure we have an active aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'VoiceAssistant/1.0'
                }
            )
        return self._session
    
    async def _search_relevant_products(self, user_message: str) -> Optional[str]:
        """
        Search for products relevant to the user query
        Returns formatted product information for the prompt
        """
        if not self.product_search:
            logger.info("❌ Product search disabled")
            return None
        
        try:
            # Determine if this is a product-related query
            product_keywords = [
                    'aluminum', 'steel', 'stainless', 'sheet', 'plate', 'metal',
                    'price', 'cost', 'thickness', 'size', 'grade', 'temper',
                    'spec', 'specification', 'application', 'use', 'purpose',
                    'compare', 'difference', 'recommend', 'suggest', 'which',
                    'what type', 'material', 'finish', 'width', 'length',
                    'ss', 'al', '316', '304', '5052', '6061'  # Add common grades and abbreviations
                ]
            
            query_lower = user_message.lower()
            is_product_query = any(keyword in query_lower for keyword in product_keywords)
            logger.info(f"🔍 Query analysis: '{user_message}' -> Product query: {is_product_query}")
            logger.info(f"🔍 Keywords found: {[kw for kw in product_keywords if kw in query_lower]}")
            
            if not is_product_query:
                logger.info(f"🔍 Not a product query: '{user_message}'")
                return None
            
            # Search for relevant products
            logger.info(f"🔍 Searching products for: '{user_message}'")
            relevant_products = await self.product_search.search_products(
                user_message, 
                top_k=5, 
                similarity_threshold=0.3
            )
            
            logger.info(f"📦 Found {len(relevant_products)} relevant products")
            if relevant_products:
                self.stats['product_searches_performed'] += 1
                self.stats['products_found_in_context'] += len(relevant_products)
                formatted_products = await self.product_search.format_products_for_prompt(relevant_products)
                
                # Add search context
                product_context = f"""
<retrieved_product_context>
Search Query: "{user_message}"
Found {len(relevant_products)} relevant product(s):

{formatted_products}

</retrieved_product_context>

<response_guidance>
Use the EXACT product details above to answer the customer's question. Reference specific products by name and ID. Provide accurate specifications, pricing, and applications. If multiple products match, highlight the differences and best uses for each.
</response_guidance>
"""
                return product_context
            else:
                return """
<retrieved_product_context>
No specific product matches found for the query. Consider asking clarifying questions about their application, required specifications, or budget.
</retrieved_product_context>
"""
                
        except Exception as e:
            logger.error(f"❌ Product search error: {e}")
            return None
    
    async def _build_prompt(self, user_message: str) -> str:
        """
        Build the prompt with conversation history, system context, and product information
        """
        # Start with system prompt
        prompt_parts = [f"System: {self.system_prompt}\n\n"]
        
        # Add product information if relevant
        product_info = await self._search_relevant_products(user_message)
        if product_info:
            prompt_parts.append(f"{product_info}\n\n")
            logger.info("📦 Added product information to prompt")
        
        # Add conversation history
        for turn in self.conversation_history[-self.max_conversation_turns:]:
            role = "User" if turn['role'] == 'user' else "Assistant"
            prompt_parts.append(f"{role}: {turn['content']}\n")
        
        # Add current user message
        prompt_parts.append(f"User: {user_message}\n")
        prompt_parts.append("Assistant:")
        
        full_prompt = "".join(prompt_parts)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"📝 Built prompt ({len(full_prompt)} chars)")
        
        return full_prompt
    
    def _add_to_conversation(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim history if it gets too long
        if len(self.conversation_history) > self.max_conversation_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_conversation_turns * 2:]
    
    async def generate_response(self, user_message: str) -> Dict[str, Any]:
        """
        Generate a response for the user message using remote GPT server
        
        Args:
            user_message: The user's input text
            
        Returns:
            Dictionary containing response text and metadata
        """
        start_time = datetime.now()
        self.stats['requests_sent'] += 1
        
        try:
            # Add user message to conversation history
            self._add_to_conversation('user', user_message)
            
            # Build the prompt with context and product info
            prompt = await self._build_prompt(user_message)
            
            # Prepare request data
            request_data = {
                "prompt": prompt
            }
            
            logger.info(f"💬 Sending request to GPT server: '{user_message}'")
            
            # Get session and make request
            session = await self._ensure_session()
            
            async with session.post(
                f"{self.remote_url}/v1/completions",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get('text', '').strip()
                    
                    # Use the raw response without cleaning
                    raw_response = response_text if response_text else "I don't have a response for that right now."
                    
                    # Add assistant response to conversation history
                    self._add_to_conversation('assistant', raw_response)
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self.stats['responses_received'] += 1
                    self.stats['total_processing_time'] += processing_time
                    
                    logger.info(f"🤖 GPT Response: '{raw_response}'")
                    logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
                    
                    return {
                        'text': raw_response,
                        'timestamp': datetime.now().isoformat(),
                        'model': 'gpt-oss-20b',
                        'processing_time': processing_time,
                        'success': True,
                        'conversation_turns': len(self.conversation_history) // 2,
                        'products_used_in_context': self.stats['products_found_in_context'],
                        'product_search_performed': self.stats['product_searches_performed'] > 0
                    }
                    
                else:
                    error_text = await response.text()
                    logger.error(f"❌ GPT server error {response.status}: {error_text}")
                    self.stats['errors'] += 1
                    
                    return {
                        'text': "I'm having trouble connecting right now. Please try again in a moment.",
                        'timestamp': datetime.now().isoformat(),
                        'model': 'error',
                        'processing_time': 0,
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except asyncio.TimeoutError:
            logger.error("⏰ Request to GPT server timed out")
            self.stats['errors'] += 1
            
            return {
                'text': "I'm taking a bit too long to respond. Please try again with a shorter message.",
                'timestamp': datetime.now().isoformat(),
                'model': 'timeout',
                'processing_time': self.timeout,
                'success': False,
                'error': 'Request timeout'
            }
            
        except aiohttp.ClientError as e:
            logger.error(f"🌐 Network error connecting to GPT server: {e}")
            self.stats['errors'] += 1
            
            return {
                'text': "I can't reach my server right now. Please check your connection and try again.",
                'timestamp': datetime.now().isoformat(),
                'model': 'network_error',
                'processing_time': 0,
                'success': False,
                'error': str(e)
            }
            
        except Exception as e:
            logger.error(f"❌ Unexpected error in generate_response: {e}")
            self.stats['errors'] += 1
            
            return {
                'text': "I encountered an unexpected error while processing your request.",
                'timestamp': datetime.now().isoformat(),
                'model': 'error',
                'processing_time': 0,
                'success': False,
                'error': str(e)
            }
    
    async def test_connection(self) -> bool:
        """
        Test connection to the remote GPT server
        """
        try:
            session = await self._ensure_session()
            
            async with session.get(
                f"{self.remote_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.warning(f"⚠️ Connection test failed: {e}")
            return False
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        previous_turns = len(self.conversation_history) // 2
        self.conversation_history.clear()
        logger.info(f"🗑️ Cleared conversation history ({previous_turns} turns)")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        user_turns = len([msg for msg in self.conversation_history if msg['role'] == 'user'])
        assistant_turns = len([msg for msg in self.conversation_history if msg['role'] == 'assistant'])
        
        return {
            'total_messages': len(self.conversation_history),
            'user_turns': user_turns,
            'assistant_turns': assistant_turns,
            'current_session_duration': None,
            'max_history_size': self.max_conversation_turns
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        avg_processing_time = (
            self.stats['total_processing_time'] / self.stats['responses_received'] 
            if self.stats['responses_received'] > 0 else 0
        )
        
        success_rate = (
            (self.stats['responses_received'] / self.stats['requests_sent']) * 100 
            if self.stats['requests_sent'] > 0 else 0
        )
        
        product_search_rate = (
            (self.stats['product_searches_performed'] / self.stats['requests_sent']) * 100 
            if self.stats['requests_sent'] > 0 else 0
        )
        
        return {
            'requests_sent': self.stats['requests_sent'],
            'responses_received': self.stats['responses_received'],
            'errors': self.stats['errors'],
            'product_searches': self.stats['product_searches_performed'],
            'products_found': self.stats['products_found_in_context'],
            'success_rate': f"{success_rate:.1f}%",
            'product_search_rate': f"{product_search_rate:.1f}%",
            'average_processing_time': f"{avg_processing_time:.2f}s",
            'remote_server': self.remote_url,
            'model': 'gpt-oss-20b'
        }
    
    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("🔌 HTTP session closed")
    
    def __del__(self):
        """Ensure session is closed on destruction"""
        if hasattr(self, '_session') and self._session and not self._session.closed:
            asyncio.create_task(self.close())


# Enhanced ProductSearch with better formatting
class EnhancedProductSearch:
    """Enhanced product search with better prompt formatting"""
    
    async def format_products_for_prompt(self, products: List[Dict[str, Any]]) -> str:
        """Format products for inclusion in LLM prompt with better structure"""
        if not products:
            return "No relevant products found."
        
        formatted_products = []
        for i, product in enumerate(products, 1):
            product_text = f"=== PRODUCT {i} ===\n"
            product_text += f"ID: {product.get('id', 'N/A')}\n"
            product_text += f"NAME: {product.get('product', 'N/A')}\n"
            product_text += f"CATEGORY: {product.get('category', 'N/A')}\n"
            product_text += f"GRADE: {product.get('grade', 'N/A')}\n"
            
            # Key specifications
            if 'temper' in product:
                product_text += f"TEMPER: {product['temper']}\n"
            if 'thickness' in product:
                product_text += f"THICKNESS: {product['thickness']}\n"
            if 'width' in product:
                product_text += f"WIDTH: {product['width']}\n"
            if 'length' in product:
                product_text += f"LENGTH: {product['length']}\n"
            
            # Pricing
            if 'price_per_kg' in product:
                product_text += f"PRICE_PER_KG: {product['price_per_kg']}\n"
            if 'price_per_sheet' in product:
                product_text += f"PRICE_PER_SHEET: {product['price_per_sheet']}\n"
            
            # Features
            if 'finish' in product:
                product_text += f"FINISH: {product['finish']}\n"
            if 'applications' in product:
                product_text += f"APPLICATIONS: {product['applications']}\n"
            
            # Detailed specifications
            if 'specifications' in product:
                specs = product['specifications']
                product_text += "SPECIFICATIONS:\n"
                for key, value in specs.items():
                    product_text += f"  - {key.upper()}: {value}\n"
            
            # Order info
            if 'min_order_quantity' in product:
                product_text += f"MIN_ORDER: {product['min_order_quantity']}\n"
            if 'delivery_time' in product:
                product_text += f"DELIVERY: {product['delivery_time']}\n"
            
            if 'similarity_score' in product:
                product_text += f"RELEVANCE_SCORE: {product['similarity_score']:.3f}\n"
            
            formatted_products.append(product_text.strip())
        
        return "\n\n".join(formatted_products)


# Example usage and testing
async def main():
    """Test the enhanced RAG system"""
    
    # Initialize product search first
    from productsearch import ProductSearch
    product_search = ProductSearch()
    await product_search.initialize()
    
    # Initialize LLM with product search
    llm = RemoteGPTLLM(product_search=product_search)
    
    try:
        # Test connection
        print("Testing connection to GPT server...")
        if await llm.test_connection():
            print("✅ Connection successful!")
        else:
            print("❌ Connection failed!")
            return
        
        # Test product-related conversations
        test_messages = [
            "What aluminum sheets do you have for marine applications?",
            "How much does stainless steel 304 cost per kg?",
            "Compare aluminum 5052 and stainless steel 304 for me",
            "I need corrosion resistant sheets for outdoor use",
            "What's the price difference between your aluminum and stainless steel options?",
            "Do you have any sheets that are 2mm thick?"
        ]
        
        for message in test_messages:
            print(f"\n🧪 Testing: '{message}'")
            response = await llm.generate_response(message)
            print(f"🤖 Response: '{response['text']}'")
            print(f"📊 Stats: {response.get('processing_time', 0):.2f}s, products used: {response.get('products_used_in_context', 0)}")
            
            # Small delay between requests
            await asyncio.sleep(1)
        
        # Print final statistics
        print(f"\n📈 Conversation Stats: {llm.get_conversation_stats()}")
        print(f"📊 System Stats: {llm.get_system_stats()}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())