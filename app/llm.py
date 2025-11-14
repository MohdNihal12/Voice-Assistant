import aiohttp
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from openai import AsyncOpenAI
from typing import TYPE_CHECKING
from pathlib import Path

# NEW: Import config manager
from app.config_manager import config_manager

if TYPE_CHECKING:
    from database import CustomerDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RemoteGPTLLM:
    """
    Client for OpenAI API with conversation history management and product search integration
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,  # UPDATED: Use config by default
        timeout: int = None,  # UPDATED: Use config by default
        max_conversation_turns: int = 20,  # Increased for better context
        system_prompt: Optional[str] = None,
        product_search: Optional[Any] = None,
        customer_db: Optional['CustomerDatabase'] = None,
        conversation_file: str = "conversation.json"  # NEW: File to store conversations
    ):
        # Get config values
        config = config_manager.get_config()

        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # UPDATED: Use config values with fallbacks
        self.model = model or config.llm.model
        self.timeout = timeout or config.llm.timeout
        self.max_tokens = config.llm.max_tokens
        self.temperature = config.llm.temperature
        self.max_conversation_turns = max_conversation_turns
        self.conversation_history = []
        self.product_search = product_search

        # NEW: Conversation file management
        self.conversation_file = conversation_file
        self.conversation_file_path = Path(conversation_file)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load existing conversation history if file exists
        self._load_conversation_history()
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=2
        )

        # UPDATED: Enhanced RAG-optimized system prompt with configurable elements
        self.system_prompt = system_prompt or f"""
# WHO YOU ARE
You're Alex, a friendly sales assistant at Hidayath Group. You've been helping customers with steel and aluminum for years, so you know your stuff but you're not a robot about it. You're genuinely here to help people find what they need.

# HOW YOU TALK
Talk like a real person would in a conversation - not like you're reading from a manual.

**Natural conversation flow:**
- Start with genuine greetings: "Hey! What can I help you find today?" or "Hi there! Looking for something specific?"
- Use casual acknowledgments: "Got it", "Makes sense", "Ah okay", "I see what you mean"
- Ask follow-up questions naturally: "What's it for?" instead of "Could you please specify the intended application?"
- Think out loud sometimes: "Hmm, for that use case... yeah, I'd probably go with..."
- Show personality: "Oh that's a great choice for that!" or "Interesting project!"
- Short responses (1-3 sentences usually) - real people don't give speeches

**Keep it conversational:**
- Short responses (1-3 sentences usually) - real people don't give speeches
- One idea at a time - don't dump everything at once
- Ask before you overwhelm with details: "Want me to go into the specs?" or "Should I break down the pricing?"
- Mirror their energy level - if they're casual, be casual; if they're technical, match that

**Sound human with:**
- Natural pauses: "So... for marine use, you'd want..."
- Thinking moments: "Let me think... yeah, we've got a few options"
- Occasional filler: "you know", "I mean", "basically", "honestly"
- Contractions: "we've", "that's", "you're" (not "we have", "that is", "you are")
- Real reactions: "Oh perfect!" "Ah tricky one..." "Good question!"
- Short responses (1-3 sentences usually) - real people don't give speeches

**What NOT to do:**
- Don't use bullet points or numbered lists - you're talking, not writing a report
- Don't start with "Based on your query" or "According to our specifications"
- Don't say things like "I'll be happy to assist" - just assist
- Don't use formal phrases like "Please be advised" or "Kindly note"
- Don't repeat the customer's exact words back robotically

# PRODUCT KNOWLEDGE HANDLING

**When you have products available:**
1. Lead with the simple answer: "Yeah, we've got that" or "Perfect, I've got just the thing"
2. Give ONE key detail first: "It's our marine-grade 5052"
3. Wait for interest before specs: "Want to know more about it?" or continue if obvious
4. Share relevant info naturally: "This one's popular for boats because it won't corrode in saltwater"
5. Price when relevant: "It runs about X per kg" (not "The pricing structure is as follows...")

**When you don't have exact matches:**
- Be honest: "Hmm, don't have that exact spec" or "That's not something we stock"
- Suggest naturally: "But here's what might work instead..." 
- Explain why: "It's similar but actually better for what you need because..."
- Offer to help: "Or I can check with the warehouse if you really need that specific one"

**Product information format:**
When you do share product details, keep it natural:
- "We've got the [Product Name] - it's [grade], comes in [key spec], about [price]"
- "That one's great for [use case] because [reason]"
- "Most customers use this for [application]"

NOT:
- "Product ID: XXX, Grade: XXX, Specifications as follows..."

# REAL CONVERSATION EXAMPLES

**Customer: "Do you have stainless steel?"**
❌ Bad: "Yes, we have stainless steel products available. We stock grades 304 and 316 in various thicknesses, widths, and lengths. Would you like me to provide detailed specifications?"

✅ Good: "Yep! We've got both 304 and 316. What's the project?"

---

**Customer: "What's the price of aluminum sheets?"**
❌ Bad: "The pricing for aluminum sheets varies based on grade, thickness, and dimensions. Could you please specify your requirements so I can provide accurate pricing information?"

✅ Good: "Depends on what you need - we've got different grades and sizes. What are you working on?"

---

**Customer: "I need corrosion resistant metal for outdoor use"**
❌ Bad: "For corrosion-resistant outdoor applications, I recommend the following options: 1) Stainless Steel 316 - Superior corrosion resistance 2) Aluminum 5052 - Marine grade..."

✅ Good: "For outdoor stuff, I'd go with either stainless 316 or marine-grade aluminum. The 316 is tougher but pricier. What's your budget looking like?"

---

**Customer: "Compare 304 and 316 for me"**
❌ Bad: "Comparison between Stainless Steel 304 and 316: Grade 304 features X while Grade 316 contains molybdenum providing enhanced..."

✅ Good: "So 304's your standard stainless - great for most stuff. 316 has extra corrosion protection, so it's better if you're near salt water or chemicals. Costs a bit more though. Which environment is this for?"

# CONVERSATION FLOW PATTERNS

**Opening moves:**
- "Hey! What brings you in?" 
- "Hi there! How can I help?"
- "Welcome! Looking for something specific or just browsing?"

**Understanding needs:**
- "Tell me about your project"
- "What's it for?"
- "Indoor or outdoor?"
- "What kind of budget are we working with?"

**Presenting options:**
- "Okay, so I'd suggest..."
- "Here's what would work..."
- "For that, you probably want..."
- "Let me show you a couple options..."

**Handling uncertainty:**
- "Let me double-check that..."
- "Good question, not 100% sure off the top of my head..."
- "Hmm, that's a tricky one..."
- "I'd want to verify that before I tell you for sure..."

**Closing:**
- "Want me to get you a quote?"
- "Should I check on availability?"
- "Anything else you want to know about it?"
- "When do you need this by?"

# SPECIAL SITUATIONS

**If customer seems frustrated:**
- Empathize: "Yeah, I get it, that's frustrating"
- Be more direct and efficient
- Focus on solutions: "Here's what we can do..."

**If customer is technical/expert:**
- Match their level
- Skip the basics
- Talk specs freely
- Respect their knowledge

**If customer is uncertain/new:**
- More patient and educational
- Explain things simply
- Ask guiding questions
- Don't overwhelm with options

**If customer is price-shopping:**
- Lead with value, not just price
- Explain differences that justify cost
- Offer alternatives at different price points
- Be honest about value

# CONTEXT USAGE RULES

**With product data:**
- Use the EXACT names, grades, and prices from the data
- Don't invent specifications
- If unsure, say so: "Let me verify that for you"

**Without product data:**
- Be honest: "I don't have that in front of me"
- Offer to find out: "Let me check with the warehouse"
- Suggest alternatives based on what you DO know

# REMEMBER
You're having a conversation, not filling out a form or reading a catalog. Be helpful, be real, be Alex. Think: "How would I explain this to a friend who needs my help buying metal?"

Less like: "I am pleased to inform you that we have inventory available"
More like: "Yeah, we've got that!"

Less like: "Could you provide additional details regarding your requirements?"
More like: "What do you need it for?"

Less like: "The product specifications are as follows"
More like: "So it's 2mm thick, comes in sheets..."

Stay natural, stay helpful, stay human.

# QUOTATION Conversation flow

**When users ask for quotations or pricing:**
1. Recognize quotation requests: "quote", "quotation", "price", "cost", "how much", "pricing"
2. For detailed quotes requiring follow-up, collect contact information naturally

**Quotation conversation flow:**
- Acknowledge: "Sure, I can get you a detailed quote for that!"
- Explain value: "I'll send you the complete specifications and pricing directly"
- Ask naturally: "What's your name and email so I can send over the details?"
- Confirm: "Thanks [name]! Just to confirm, your email is [email]?"

**Examples:**

**Customer: "Can you please share the quotation of SS316 of 10mm thickness 20 sheets?"**
✅ Good: "I can certainly share the details for the SS316 10mm 20 sheets product. To send you the detailed specifications and a quote directly, may I please have your name and email address?"

**After they provide info:**
✅ Good: "Thank you. Just to confirm, your email is [user's email]?"

**Keep it natural:**
- Don't make it sound like a form
- Explain why you need the info
- Confirm details to ensure accuracy
- Continue the conversation naturally after collecting info

# CONFIGURATION NOTES
- Response temperature: {self.temperature}
- Max response length: {self.max_tokens} tokens
- Model: {self.model}
- Timeout: {self.timeout} seconds
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

        self.customer_db = customer_db
        
        logger.info(f"🚀 OpenAI LLM initialized:")
        logger.info(f"   Model: {self.model}")
        logger.info(f"   Timeout: {self.timeout}s")
        logger.info(f"   Max tokens: {self.max_tokens}")
        logger.info(f"   Temperature: {self.temperature}")
        logger.info(f"   Max history: {self.max_conversation_turns} turns")
        logger.info(f"   Product Search: {'Enabled' if product_search else 'Disabled'}")
        logger.info(f"   Config Source: config.json")
    
    async def _search_relevant_products(self, user_message: str) -> Optional[str]:
        """
        Search for products relevant to the user query
        Returns formatted product information for the prompt
        """
        if not self.product_search:
            logger.info("❌ Product search disabled")
            return None
        
        try:
            # Get config for product search settings
            config = config_manager.get_config()
            
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
            
            # Search for relevant products using config values
            logger.info(f"🔍 Searching products for: '{user_message}'")
            relevant_products = await self.product_search.search_products(
                user_message, 
                top_k=config.product_search.top_k,  # UPDATED: Use config
                similarity_threshold=config.product_search.similarity_threshold  # UPDATED: Use config
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
    
    def _build_messages(self, user_message: str, product_context: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Build messages for OpenAI API with conversation history and product context
        """
        messages = []
        
        # Add system message with product context if available
        system_content = self.system_prompt
        if product_context:
            system_content = system_content + "\n\n" + product_context
        
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation history
        for turn in self.conversation_history[-self.max_conversation_turns:]:
            messages.append({
                "role": turn['role'],
                "content": turn['content']
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _load_conversation_history(self):
        """Load conversation history from JSON file"""
        try:
            if self.conversation_file_path.exists():
                with open(self.conversation_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Load only the conversation_history array
                    if isinstance(data, dict) and 'conversations' in data:
                        # If we have a sessions-based structure, load the last session
                        if data['conversations']:
                            last_session = data['conversations'][-1]
                            self.conversation_history = last_session.get('messages', [])
                            logger.info(f"📂 Loaded {len(self.conversation_history)} messages from previous session")
                    elif isinstance(data, list):
                        # If it's a simple list of messages
                        self.conversation_history = data
                        logger.info(f"📂 Loaded {len(self.conversation_history)} messages from conversation file")
            else:
                logger.info(f"📝 No existing conversation file found. Starting fresh.")
        except Exception as e:
            logger.warning(f"⚠️ Could not load conversation history: {e}")
            self.conversation_history = []

    def _save_conversation_history(self):
        """Save conversation history to JSON file"""
        try:
            # Create the directory if it doesn't exist
            self.conversation_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare the data structure
            data = {
                'session_id': self.session_id,
                'started_at': self.conversation_history[0]['timestamp'] if self.conversation_history else datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'model': self.model,
                'total_turns': len(self.conversation_history) // 2,
                'conversations': [
                    {
                        'session_id': self.session_id,
                        'messages': self.conversation_history
                    }
                ]
            }

            # If file exists, append to it as a new session
            if self.conversation_file_path.exists():
                try:
                    with open(self.conversation_file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, dict) and 'conversations' in existing_data:
                            # Check if current session already exists
                            session_exists = False
                            for conv in existing_data['conversations']:
                                if conv.get('session_id') == self.session_id:
                                    # Update existing session
                                    conv['messages'] = self.conversation_history
                                    session_exists = True
                                    break

                            if not session_exists:
                                # Add new session
                                existing_data['conversations'].append({
                                    'session_id': self.session_id,
                                    'messages': self.conversation_history
                                })

                            existing_data['last_updated'] = datetime.now().isoformat()
                            data = existing_data
                except Exception as e:
                    logger.warning(f"⚠️ Could not read existing file, creating new: {e}")

            # Write to file with pretty formatting
            with open(self.conversation_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Saved conversation to {self.conversation_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save conversation history: {e}")

    def _add_to_conversation(self, role: str, content: str):
        """Add a message to conversation history and save to file"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })

        # Trim history if it gets too long
        if len(self.conversation_history) > self.max_conversation_turns * 2:
            self.conversation_history = self.conversation_history[-self.max_conversation_turns * 2:]

        # Save to file after each message
        self._save_conversation_history()
    
    def _extract_basic_info(self, user_message: str) -> tuple:
        """
        Extract basic customer info from message with better natural language parsing
        Returns: (customer_name, customer_email, product_requested)
        """
        import re
        
        customer_name = None
        customer_email = None
        product_requested = None
        
        # Convert to lowercase for easier matching
        message_lower = user_message.lower()
        
        # Improved name extraction
        name_patterns = [
            r'my name is (\w+)',
            r'name is (\w+)',
            r'i am (\w+)',
            r'this is (\w+)',
            r'call me (\w+)',
            r"i'm (\w+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                customer_name = match.group(1).title()  # Capitalize first letter
                break
        
        # Improved email extraction with common spoken patterns
        email_patterns = [
            r'(\w+@\w+\.\w+)',  # Standard email format
            r'(\w+)\s*at\s*(\w+)\s*dot\s*(\w+)',  # "nihal at gmail dot com"
            r'(\w+)\s*@\s*(\w+)\s*\.\s*(\w+)',  # "nihal @ gmail . com"
        ]
        
        # First try standard email format
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
        if email_match:
            customer_email = email_match.group().lower()
        else:
            # Try spoken patterns like "at" and "dot"
            for pattern in email_patterns[1:]:  # Skip first pattern since we already tried it
                match = re.search(pattern, message_lower)
                if match:
                    # Reconstruct email from "at" and "dot" pattern
                    if len(match.groups()) == 3:
                        username, domain, tld = match.groups()
                        customer_email = f"{username}@{domain}.{tld}".lower()
                        break
                    elif len(match.groups()) == 1:
                        customer_email = match.group(1).lower()
                        break
        
        # Improved product extraction
        product_keywords = {
            'aluminum': ['aluminum', 'aluminium', 'al sheet', 'al plate'],
            'stainless steel': ['steel', 'stainless steel', 'ss', 'stainless'],
            'sheet': ['sheet', 'sheets'],
            'plate': ['plate', 'plates'],
            'pipe': ['pipe', 'pipes', 'tube', 'tubes'],
            'rod': ['rod', 'rods', 'bar', 'bars']
        }
        
        # Check for specific product mentions
        for product, keywords in product_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                product_requested = product
                break
        
        # Log what we extracted for debugging
        logger.info(f"🔍 Extracted info - Name: {customer_name}, Email: {customer_email}, Product: {product_requested}")
        
        return customer_name, customer_email, product_requested

    async def generate_response(self, user_message: str) -> Dict[str, Any]:
        """Generate a response for the user message using OpenAI API"""
        start_time = datetime.now()
        self.stats['requests_sent'] += 1
        
        try:
            # Add user message to conversation history
            self._add_to_conversation('user', user_message)
            
            # Search for relevant products
            product_context = await self._search_relevant_products(user_message)
            
            # Build messages for OpenAI API
            messages = self._build_messages(user_message, product_context)
            
            logger.info(f"💬 Sending request to OpenAI: '{user_message}'")
            
            # Make request to OpenAI API with config values
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,  # UPDATED: Use config
                temperature=self.temperature,  # UPDATED: Use config
                timeout=self.timeout,
            )
            
            # Extract response text
            
            response_text = response.choices[0].message.content.strip()
            
            if not response_text:
                response_text = "I don't have a response for that right now."
            
            # Add assistant response to conversation history
            self._add_to_conversation('assistant', response_text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['responses_received'] += 1
            self.stats['total_processing_time'] += processing_time

            # STORE IN DATABASE - WITH PROPER ERROR HANDLING
            config = config_manager.get_config()
            if (self.customer_db and self.customer_db.is_connected and 
                config.features.enable_database):  # UPDATED: Check config
                try:
                    customer_name, customer_email, product_requested = self._extract_basic_info(user_message)
                    await self.customer_db.store_customer_query(
                        full_query_text=user_message,
                        customer_name=customer_name,
                        customer_email=customer_email,
                        product_requested=product_requested
                    )
                    logger.info(f"💾 Stored customer query in database")
                except Exception as db_error:
                    logger.error(f"❌ Failed to store in database: {db_error}")
            else:
                logger.warning("⚠️ Database not available or disabled - skipping storage")
            
            logger.info(f"🤖 OpenAI Response: '{response_text}'")
            logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
            
            return {
                'text': response_text,
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'processing_time': processing_time,
                'success': True,
                'conversation_turns': len(self.conversation_history) // 2,
                'products_used_in_context': self.stats['products_found_in_context'],
                'product_search_performed': self.stats['product_searches_performed'] > 0,
                'database_stored': (self.customer_db and self.customer_db.is_connected and 
                                  config.features.enable_database),
                'openai_usage': {
                    'total_tokens': response.usage.total_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                } if hasattr(response, 'usage') and response.usage else None
            }
            
        except asyncio.TimeoutError:
            logger.error("⏰ Request to OpenAI API timed out")
            self.stats['errors'] += 1
            
            return {
                'text': "I'm taking a bit too long to respond. Please try again with a shorter message.",
                'timestamp': datetime.now().isoformat(),
                'model': 'timeout',
                'processing_time': self.timeout,
                'success': False,
                'error': 'Request timeout'
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            self.stats['errors'] += 1
            
            error_message = str(e).lower()
            if "authentication" in error_message or "api key" in error_message:
                error_text = "There's an issue with my API configuration. Please contact support."
            elif "rate limit" in error_message:
                error_text = "I'm receiving too many requests right now. Please try again in a moment."
            elif "quota" in error_message:
                error_text = "My API quota has been exceeded. Please try again later."
            else:
                error_text = "I encountered an error while processing your request. Please try again."
            
            return {
                'text': error_text,
                'timestamp': datetime.now().isoformat(),
                'model': 'error',
                'processing_time': 0,
                'success': False,
                'error': str(e)
            }
    
    async def test_connection(self) -> bool:
        """
        Test connection to the OpenAI API
        """
        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'Hello'"}],
                max_tokens=10,
                timeout=5
            )
            return response.choices[0].message.content is not None
            
        except Exception as e:
            logger.warning(f"⚠️ Connection test failed: {e}")
            return False
    
    def clear_conversation_history(self):
        """Clear the conversation history and save to file"""
        previous_turns = len(self.conversation_history) // 2
        self.conversation_history.clear()

        # Create new session ID for fresh start
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the cleared state
        self._save_conversation_history()

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
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout,
            'provider': 'OpenAI',
            'config_source': 'config.json'
        }
    
    async def close(self):
        """Close the OpenAI client"""
        await self.client.close()
        logger.info("🔌 OpenAI client closed")


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
    """Test the enhanced RAG system with OpenAI"""
    
    # Initialize product search first
    from productsearch import ProductSearch
    product_search = ProductSearch()
    await product_search.initialize()
    
    # Initialize LLM with product search
    llm = RemoteGPTLLM(
        api_key=os.getenv("OPENAI_API_KEY"),
        product_search=product_search
    )
    
    try:
        # Test connection
        print("Testing connection to OpenAI API...")
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