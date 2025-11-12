# store customer quotations in a simple postgres database (customer name,email,quotation text,timestamp)


import os
import asyncpg
from datetime import datetime
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerDatabase:
    """
    Simple database manager for storing only essential customer data
    """
    
    def __init__(self):
        self.connection_pool = None
        self.database_url = os.getenv("DATABASE_URL")
        self.is_connected = False
        
    async def initialize(self):
        """Initialize database connection pool with better error handling"""
        try:
            if not self.database_url:
                logger.warning("❌ DATABASE_URL not found in environment variables")
                return False
            
            # Test both localhost and 127.0.0.1
            test_urls = [
                self.database_url,
                self.database_url.replace('localhost', '127.0.0.1')
            ]
            
            for test_url in test_urls:
                try:
                    logger.info(f"🔧 Testing connection to: {test_url.split('@')[1] if '@' in test_url else test_url}")
                    
                    self.connection_pool = await asyncpg.create_pool(
                        test_url,
                        min_size=1,
                        max_size=10,
                        command_timeout=30
                    )
                    
                    # Test the connection
                    async with self.connection_pool.acquire() as conn:
                        await conn.execute('SELECT 1')
                    
                    self.database_url = test_url  # Use the working URL
                    self.is_connected = True
                    
                    # Create simple table
                    await self._create_tables()
                    logger.info("✅ Database initialized successfully")
                    return True
                    
                except Exception as e:
                    logger.warning(f"⚠️ Connection failed for {test_url}: {e}")
                    if self.connection_pool:
                        await self.connection_pool.close()
                        self.connection_pool = None
                    continue
            
            logger.error("❌ All database connection attempts failed")
            return False
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            self.is_connected = False
            return False
    
    async def _create_tables(self):
        """Create simple table for customer data"""
        async with self.connection_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS customer_queries (
                    query_id SERIAL PRIMARY KEY,
                    customer_name TEXT,
                    customer_email TEXT,
                    product_requested TEXT,
                    full_query_text TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            logger.info("✅ Database table created/verified")
    
    async def store_customer_query(
        self,
        full_query_text: str,
        customer_name: Optional[str] = None,
        customer_email: Optional[str] = None,
        product_requested: Optional[str] = None
    ) -> Optional[int]:
        """
        Store only the essential customer query data
        """
        if not self.is_connected or not self.connection_pool:
            logger.warning("❌ Database not connected - skipping storage")
            return None
            
        try:
            async with self.connection_pool.acquire() as conn:
                query_id = await conn.fetchval('''
                    INSERT INTO customer_queries (
                        customer_name,
                        customer_email,
                        product_requested,
                        full_query_text
                    ) VALUES ($1, $2, $3, $4)
                    RETURNING query_id
                ''', 
                customer_name,
                customer_email,
                product_requested,
                full_query_text)
                
                logger.info(f"✅ Stored customer query with ID: {query_id}")
                return query_id
                
        except Exception as e:
            logger.error(f"❌ Failed to store customer query: {e}")
            return None
    
    async def close(self):
        """Close database connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            self.is_connected = False
            logger.info("🔌 Database connection pool closed")

# Global database instance
customer_db = CustomerDatabase()