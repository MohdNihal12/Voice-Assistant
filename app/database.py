import os
import asyncpg
from datetime import datetime
from typing import Optional
import logging
from dotenv import load_dotenv

# NEW: Import config manager
from app.config_manager import config_manager

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomerDatabase:
    """
    Simple database manager for storing only essential customer data with config integration
    """
    
    def __init__(self):
        self.connection_pool = None
        self.database_url = os.getenv("DATABASE_URL")
        self.is_connected = False
        
    async def initialize(self):
        """Initialize database connection pool with config-based feature flags"""
        try:
            # Get config values
            config = config_manager.get_config()
            
            # Check if database is enabled in config
            if not config.features.enable_database:
                logger.info("⏭️ Database disabled in configuration")
                self.is_connected = False
                return True  # Return True since it's intentionally disabled
            
            if not self.database_url:
                logger.warning("❌ DATABASE_URL not found in environment variables")
                self.is_connected = False
                return False
            
            # Test both localhost and 127.0.0.1
            test_urls = [
                self.database_url,
                self.database_url.replace('localhost', '127.0.0.1'),
                self.database_url.replace('127.0.0.1', 'localhost')
            ]
            
            for test_url in test_urls:
                try:
                    # Mask password in logs for security
                    safe_url = self._mask_database_url(test_url)
                    logger.info(f"🔧 Testing database connection: {safe_url}")
                    
                    self.connection_pool = await asyncpg.create_pool(
                        test_url,
                        min_size=1,
                        max_size=config.database.max_connections,  # UPDATED: Use config
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
                    logger.info(f"⚙️ Config: Max connections: {config.database.max_connections}, Enabled: {config.features.enable_database}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"⚠️ Connection failed for {self._mask_database_url(test_url)}: {e}")
                    if self.connection_pool:
                        await self.connection_pool.close()
                        self.connection_pool = None
                    continue
            
            logger.error("❌ All database connection attempts failed")
            self.is_connected = False
            return False
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            self.is_connected = False
            return False
    
    def _mask_database_url(self, url: str) -> str:
        """Mask password in database URL for secure logging"""
        try:
            if '@' in url:
                # Mask password: postgresql://user:password@host:port/database
                parts = url.split('@')
                auth_part = parts[0]
                if ':' in auth_part and '//' in auth_part:
                    protocol, credentials = auth_part.split('//')
                    if ':' in credentials:
                        user, password = credentials.split(':', 1)
                        masked_auth = f"{protocol}//{user}:****@{parts[1]}"
                        return masked_auth
            return url
        except:
            return "***masked***"
    
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
            
            # Create index for better performance
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_customer_queries_timestamp 
                ON customer_queries(timestamp DESC)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_customer_queries_email 
                ON customer_queries(customer_email) WHERE customer_email IS NOT NULL
            ''')
            
            logger.info("✅ Database tables and indexes created/verified")
    
    async def store_customer_query(
        self,
        full_query_text: str,
        customer_name: Optional[str] = None,
        customer_email: Optional[str] = None,
        product_requested: Optional[str] = None
    ) -> Optional[int]:
        """
        Store only the essential customer query data with config check
        """
        # Check if database is enabled in config
        config = config_manager.get_config()
        if not config.features.enable_database:
            logger.debug("⏭️ Database disabled - skipping storage")
            return None
            
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
    
    async def get_recent_queries(self, limit: int = 10) -> list:
        """Get recent customer queries"""
        config = config_manager.get_config()
        if not config.features.enable_database:
            logger.debug("⏭️ Database disabled - cannot retrieve queries")
            return []
            
        if not self.is_connected or not self.connection_pool:
            return []
            
        try:
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch('''
                    SELECT query_id, customer_name, customer_email, 
                           product_requested, full_query_text, timestamp
                    FROM customer_queries 
                    ORDER BY timestamp DESC 
                    LIMIT $1
                ''', limit)
                
                queries = []
                for row in rows:
                    queries.append(dict(row))
                
                logger.info(f"📊 Retrieved {len(queries)} recent queries")
                return queries
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve queries: {e}")
            return []
    
    async def get_query_stats(self) -> dict:
        """Get database statistics"""
        config = config_manager.get_config()
        if not config.features.enable_database:
            return {
                'enabled': False,
                'message': 'Database disabled in configuration'
            }
            
        if not self.is_connected or not self.connection_pool:
            return {
                'enabled': True,
                'connected': False,
                'message': 'Database not connected'
            }
            
        try:
            async with self.connection_pool.acquire() as conn:
                total_queries = await conn.fetchval('SELECT COUNT(*) FROM customer_queries')
                today_queries = await conn.fetchval('''
                    SELECT COUNT(*) FROM customer_queries 
                    WHERE timestamp >= CURRENT_DATE
                ''')
                unique_emails = await conn.fetchval('''
                    SELECT COUNT(DISTINCT customer_email) FROM customer_queries 
                    WHERE customer_email IS NOT NULL
                ''')
                
                return {
                    'enabled': True,
                    'connected': True,
                    'total_queries': total_queries,
                    'queries_today': today_queries,
                    'unique_emails': unique_emails,
                    'config_source': 'config.json',
                    'max_connections': config.database.max_connections
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to get query stats: {e}")
            return {
                'enabled': True,
                'connected': True,
                'error': str(e)
            }
    
    async def test_connection(self) -> bool:
        """Test database connection"""
        config = config_manager.get_config()
        if not config.features.enable_database:
            return False
            
        if not self.is_connected or not self.connection_pool:
            return False
            
        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchval('SELECT 1')
                return result == 1
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    async def close(self):
        """Close database connection pool"""
        config = config_manager.get_config()
        if not config.features.enable_database:
            return
            
        if self.connection_pool:
            await self.connection_pool.close()
            self.is_connected = False
            logger.info("🔌 Database connection pool closed")

# Global database instance
customer_db = CustomerDatabase()


# Example usage and testing
async def main():
    """Test the database system"""
    db = CustomerDatabase()
    
    # Initialize database
    success = await db.initialize()
    if success:
        print("✅ Database initialized successfully")
        
        # Test storing a query
        query_id = await db.store_customer_query(
            full_query_text="Hello, I need stainless steel sheets",
            customer_name="John Doe",
            customer_email="john@example.com",
            product_requested="Stainless Steel"
        )
        
        if query_id:
            print(f"✅ Stored query with ID: {query_id}")
        
        # Get recent queries
        recent_queries = await db.get_recent_queries(5)
        print(f"📊 Recent queries: {len(recent_queries)}")
        
        # Get statistics
        stats = await db.get_query_stats()
        print(f"📈 Database stats: {stats}")
        
        # Test connection
        connection_ok = await db.test_connection()
        print(f"🔗 Connection test: {'✅ OK' if connection_ok else '❌ Failed'}")
        
        # Close connection
        await db.close()
    else:
        print("❌ Database initialization failed")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())