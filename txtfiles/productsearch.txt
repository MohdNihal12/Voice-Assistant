import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import logging
import os

# NEW: Import config manager
from app.config_manager import config_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductSearch:
    """
    Product search using SentenceTransformers for semantic similarity
    """
    
    def __init__(self, product_file: str = None, model_name: str = None):  # UPDATED: Use config by default
        # Get config values
        config = config_manager.get_config()
        
        # UPDATED: Use config values with fallbacks
        self.product_file = product_file or config.product_search.product_file
        self.model_name = model_name or "all-MiniLM-L6-v2"  # Default model
        self.products = []
        self.product_embeddings = None
        self.model = None
        self._initialized = False
        
        # Statistics
        self.stats = {
            'searches_performed': 0,
            'total_products_retrieved': 0,
            'average_similarity_score': 0.0,
            'failed_searches': 0
        }
        
    async def initialize(self):
        """Initialize the product search system"""
        try:
            # Check if product search is enabled
            config = config_manager.get_config()
            if not config.features.enable_product_search:
                logger.info("⏭️ Product search disabled in config")
                self._initialized = True
                return
            
            # Load products
            await self._load_products()
            
            # Initialize sentence transformer model
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Generate embeddings for all products
            await self._generate_embeddings()
            
            self._initialized = True
            logger.info(f"✅ ProductSearch initialized with {len(self.products)} products")
            logger.info(f"⚙️ Config: {self.product_file}, threshold: {config.product_search.similarity_threshold}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ProductSearch: {e}")
            self.stats['failed_searches'] += 1
            raise
    
    async def _load_products(self):
        """Load products from JSON file"""
        try:
            # Check if product file exists
            if not os.path.exists(self.product_file):
                logger.warning(f"📁 Product file not found: {self.product_file}")
                # Create a sample product file if it doesn't exist
                await self._create_sample_products()
                return
            
            with open(self.product_file, 'r', encoding='utf-8') as f:
                self.products = json.load(f)
            logger.info(f"📦 Loaded {len(self.products)} products from {self.product_file}")
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in product file: {e}")
            self.products = []
        except Exception as e:
            logger.error(f"❌ Failed to load products: {e}")
            self.products = []
    
    async def _create_sample_products(self):
        """Create sample products if product file doesn't exist"""
        try:
            sample_products = [
                {
                    "id": "SS304-SHEET",
                    "product": "Stainless Steel 304 Sheet",
                    "category": "Stainless Steel",
                    "grade": "304",
                    "thickness": "2mm",
                    "width": "1000mm",
                    "length": "2000mm",
                    "price_per_kg": "₹250",
                    "price_per_sheet": "₹4500",
                    "finish": "2B",
                    "applications": "Kitchen equipment, food processing, architectural",
                    "specifications": {
                        "composition": "18% Chromium, 8% Nickel",
                        "tensile_strength": "515 MPa",
                        "yield_strength": "205 MPa",
                        "corrosion_resistance": "Good"
                    },
                    "min_order_quantity": "10 sheets",
                    "delivery_time": "3-5 days"
                },
                {
                    "id": "AL5052-SHEET",
                    "product": "Aluminum 5052 Sheet",
                    "category": "Aluminum",
                    "grade": "5052",
                    "thickness": "3mm",
                    "width": "1200mm",
                    "length": "2400mm",
                    "price_per_kg": "₹320",
                    "price_per_sheet": "₹5200",
                    "finish": "Mill finish",
                    "applications": "Marine applications, automotive parts, electronic chassis",
                    "specifications": {
                        "composition": "2.5% Magnesium, 0.25% Chromium",
                        "tensile_strength": "228 MPa",
                        "yield_strength": "193 MPa",
                        "corrosion_resistance": "Excellent in marine environments"
                    },
                    "min_order_quantity": "5 sheets",
                    "delivery_time": "2-4 days"
                }
            ]
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.product_file), exist_ok=True)
            
            with open(self.product_file, 'w', encoding='utf-8') as f:
                json.dump(sample_products, f, indent=2, ensure_ascii=False)
            
            self.products = sample_products
            logger.info(f"📝 Created sample product file with {len(sample_products)} products at {self.product_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create sample products: {e}")
            self.products = []
    
    async def _generate_embeddings(self):
        """Generate embeddings for all products"""
        if not self.products:
            logger.warning("No products to generate embeddings for")
            return
        
        # Create text representations for each product
        product_texts = []
        for product in self.products:
            text_representation = self._product_to_text(product)
            product_texts.append(text_representation)
        
        # Generate embeddings
        logger.info("Generating product embeddings...")
        self.product_embeddings = self.model.encode(product_texts, show_progress_bar=False)  # UPDATED: Disable progress bar for cleaner logs
        logger.info(f"✅ Generated embeddings for {len(product_texts)} products")
    
    def _product_to_text(self, product: Dict[str, Any]) -> str:
        """Convert product data to searchable text with better natural language"""
        text_parts = []
        
        # Product name with variations
        product_name = product.get('product', '')
        product_id = product.get('id', '')
        grade = product.get('grade', '')
        
        text_parts.append(f"Product: {product_name}")
        text_parts.append(f"Also known as: {product_id} {grade}")
        text_parts.append(f"Product ID: {product_id}")
        text_parts.append(f"Category: {product.get('category', '')}")
        text_parts.append(f"Grade: {grade}")
        
        # Add natural language variants for common names
        if 'SS' in product_id or 'stainless' in product_name.lower():
            text_parts.append(f"Stainless steel grade {grade}")
            text_parts.append(f"SS {grade}")
        if 'AL' in product_id or 'aluminum' in product_name.lower():
            text_parts.append(f"Aluminum grade {grade}")
            text_parts.append(f"AL {grade}")
        
        # Add specifications
        if 'thickness' in product:
            text_parts.append(f"Thickness: {product['thickness']}")
        if 'width' in product:
            text_parts.append(f"Width: {product['width']}")
        if 'length' in product:
            text_parts.append(f"Length: {product['length']}")
        if 'temper' in product:
            text_parts.append(f"Temper: {product['temper']}")
        
        # Add applications and use cases
        if 'applications' in product:
            text_parts.append(f"Best for: {product['applications']}")
            text_parts.append(f"Used in: {product['applications']}")
            text_parts.append(f"Applications: {product['applications']}")
        
        # Add pricing information
        if 'price_per_kg' in product:
            text_parts.append(f"Price per kg: {product['price_per_kg']}")
        if 'price_per_sheet' in product:
            text_parts.append(f"Price per sheet: {product['price_per_sheet']}")
        
        # Add specifications
        if 'specifications' in product:
            specs = product['specifications']
            for key, value in specs.items():
                text_parts.append(f"{key}: {value}")
        
        return " ".join(text_parts)
    
    async def search_products(self, query: str, top_k: int = None, similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Search for products similar to the query
        
        Args:
            query: User query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of product dictionaries with similarity scores
        """
        # Get config values
        config = config_manager.get_config()
        
        # UPDATED: Use config values with parameter fallbacks
        top_k = top_k or config.product_search.top_k
        similarity_threshold = similarity_threshold or config.product_search.similarity_threshold
        
        # Check if product search is enabled
        if not config.features.enable_product_search:
            logger.info("⏭️ Product search disabled - returning empty results")
            return []
        
        if not self._initialized:
            raise RuntimeError("ProductSearch not initialized. Call initialize() first.")
        
        if not self.products or self.product_embeddings is None:
            logger.warning("No products available for search")
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]
            
            # Get results above threshold
            results = []
            for idx, similarity in enumerate(similarities):
                if similarity >= similarity_threshold:
                    product_copy = self.products[idx].copy()
                    product_copy['similarity_score'] = float(similarity)
                    results.append((similarity, product_copy))
            
            # Sort by similarity and take top-k
            results.sort(key=lambda x: x[0], reverse=True)
            top_results = [product for _, product in results[:top_k]]
            
            # Update statistics
            self.stats['searches_performed'] += 1
            self.stats['total_products_retrieved'] += len(top_results)
            if top_results:
                avg_score = sum(product['similarity_score'] for product in top_results) / len(top_results)
                self.stats['average_similarity_score'] = avg_score
            
            logger.info(f"🔍 Search: '{query}' -> Found {len(top_results)} products (threshold: {similarity_threshold}, top_k: {top_k})")
            
            return top_results
            
        except Exception as e:
            logger.error(f"❌ Search error for query '{query}': {e}")
            self.stats['failed_searches'] += 1
            return []
    
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID"""
        config = config_manager.get_config()
        if not config.features.enable_product_search:
            return None
            
        for product in self.products:
            if product.get('id') == product_id:
                return product
        return None
    
    async def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all products in a category"""
        config = config_manager.get_config()
        if not config.features.enable_product_search:
            return []
            
        return [product for product in self.products if product.get('category', '').lower() == category.lower()]
    
    async def format_products_for_prompt(self, products: List[Dict[str, Any]]) -> str:
        """Format products for inclusion in LLM prompt"""
        config = config_manager.get_config()
        if not config.features.enable_product_search:
            return "Product search is currently disabled."
            
        if not products:
            return "No relevant products found."
        
        formatted_products = []
        for i, product in enumerate(products, 1):
            product_text = f"PRODUCT {i}:\n"
            product_text += f"ID: {product.get('id', 'N/A')}\n"
            product_text += f"Product: {product.get('product', 'N/A')}\n"
            product_text += f"Category: {product.get('category', 'N/A')}\n"
            product_text += f"Grade: {product.get('grade', 'N/A')}\n"
            
            if 'temper' in product:
                product_text += f"Temper: {product['temper']}\n"
            if 'thickness' in product:
                product_text += f"Thickness: {product['thickness']}\n"
            if 'width' in product:
                product_text += f"Width: {product['width']}\n"
            if 'length' in product:
                product_text += f"Length: {product['length']}\n"
            if 'price_per_kg' in product:
                product_text += f"Price per kg: {product['price_per_kg']}\n"
            if 'price_per_sheet' in product:
                product_text += f"Price per sheet: {product['price_per_sheet']}\n"
            if 'finish' in product:
                product_text += f"Finish: {product['finish']}\n"
            if 'applications' in product:
                product_text += f"Applications: {product['applications']}\n"
            if 'specifications' in product:
                specs = product['specifications']
                product_text += "Specifications:\n"
                for key, value in specs.items():
                    product_text += f"  - {key}: {value}\n"
            
            if 'similarity_score' in product:
                product_text += f"Relevance Score: {product['similarity_score']:.2f}\n"
            
            formatted_products.append(product_text.strip())
        
        return "\n\n".join(formatted_products)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search system statistics"""
        config = config_manager.get_config()
        
        base_stats = {
            'initialized': self._initialized,
            'products_loaded': len(self.products),
            'embeddings_generated': self.product_embeddings is not None,
            'model': self.model_name,
            'embedding_dimension': self.product_embeddings.shape[1] if self.product_embeddings is not None else 0,
            'product_search_enabled': config.features.enable_product_search,
            'config_source': 'config.json'
        }
        
        # Add performance stats if searches have been performed
        if self.stats['searches_performed'] > 0:
            base_stats.update({
                'searches_performed': self.stats['searches_performed'],
                'total_products_retrieved': self.stats['total_products_retrieved'],
                'average_similarity_score': round(self.stats['average_similarity_score'], 3),
                'failed_searches': self.stats['failed_searches'],
                'average_products_per_search': round(self.stats['total_products_retrieved'] / self.stats['searches_performed'], 2)
            })
        
        # Add config values
        base_stats.update({
            'config_values': {
                'product_file': config.product_search.product_file,
                'top_k': config.product_search.top_k,
                'similarity_threshold': config.product_search.similarity_threshold
            }
        })
        
        return base_stats
    
    async def reload_products(self):
        """Reload products from file (useful for updates)"""
        logger.info("🔄 Reloading products...")
        await self._load_products()
        if self.products:
            await self._generate_embeddings()
        logger.info("✅ Products reloaded")


# Async context manager support
class ProductSearchManager:
    def __init__(self, product_file: str = None):
        config = config_manager.get_config()
        self.product_search = ProductSearch(product_file or config.product_search.product_file)
    
    async def __aenter__(self):
        await self.product_search.initialize()
        return self.product_search
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass


# Example usage
async def main():
    """Test the ProductSearch system"""
    search = ProductSearch()
    await search.initialize()
    
    # Test searches
    test_queries = [
        "aluminum sheets for marine applications",
        "stainless steel 304 price",
        "corrosion resistant materials",
        "thin metal sheets"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = await search.search_products(query, top_k=2)
        
        for i, product in enumerate(results, 1):
            print(f"  {i}. {product['product']} (Score: {product['similarity_score']:.3f})")
    
    print(f"\n📊 System stats: {search.get_stats()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())