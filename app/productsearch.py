import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductSearch:
    """
    Product search using SentenceTransformers for semantic similarity
    """
    
    def __init__(self, product_file: str = "data/product.json", model_name: str = "all-MiniLM-L6-v2"):
        self.product_file = product_file
        self.model_name = model_name
        self.products = []
        self.product_embeddings = None
        self.model = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the product search system"""
        try:
            # Load products
            await self._load_products()
            
            # Initialize sentence transformer model
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Generate embeddings for all products
            await self._generate_embeddings()
            
            self._initialized = True
            logger.info(f"✅ ProductSearch initialized with {len(self.products)} products")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ProductSearch: {e}")
            raise
    
    async def _load_products(self):
        """Load products from JSON file"""
        try:
            with open(self.product_file, 'r', encoding='utf-8') as f:
                self.products = json.load(f)
            logger.info(f"📦 Loaded {len(self.products)} products from {self.product_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load products: {e}")
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
        self.product_embeddings = self.model.encode(product_texts, show_progress_bar=True)
        logger.info(f"✅ Generated embeddings for {len(product_texts)} products")
    
    def _product_to_text(self, product: Dict[str, Any]) -> str:
        """Convert product data to searchable text"""
        text_parts = []
        
        # Basic product info
        text_parts.append(f"Product: {product.get('product', '')}")
        text_parts.append(f"Category: {product.get('category', '')}")
        text_parts.append(f"Grade: {product.get('grade', '')}")
        text_parts.append(f"ID: {product.get('id', '')}")
        
        # Specifications
        if 'temper' in product:
            text_parts.append(f"Temper: {product['temper']}")
        if 'thickness' in product:
            text_parts.append(f"Thickness: {product['thickness']}")
        if 'width' in product:
            text_parts.append(f"Width: {product['width']}")
        if 'length' in product:
            text_parts.append(f"Length: {product['length']}")
        if 'finish' in product:
            text_parts.append(f"Finish: {product['finish']}")
        
        # Applications
        if 'applications' in product:
            text_parts.append(f"Applications: {product['applications']}")
        
        # Pricing
        if 'price_per_kg' in product:
            text_parts.append(f"Price per kg: {product['price_per_kg']}")
        if 'price_per_sheet' in product:
            text_parts.append(f"Price per sheet: {product['price_per_sheet']}")
        
        # Specifications
        if 'specifications' in product:
            specs = product['specifications']
            for key, value in specs.items():
                text_parts.append(f"{key}: {value}")
        
        return " ".join(text_parts)
    
    async def search_products(self, query: str, top_k: int = 3, similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search for products similar to the query
        
        Args:
            query: User query text
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of product dictionaries with similarity scores
        """
        if not self._initialized:
            raise RuntimeError("ProductSearch not initialized. Call initialize() first.")
        
        if not self.products or self.product_embeddings is None:
            logger.warning("No products available for search")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.product_embeddings)[0]
        
        # Get top-k results
        results = []
        for idx, similarity in enumerate(similarities):
            if similarity >= similarity_threshold:
                product_copy = self.products[idx].copy()
                product_copy['similarity_score'] = float(similarity)
                results.append((similarity, product_copy))
        
        # Sort by similarity and take top-k
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = [product for _, product in results[:top_k]]
        
        logger.info(f"🔍 Search: '{query}' -> Found {len(top_results)} products (threshold: {similarity_threshold})")
        
        return top_results
    
    async def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID"""
        for product in self.products:
            if product.get('id') == product_id:
                return product
        return None
    
    async def get_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all products in a category"""
        return [product for product in self.products if product.get('category', '').lower() == category.lower()]
    
    async def format_products_for_prompt(self, products: List[Dict[str, Any]]) -> str:
        """Format products for inclusion in LLM prompt"""
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
        return {
            'initialized': self._initialized,
            'products_loaded': len(self.products),
            'embeddings_generated': self.product_embeddings is not None,
            'model': self.model_name,
            'embedding_dimension': self.product_embeddings.shape[1] if self.product_embeddings is not None else 0
        }


# Async context manager support
class ProductSearchManager:
    def __init__(self, product_file: str = "product.json"):
        self.product_search = ProductSearch(product_file)
    
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