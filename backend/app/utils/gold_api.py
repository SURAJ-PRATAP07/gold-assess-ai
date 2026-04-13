# backend/app/utils/gold_api.py

import httpx
from datetime import datetime
from typing import Dict, Optional

class GoldPriceAPI:
    """
    Get real-time gold prices from public APIs.
    """
    
    def __init__(self):
        # Free gold price API (no key required)
        self.api_url = "https://api.gold-api.com/price/XAU"
        
        # Fallback prices if API fails (INR per gram)
        self.fallback_prices = {
            "24K": 6850,
            "22K": 6250,
            "18K": 5120,
            "14K": 3980,
            "10K": 2850
        }
    
    async def get_current_prices(self) -> Dict:
        """
        Get current gold prices for different karats.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.api_url, timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    # API returns price per ounce, convert to INR per gram
                    price_per_ounce_usd = data.get("price", 2000)
                    
                    # Get USD to INR rate
                    usd_inr = await self._get_usd_inr_rate()
                    
                    # Convert to INR per gram (1 ounce = 31.1035 grams)
                    price_per_gram_24k = (price_per_ounce_usd * usd_inr) / 31.1035
                    
                    return self._calculate_all_karats(price_per_gram_24k)
                else:
                    return self.fallback_prices
                    
        except Exception as e:
            print(f"Gold API error: {e}")
            return self.fallback_prices
    
    async def _get_usd_inr_rate(self) -> float:
        """Get USD to INR exchange rate."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.exchangerate-api.com/v4/latest/USD",
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("rates", {}).get("INR", 83.0)
        except:
            pass
        return 83.0  # Default fallback
    
    def _calculate_all_karats(self, price_24k_per_gram: float) -> Dict:
        """Calculate prices for all karat levels."""
        return {
            "24K": round(price_24k_per_gram),
            "22K": round(price_24k_per_gram * 0.916),
            "18K": round(price_24k_per_gram * 0.75),
            "14K": round(price_24k_per_gram * 0.585),
            "10K": round(price_24k_per_gram * 0.417),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_price_for_karat(self, karat: str, prices: Optional[Dict] = None) -> int:
        """Get price for specific karat."""
        if prices is None:
            prices = self.fallback_prices
        
        # Normalize karat string
        karat = karat.upper().replace("K", "") + "K"
        return prices.get(karat, prices.get("22K", 6250))

# Singleton instance
gold_api = GoldPriceAPI()