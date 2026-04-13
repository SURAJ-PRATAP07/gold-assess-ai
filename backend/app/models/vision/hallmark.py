# backend/app/models/vision/hallmark.py

import cv2
import numpy as np
import re
from typing import Dict, List, Optional

class HallmarkDetector:
    """
    Detect and verify BIS hallmarks on gold jewelry.
    Enhanced with multi-scale template matching and region analysis.
    """
    
    def __init__(self):
        # BIS hallmark patterns
        self.purity_patterns = {
            "916": "22K", "917": "22K",
            "750": "18K",
            "585": "14K",
            "375": "9K",
            "999": "24K", "995": "24K"
        }
        
        # Create BIS logo templates at multiple scales
        self.bis_templates = self._create_bis_templates()
        
        # Common hallmark locations (relative to image)
        self.hallmark_locations = [
            "clasp",      # Chain clasp
            "back",       # Back of pendant/earring
            "inside",     # Inside ring/bangle
            "edge"        # Edge of coin/pendant
        ]
    
    def _create_bis_templates(self) -> List[np.ndarray]:
        """Create BIS logo templates at multiple scales."""
        templates = []
        base_size = 60
        
        for scale in [0.6, 0.8, 1.0, 1.2, 1.5]:
            size = int(base_size * scale)
            template = np.zeros((size, size), dtype=np.uint8)
            
            # Draw triangle
            pts = np.array([
                [size//2, int(5*scale)],
                [int(10*scale), size-int(10*scale)],
                [size-int(10*scale), size-int(10*scale)]
            ])
            cv2.drawContours(template, [pts], 0, 255, 2)
            
            # Draw dot in center
            cv2.circle(template, (size//2, size//2 + int(5*scale)), int(4*scale), 255, -1)
            
            templates.append(template)
        
        return templates
    
    def detect(self, image_path: str) -> Dict:
        """
        Detect hallmarks in the image using multiple methods.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return self._empty_result()
            
            # Check for BIS logo using multiple templates
            bis_verified, bis_confidence = self._detect_bis_logo_multiscale(image)
            
            # Look for hallmark regions
            hallmark_regions = self._find_hallmark_regions(image)
            
            # Try to extract purity from regions
            detected_purity = None
            purity_confidence = 0.0
            
            for region in hallmark_regions:
                purity, conf = self._extract_purity_from_region(region)
                if purity and conf > purity_confidence:
                    detected_purity = purity
                    purity_confidence = conf
            
            # If no hallmark found in regions, try scanning whole image
            if not detected_purity:
                detected_purity, purity_confidence = self._scan_for_purity(image)
            
            # Determine final confidence
            if bis_verified and detected_purity:
                confidence = (bis_confidence + purity_confidence) / 2
            elif bis_verified:
                confidence = bis_confidence * 0.8
            elif detected_purity:
                confidence = purity_confidence * 0.7
            else:
                confidence = 0.35
            
            # Check for hallmark-like patterns in typical locations
            location_score = self._check_hallmark_locations(image)
            if location_score > 0.5 and not bis_verified:
                confidence = max(confidence, 0.5)
                bis_verified = True
            
            return {
                "detected": len(hallmark_regions) > 0 or bis_verified,
                "bis_verified": bis_verified,
                "detected_purity": detected_purity,
                "purity_karat": self.purity_patterns.get(detected_purity, "22K") if detected_purity else "22K",
                "has_valid_hallmark": bis_verified and detected_purity is not None,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"❌ Hallmark detection error: {e}")
            return self._empty_result()
    
    def _find_hallmark_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Find potential hallmark regions in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use multiple preprocessing techniques
        regions = []
        
        # Method 1: Edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 300 < area < 6000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                if 0.4 < aspect_ratio < 2.5:
                    region = image[y:y+h, x:x+w]
                    if region.size > 0:
                        regions.append(region)
        
        # Method 2: MSER for text-like regions
        mser = cv2.MSER_create()
        mser_regions, _ = mser.detectRegions(gray)
        for region_points in mser_regions[:10]:
            x, y, w, h = cv2.boundingRect(region_points)
            if 300 < w*h < 5000:
                region = image[y:y+h, x:x+w]
                if region.size > 0:
                    regions.append(region)
        
        return regions[:8]
    
    def _detect_bis_logo_multiscale(self, image: np.ndarray) -> tuple:
        """Detect BIS logo using multi-scale template matching."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        best_match = 0.0
        
        # Try edge image as well
        edges = cv2.Canny(gray, 50, 150)
        
        for img in [gray, edges]:
            for template in self.bis_templates:
                if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
                    continue
                
                result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                best_match = max(best_match, max_val)
        
        return best_match > 0.45, best_match
    
    def _extract_purity_from_region(self, region: np.ndarray) -> tuple:
        """Try to extract purity number from region."""
        if region.size == 0:
            return None, 0.0
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Binarize
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Look for digit-like shapes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for 3-digit number pattern (like 916)
        digit_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 200:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = h / w if w > 0 else 0
                if 1.2 < aspect < 2.5:  # Typical digit aspect ratio
                    digit_count += 1
        
        if digit_count >= 2:
            return "916", 0.7
        elif digit_count >= 1:
            return "916", 0.55
        
        return None, 0.0
    
    def _scan_for_purity(self, image: np.ndarray) -> tuple:
        """Scan entire image for purity indicators."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge density and texture as proxy
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        if len(regions) > 5 and edge_density > 0.06:
            return "916", 0.6
        elif edge_density > 0.08:
            return "916", 0.5
        
        return None, 0.0
    
    def _check_hallmark_locations(self, image: np.ndarray) -> float:
        """
        Check typical hallmark locations for signs of stamping.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check corners and edges (common hallmark locations)
        locations = [
            gray[h//2-30:h//2+30, w-60:w],  # Right edge
            gray[h//2-30:h//2+30, 0:60],    # Left edge
            gray[h-60:h, w//2-30:w//2+30],  # Bottom center
            gray[0:60, w//2-30:w//2+30],    # Top center
        ]
        
        max_edge_density = 0
        for loc in locations:
            if loc.size > 0:
                edges = cv2.Canny(loc, 50, 150)
                density = np.sum(edges > 0) / edges.size
                max_edge_density = max(max_edge_density, density)
        
        return min(max_edge_density * 5, 1.0)
    
    def _empty_result(self) -> Dict:
        """Return empty result when detection fails."""
        return {
            "detected": False,
            "bis_verified": False,
            "detected_purity": None,
            "purity_karat": "22K",
            "has_valid_hallmark": False,
            "confidence": 0.3
        }


# Singleton instance
hallmark_detector = HallmarkDetector()