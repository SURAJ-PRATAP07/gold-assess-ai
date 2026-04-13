# backend/app/models/vision/weight.py

import cv2
import numpy as np
from typing import Dict, Tuple

class WeightEstimator:
    """
    Estimate jewelry weight from images using computer vision.
    Enhanced with reference object detection and better segmentation.
    """
    
    def __init__(self):
        # Density of gold alloys (g/cm³)
        self.density = {
            "24K": 19.32,
            "22K": 17.45,
            "18K": 15.40,
            "14K": 13.07,
            "10K": 10.50
        }
        
        # Reference object sizes (mm)
        self.reference_sizes = {
            "10_rupee_coin": 27.0,
            "5_rupee_coin": 23.0,
            "2_rupee_coin": 25.0,
            "1_rupee_coin": 22.0,
            "finger": 16.0,
            "credit_card_width": 85.6
        }
        
        # Average weights by jewelry type (grams)
        self.avg_weights = {
            "chain": 20.0,
            "ring": 5.0,
            "bangle": 15.0,
            "bracelet": 12.0,
            "necklace": 25.0,
            "earring": 3.0,
            "pendant": 8.0,
            "mangalsutra": 30.0
        }
        
        # Thickness estimates by type (mm)
        self.thickness_map = {
            "chain": 1.5,
            "ring": 2.0,
            "bangle": 2.5,
            "bracelet": 2.0,
            "necklace": 1.8,
            "earring": 1.2,
            "pendant": 2.0,
            "mangalsutra": 2.5
        }
    
    def detect_reference_object(self, image: np.ndarray) -> Dict:
        """Detect reference objects for scale calibration."""
        h, w = image.shape[:2]
        
        # Look for coins (circular objects)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=100, param2=30, minRadius=15, maxRadius=120
        )
        
        if circles is not None and len(circles[0]) > 0:
            # Found circles - use the most prominent one
            circles = np.uint16(np.around(circles))
            best_circle = circles[0][0]
            radius = best_circle[2]
            
            # Determine coin type based on size
            if 40 < radius < 60:
                coin_type = "10_rupee_coin"
            elif 30 < radius < 45:
                coin_type = "5_rupee_coin"
            else:
                coin_type = "2_rupee_coin"
            
            size_mm = self.reference_sizes.get(coin_type, 25.0)
            
            return {
                "detected": True,
                "type": coin_type,
                "size_mm": size_mm,
                "radius_pixels": radius,
                "scale_mm_per_pixel": size_mm / (radius * 2),
                "confidence": 0.85
            }
        
        # Look for rectangular objects (cards)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5000 < area < 50000:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / ch if ch > 0 else 0
                if 1.4 < aspect < 1.8:  # Credit card aspect ratio
                    size_mm = self.reference_sizes["credit_card_width"]
                    return {
                        "detected": True,
                        "type": "credit_card",
                        "size_mm": size_mm,
                        "width_pixels": cw,
                        "scale_mm_per_pixel": size_mm / cw,
                        "confidence": 0.8
                    }
        
        return {"detected": False}
    
    def estimate(self, image_path: str, karat: str = "22K", jewelry_type: str = "chain") -> Dict:
        """
        Estimate weight from image with enhanced accuracy.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return self._fallback_estimate(karat, jewelry_type)
            
            # Detect reference object for scale
            reference = self.detect_reference_object(image)
            
            # Segment jewelry from background
            mask = self._segment_jewelry_enhanced(image)
            
            if mask is None or np.sum(mask) < 100:
                return self._fallback_estimate(karat, jewelry_type)
            
            # Get scale factor
            if reference["detected"]:
                scale_mm_per_pixel = reference["scale_mm_per_pixel"]
                scale_confidence = reference["confidence"]
                print(f"   ✓ Reference detected: {reference['type']}")
            else:
                # Estimate based on image dimensions
                h, w = image.shape[:2]
                scale_mm_per_pixel = 40.0 / max(h, w)
                scale_confidence = 0.5
                print(f"   ⚠️  No reference object - using estimated scale")
            
            # Calculate area in pixels
            pixel_area = np.sum(mask > 0)
            
            # Convert to real area (mm²)
            area_mm2 = pixel_area * (scale_mm_per_pixel ** 2)
            
            # Estimate thickness based on jewelry type and area
            thickness_mm = self._estimate_thickness(jewelry_type, area_mm2)
            
            # Calculate volume (mm³)
            # Shape correction based on type
            shape_corrections = {
                "chain": 0.55, "ring": 0.8, "bangle": 0.7,
                "bracelet": 0.65, "necklace": 0.6, "earring": 0.75,
                "pendant": 0.7, "mangalsutra": 0.65
            }
            shape_factor = shape_corrections.get(jewelry_type.lower(), 0.7)
            
            volume_mm3 = area_mm2 * thickness_mm * shape_factor
            
            # Convert to cm³
            volume_cm3 = volume_mm3 / 1000
            
            # Calculate weight
            density = self.density.get(karat, 17.45)
            weight_grams = volume_cm3 * density
            
            # Apply hollow adjustment
            is_hollow, hollow_confidence = self._detect_hollow_enhanced(mask, image)
            if is_hollow:
                weight_grams *= 0.65
                print(f"   ⚠️  Hollow detected - weight adjusted")
            
            # Ensure weight is reasonable for jewelry type
            avg_weight = self.avg_weights.get(jewelry_type.lower(), 15.0)
            if weight_grams > avg_weight * 3.5:
                weight_grams = avg_weight * 2.0
            elif weight_grams < avg_weight * 0.15:
                weight_grams = avg_weight * 0.4
            
            # Calculate uncertainty
            uncertainty = self._calculate_uncertainty_enhanced(
                scale_mm_per_pixel, mask, scale_confidence, 
                reference["detected"], is_hollow
            )
            
            # Overall confidence
            confidence = (1 - uncertainty) * scale_confidence
            if not reference["detected"]:
                confidence *= 0.8
            
            return {
                "estimated_volume_cm3": round(volume_cm3, 3),
                "estimated_weight_grams": round(weight_grams, 2),
                "weight_range": {
                    "min": round(weight_grams * (1 - uncertainty), 2),
                    "max": round(weight_grams * (1 + uncertainty), 2)
                },
                "is_hollow": is_hollow,
                "scale_mm_per_pixel": round(scale_mm_per_pixel, 4),
                "reference_detected": reference["detected"],
                "uncertainty_percent": round(uncertainty * 100, 1),
                "confidence": round(confidence, 2)
            }
            
        except Exception as e:
            print(f"❌ Weight estimation error: {e}")
            return self._fallback_estimate(karat, jewelry_type)
    
    def _segment_jewelry_enhanced(self, image: np.ndarray) -> np.ndarray:
        """Enhanced segmentation using GrabCut and morphological operations."""
        h, w = image.shape[:2]
        
        # Create initial mask
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Smart rectangle based on image content
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            # Expand slightly
            x = max(0, x - 10)
            y = max(0, y - 10)
            cw = min(w - x, cw + 20)
            ch = min(h - y, ch + 20)
            rect = (x, y, cw, ch)
        else:
            rect = (w//4, h//4, w//2, h//2)
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask
            jewelry_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            
            # Morphological cleanup
            kernel = np.ones((5,5), np.uint8)
            jewelry_mask = cv2.morphologyEx(jewelry_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            jewelry_mask = cv2.morphologyEx(jewelry_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Keep largest connected component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(jewelry_mask, connectivity=8)
            if num_labels > 1:
                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                jewelry_mask = (labels == largest_label).astype(np.uint8)
            
            return jewelry_mask
            
        except:
            # Fallback: simple threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return (thresh > 0).astype(np.uint8)
    
    def _estimate_thickness(self, jewelry_type: str, area_mm2: float) -> float:
        """Estimate jewelry thickness based on type and area."""
        base_thickness = self.thickness_map.get(jewelry_type.lower(), 2.0)
        
        # Adjust based on area
        if area_mm2 > 1500:
            base_thickness *= 1.3
        elif area_mm2 > 800:
            base_thickness *= 1.1
        elif area_mm2 < 200:
            base_thickness *= 0.8
        elif area_mm2 < 100:
            base_thickness *= 0.6
        
        return base_thickness
    
    def _detect_hollow_enhanced(self, mask: np.ndarray, image: np.ndarray) -> Tuple[bool, float]:
        """Enhanced hollow detection using multiple features."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.5
        
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        
        solidity = area / hull_area if hull_area > 0 else 1
        
        # Check for internal holes in mask
        inverted_mask = 1 - mask
        hole_contours, _ = cv2.findContours(inverted_mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        internal_holes = 0
        for hc in hole_contours:
            hole_area = cv2.contourArea(hc)
            if 50 < hole_area < area * 0.3:
                internal_holes += 1
        
        # Combine signals
        is_hollow = solidity < 0.75 or internal_holes >= 1
        confidence = (1 - solidity) * 0.7 + (internal_holes * 0.15)
        confidence = min(confidence, 0.9)
        
        return is_hollow, round(confidence, 3)
    
    def _calculate_uncertainty_enhanced(self, scale: float, mask: np.ndarray, 
                                          scale_conf: float, has_ref: bool, 
                                          is_hollow: bool) -> float:
        """Enhanced uncertainty calculation."""
        base_uncertainty = 0.18
        
        # Scale reliability
        if not has_ref:
            base_uncertainty += 0.12
        elif scale_conf < 0.7:
            base_uncertainty += 0.08
        
        # Segmentation quality
        if mask is not None:
            mask_coverage = np.sum(mask > 0) / mask.size
            if mask_coverage < 0.05:
                base_uncertainty += 0.10
            elif mask_coverage < 0.10:
                base_uncertainty += 0.05
            elif mask_coverage > 0.25:
                base_uncertainty -= 0.03
        
        # Hollow items are harder to estimate
        if is_hollow:
            base_uncertainty += 0.08
        
        # Cap uncertainty
        return min(base_uncertainty, 0.35)
    
    def _fallback_estimate(self, karat: str, jewelry_type: str) -> Dict:
        """Fallback estimation when image analysis fails."""
        weight = self.avg_weights.get(jewelry_type.lower(), 15.0)
        
        return {
            "estimated_volume_cm3": round(weight / self.density.get(karat, 17.45), 3),
            "estimated_weight_grams": weight,
            "weight_range": {
                "min": round(weight * 0.7, 2),
                "max": round(weight * 1.3, 2)
            },
            "is_hollow": False,
            "scale_mm_per_pixel": 0.1,
            "reference_detected": False,
            "uncertainty_percent": 28.0,
            "confidence": 0.60
        }


# Singleton instance
weight_estimator = WeightEstimator()