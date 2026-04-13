# backend/app/models/vision/classifier.py

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple
import random

class JewelryClassifier:
    """
    AI model to classify jewelry type and estimate purity from images.
    Uses color analysis and shape detection optimized for Indian gold.
    """
    
    def __init__(self):
        self.device = "cpu"
        print(f"✅ JewelryClassifier initialized on {self.device}")
        
        # Jewelry types
        self.jewelry_types = [
            "chain", "ring", "bangle", "bracelet", 
            "necklace", "earring", "pendant", "mangalsutra"
        ]
        
        # Karat levels
        self.karat_levels = ["24K", "22K", "18K", "14K", "10K"]
        
        # Optimized gold color profiles for Indian jewelry (HSV ranges)
        self.gold_hsv_profiles = {
            "24K": {"hue": (40, 55), "sat": (50, 100), "val": (60, 100)},
            "22K": {"hue": (35, 52), "sat": (45, 95), "val": (55, 95)},   # Most common Indian gold
            "18K": {"hue": (30, 48), "sat": (40, 90), "val": (50, 90)},
            "14K": {"hue": (25, 45), "sat": (35, 85), "val": (45, 85)},
            "10K": {"hue": (20, 42), "sat": (30, 80), "val": (40, 80)}
        }
        
        # Expected area ranges for different jewelry types (in pixels at standard distance)
        self.type_area_ranges = {
            "chain": (5000, 50000),
            "ring": (1000, 8000),
            "bangle": (8000, 40000),
            "bracelet": (5000, 30000),
            "necklace": (10000, 60000),
            "earring": (500, 5000),
            "pendant": (2000, 15000),
            "mangalsutra": (15000, 70000)
        }
    
    def classify(self, image_path: str) -> Dict:
        """
        Analyze image and return classification results.
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return self._fallback_classification()
            
            # Get color-based purity estimate
            color_analysis = self._analyze_color(image)
            
            # Detect jewelry type from shape
            jewelry_type = self._detect_jewelry_type(image)
            
            # Detect if hollow
            is_hollow = self._detect_hollow(image)
            
            # Estimate wear
            wear_level = self._estimate_wear(image)
            
            # Calculate final confidence
            type_confidence = jewelry_type["confidence"]
            
            # Boost confidence if color and shape agree
            expected_karat_for_type = self._get_expected_karat_for_type(jewelry_type["type"])
            if expected_karat_for_type == color_analysis["karat"]:
                color_analysis["confidence"] = min(color_analysis["confidence"] * 1.1, 0.95)
            
            return {
                "jewelry_type": jewelry_type["type"],
                "type_confidence": round(type_confidence, 3),
                "estimated_karat": color_analysis["karat"],
                "karat_confidence": round(color_analysis["confidence"], 3),
                "color_analysis": color_analysis,
                "is_hollow": is_hollow,
                "wear_level": wear_level["level"]
            }
            
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return self._fallback_classification()
    
    def _analyze_color(self, image: np.ndarray) -> Dict:
        """
        Analyze gold color in HSV space to estimate purity.
        Optimized for Indian jewelry lighting conditions.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for gold-like colors (wider range for Indian gold)
        lower_gold = np.array([15, 30, 40])
        upper_gold = np.array([65, 255, 255])
        mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # Apply morphological operations to clean mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        # Get non-zero pixels
        non_zero = masked_hsv[mask > 0]
        
        if len(non_zero) < 100:
            # Not enough gold pixels - try without mask on center region
            h, w = hsv.shape[:2]
            center_hsv = hsv[h//4:3*h//4, w//4:3*w//4]
            non_zero = center_hsv.reshape(-1, 3)
            
            if len(non_zero) < 100:
                return {"karat": "22K", "confidence": 0.45, "hsv": {"hue": 42, "saturation": 75, "value": 80}}
        
        # Calculate mean HSV values (use median for robustness)
        mean_h = np.median(non_zero[:, 0])
        mean_s = np.median(non_zero[:, 1])
        mean_v = np.median(non_zero[:, 2])
        
        # Also calculate standard deviation for confidence
        std_h = np.std(non_zero[:, 0])
        std_s = np.std(non_zero[:, 1])
        color_consistency = 1.0 - min((std_h + std_s) / 100, 0.5)
        
        # Find best matching karat
        best_karat = "22K"
        best_score = 0
        
        for karat, profile in self.gold_hsv_profiles.items():
            # Check if values are within ranges
            h_in_range = profile["hue"][0] <= mean_h <= profile["hue"][1]
            s_in_range = profile["sat"][0] <= mean_s <= profile["sat"][1]
            v_in_range = profile["val"][0] <= mean_v <= profile["val"][1]
            
            # Calculate match score
            score = 0
            if h_in_range: score += 0.35
            if s_in_range: score += 0.35
            if v_in_range: score += 0.30
            
            # Bonus for being close to center of range
            h_center = (profile["hue"][0] + profile["hue"][1]) / 2
            s_center = (profile["sat"][0] + profile["sat"][1]) / 2
            score += 0.15 * (1 - abs(mean_h - h_center) / 25)
            score += 0.15 * (1 - abs(mean_s - s_center) / 40)
            
            if score > best_score:
                best_score = score
                best_karat = karat
        
        # Adjust confidence based on color consistency
        confidence = min(best_score * color_consistency, 0.95)
        
        return {
            "karat": best_karat,
            "confidence": round(confidence, 3),
            "hsv": {
                "hue": round(float(mean_h), 1),
                "saturation": round(float(mean_s), 1),
                "value": round(float(mean_v), 1)
            },
            "color_consistency": round(color_consistency, 3)
        }
    
    def _detect_jewelry_type(self, image: np.ndarray) -> Dict:
        """
        Detect jewelry type from shape characteristics.
        Enhanced with multiple features.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold for better segmentation
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Try with edges if threshold didn't work
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {"type": "chain", "confidence": 0.5}
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 1
        
        # Calculate shape features
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate convex hull and solidity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1
        
        # Calculate extent (area / bounding box area)
        bbox_area = w * h
        extent = area / bbox_area if bbox_area > 0 else 0
        
        # Image dimensions
        img_h, img_w = image.shape[:2]
        
        # Detect type based on multiple features
        if aspect_ratio > 3.0:
            detected_type = "chain"
            confidence = 0.85
        elif aspect_ratio > 2.0 and extent < 0.5:
            detected_type = "chain"
            confidence = 0.75
        elif circularity > 0.65 and aspect_ratio > 0.7 and aspect_ratio < 1.4:
            if area < 5000:
                detected_type = "ring"
                confidence = 0.8
            else:
                detected_type = "bangle"
                confidence = 0.75
        elif aspect_ratio > 1.5 and w > img_w * 0.5:
            detected_type = "necklace"
            confidence = 0.7
        elif area < 4000 and aspect_ratio < 2.0:
            detected_type = "earring"
            confidence = 0.75
        elif solidity < 0.7 and area > 5000:
            detected_type = "mangalsutra"
            confidence = 0.7
        else:
            detected_type = "pendant"
            confidence = 0.65
        
        # Adjust confidence based on feature agreement
        expected_area_range = self.type_area_ranges.get(detected_type, (1000, 50000))
        if expected_area_range[0] <= area <= expected_area_range[1]:
            confidence = min(confidence * 1.1, 0.95)
        else:
            confidence = confidence * 0.9
        
        return {
            "type": detected_type, 
            "confidence": round(confidence, 3),
            "features": {
                "area": int(area),
                "aspect_ratio": round(aspect_ratio, 2),
                "circularity": round(circularity, 3),
                "solidity": round(solidity, 3)
            }
        }
    
    def _detect_hollow(self, image: np.ndarray) -> Dict:
        """
        Detect if jewelry is hollow using edge density and internal contours.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Find internal edges (holes)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count internal contours (holes)
        internal_contours = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 5000:  # Size range for hollow openings
                internal_contours += 1
        
        # Hollow items have more internal edges and holes
        is_hollow = edge_density > 0.08 or internal_contours >= 2
        
        confidence = min((edge_density * 6 + internal_contours * 0.15), 0.9)
        
        return {
            "is_hollow": is_hollow,
            "confidence": round(confidence, 3),
            "edge_density": round(edge_density, 4),
            "internal_contours": internal_contours
        }
    
    def _estimate_wear(self, image: np.ndarray) -> Dict:
        """
        Estimate wear and tear from surface texture.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance (texture roughness)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(gray.astype(float), -1, kernel)
        local_var = cv2.filter2D((gray.astype(float) - local_mean)**2, -1, kernel)
        
        roughness = np.mean(local_var)
        
        # Also check for scratches using edge detection
        edges = cv2.Canny(gray, 80, 200)
        scratch_density = np.sum(edges > 0) / edges.size
        
        if roughness < 50 and scratch_density < 0.05:
            level = "minimal"
            confidence = 0.8
        elif roughness < 150 and scratch_density < 0.12:
            level = "moderate"
            confidence = 0.75
        else:
            level = "heavy"
            confidence = 0.8
        
        return {
            "level": level,
            "roughness_score": round(float(roughness), 1),
            "scratch_density": round(scratch_density, 4),
            "confidence": confidence
        }
    
    def _get_expected_karat_for_type(self, jewelry_type: str) -> str:
        """Get typical karat for jewelry type (Indian market)."""
        typical_karats = {
            "chain": "22K",
            "ring": "22K",
            "bangle": "22K",
            "bracelet": "22K",
            "necklace": "22K",
            "earring": "22K",
            "pendant": "22K",
            "mangalsutra": "22K"
        }
        return typical_karats.get(jewelry_type, "22K")
    
    def _fallback_classification(self) -> Dict:
        """Fallback when analysis fails."""
        return {
            "jewelry_type": "chain",
            "type_confidence": 0.5,
            "estimated_karat": "22K",
            "karat_confidence": 0.6,
            "color_analysis": {"karat": "22K", "confidence": 0.6, "hsv": {"hue": 42, "saturation": 75, "value": 80}, "color_consistency": 0.7},
            "is_hollow": {"is_hollow": False, "confidence": 0.5},
            "wear_level": "minimal"
        }


# Singleton instance
classifier = JewelryClassifier()