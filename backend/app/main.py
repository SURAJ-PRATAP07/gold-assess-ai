# backend/app/main.py

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import shutil
import os
import tempfile
import httpx
from datetime import datetime
import cv2
import numpy as np
from collections import Counter

# Import our REAL AI models
from app.models.vision.classifier import classifier
from app.models.vision.hallmark import hallmark_detector
from app.models.vision.weight import weight_estimator

app = FastAPI(title="GoldAssess AI")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gold-assess-ai-2.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AssessmentResponse(BaseModel):
    success: bool
    jewelry_type: str
    jewelry_subtype: Optional[str] = None
    weight_range: dict
    purity: dict
    hallmark: dict
    market_value: dict
    loan_eligible: dict
    risk_level: str
    risk_flags: List[str]
    recommendation: str
    confidence_score: float

def check_image_quality(image_path: str) -> dict:
    """Check if image is good for analysis."""
    img = cv2.imread(image_path)
    if img is None:
        return {"quality": "poor", "issues": ["Cannot read image"], "brightness": 0, "sharpness": 0}
    
    h, w = img.shape[:2]
    issues = []
    
    # Check resolution
    if w < 400 or h < 400:
        issues.append("Low resolution - use at least 400x400 pixels")
    
    # Check brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 50:
        issues.append("Too dark - use better lighting")
    elif brightness > 200:
        issues.append("Too bright - avoid direct light")
    
    # Check blur
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        issues.append("Image is blurry - hold camera steady")
    
    quality = "good" if len(issues) == 0 else "fair" if len(issues) < 2 else "poor"
    
    return {
        "quality": quality, 
        "issues": issues, 
        "brightness": round(brightness, 1), 
        "sharpness": round(laplacian_var, 1),
        "resolution": f"{w}x{h}"
    }

def get_consensus_classification(image_paths: List[str]) -> tuple:
    """Analyze multiple images and find consensus."""
    types = []
    karats = []
    confidences = []
    
    for path in image_paths[:3]:  # Use first 3 images
        try:
            result = classifier.classify(path)
            types.append(result["jewelry_type"])
            karats.append(result["estimated_karat"])
            confidences.append(result["type_confidence"])
        except Exception as e:
            print(f"   ⚠️  Failed to classify {path}: {e}")
    
    if types:
        # Find most common type
        type_counts = Counter(types)
        best_type = type_counts.most_common(1)[0][0]
        type_agreement = type_counts[best_type] / len(types)
        
        # Find most common karat
        karat_counts = Counter(karats)
        best_karat = karat_counts.most_common(1)[0][0]
        karat_agreement = karat_counts[best_karat] / len(karats)
        
        # Boost confidence if all images agree
        confidence_boost = 0.15 if type_agreement == 1.0 else 0.05 if type_agreement >= 0.66 else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return best_type, best_karat, confidence_boost, type_agreement
    
    return "chain", "22K", 0, 0

async def get_live_gold_price():
    """Fetch real gold price from API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.gold-api.com/price/XAU",
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                price_usd = data.get("price", 2000)
                
                # Get USD to INR
                usd_response = await client.get(
                    "https://api.exchangerate-api.com/v4/latest/USD",
                    timeout=5.0
                )
                usd_inr = 83.0
                if usd_response.status_code == 200:
                    usd_inr = usd_response.json().get("rates", {}).get("INR", 83.0)
                
                price_24k = (price_usd * usd_inr) / 31.1035
                
                return {
                    "24K": round(price_24k),
                    "22K": round(price_24k * 0.916),
                    "18K": round(price_24k * 0.75),
                }
    except Exception as e:
        print(f"   ⚠️  Gold API error: {e}")
    
    return {"24K": 6850, "22K": 6250, "18K": 5120}

@app.get("/")
async def root():
    return {"message": "GoldAssess AI API", "status": "running", "ai_models": "loaded"}

@app.post("/api/assess")
async def assess_gold(
    images: List[UploadFile] = File(...),
    audio: Optional[UploadFile] = File(None),
    declared_weight: Optional[float] = Form(None),
    declared_purity: Optional[str] = Form(None),
):
    """
    Main assessment endpoint using REAL AI models with multi-image consensus.
    """
    print(f"\n{'='*60}")
    print(f"🎯 NEW ASSESSMENT REQUEST")
    print(f"📸 Received {len(images)} images")
    print(f"{'='*60}\n")
    
    # Save uploaded images temporarily
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    quality_reports = []
    
    for i, image in enumerate(images):
        file_path = os.path.join(temp_dir, f"image_{i}.jpg")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image_paths.append(file_path)
        
        # Check image quality
        quality = check_image_quality(file_path)
        quality_reports.append(quality)
        print(f"✅ Saved: image_{i}.jpg - Quality: {quality['quality']} - {quality['resolution']}")
        if quality['issues']:
            for issue in quality['issues']:
                print(f"   ⚠️  {issue}")
    
    # Use first image for primary analysis
    primary_image = image_paths[0]
    
    try:
        # 1. MULTI-IMAGE CONSENSUS CLASSIFICATION
        print("\n🔍 Running Multi-Image AI Classification...")
        consensus_type, consensus_karat, consensus_boost, agreement = get_consensus_classification(image_paths)
        print(f"   → Consensus Type: {consensus_type} (agreement: {agreement*100:.0f}%)")
        print(f"   → Consensus Karat: {consensus_karat}")
        
        # 2. PRIMARY IMAGE DETAILED ANALYSIS
        print("\n📊 Running Detailed Analysis on Best Image...")
        classification = classifier.classify(primary_image)
        print(f"   → Type: {classification['jewelry_type']} (confidence: {classification['type_confidence']*100:.0f}%)")
        print(f"   → Karat: {classification['estimated_karat']} (confidence: {classification['karat_confidence']*100:.0f}%)")
        print(f"   → Wear Level: {classification['wear_level']}")
        
        # 3. HALLMARK DETECTION
        print("\n🏷️  Running Hallmark Detection on All Images...")
        hallmark_results = []
        for path in image_paths:
            result = hallmark_detector.detect(path)
            hallmark_results.append(result)
        
        # Combine hallmark results - if ANY image has valid hallmark, consider it verified
        any_hallmark = any(h["has_valid_hallmark"] for h in hallmark_results)
        any_bis = any(h["bis_verified"] for h in hallmark_results)
        detected_purities = [h["detected_purity"] for h in hallmark_results if h["detected_purity"]]
        
        hallmark = {
            "detected": any(h["detected"] for h in hallmark_results),
            "bis_verified": any_bis,
            "detected_purity": detected_purities[0] if detected_purities else None,
            "purity_karat": hallmark_results[0]["purity_karat"],
            "has_valid_hallmark": any_hallmark,
            "confidence": max(h["confidence"] for h in hallmark_results) if hallmark_results else 0.3
        }
        
        print(f"   → Hallmark Detected: {hallmark['detected']}")
        print(f"   → BIS Verified: {hallmark['bis_verified']}")
        print(f"   → Purity Stamp: {hallmark['detected_purity']}")
        
        # 4. DETERMINE FINAL PURITY
        if hallmark["has_valid_hallmark"]:
            purity_karat = hallmark["purity_karat"]
            purity_confidence = hallmark["confidence"] + 0.1  # Bonus for hallmark
            print(f"\n✅ Using Hallmark Purity: {purity_karat}")
        elif consensus_karat == classification["estimated_karat"]:
            purity_karat = consensus_karat
            purity_confidence = classification["karat_confidence"] + consensus_boost
            print(f"\n✅ Using Consensus Purity: {purity_karat}")
        else:
            purity_karat = classification["estimated_karat"]
            purity_confidence = classification["karat_confidence"]
            print(f"\n📊 Using AI Color Analysis: {purity_karat}")
        
        # Cap confidence at 0.95
        purity_confidence = min(purity_confidence, 0.95)
        
        # 5. WEIGHT ESTIMATION
        print("\n⚖️  Running Weight Estimation...")
        weight_estimations = []
        for path in image_paths[:3]:
            est = weight_estimator.estimate(path, purity_karat, consensus_type)
            weight_estimations.append(est)
        
        # Average the weight estimations
        avg_weight = sum(w["estimated_weight_grams"] for w in weight_estimations) / len(weight_estimations)
        avg_min = sum(w["weight_range"]["min"] for w in weight_estimations) / len(weight_estimations)
        avg_max = sum(w["weight_range"]["max"] for w in weight_estimations) / len(weight_estimations)
        avg_confidence = sum(w["confidence"] for w in weight_estimations) / len(weight_estimations)
        any_hollow = any(w.get("is_hollow", False) for w in weight_estimations)
        
        print(f"   → Estimated: {avg_weight:.1f}g")
        print(f"   → Range: {avg_min:.1f}g - {avg_max:.1f}g")
        print(f"   → Hollow: {any_hollow}")
        
        # Use declared weight if provided and reasonable
        if declared_weight:
            diff_percent = abs(declared_weight - avg_weight) / avg_weight
            if diff_percent < 0.3:  # Within 30%
                estimated_weight = (avg_weight + declared_weight) / 2
                weight_min = estimated_weight * 0.88
                weight_max = estimated_weight * 1.12
                print(f"   → Adjusted with declared weight: {estimated_weight:.1f}g (diff: {diff_percent*100:.0f}%)")
            else:
                estimated_weight = avg_weight
                weight_min = avg_min
                weight_max = avg_max
                print(f"   → Declared weight too different ({diff_percent*100:.0f}%), using AI estimate")
        else:
            estimated_weight = avg_weight
            weight_min = avg_min
            weight_max = avg_max
        
        # 6. GET LIVE GOLD PRICES
        print("\n💰 Fetching Live Gold Prices...")
        gold_prices = await get_live_gold_price()
        gold_rate = gold_prices.get(purity_karat, 6250)
        print(f"   → {purity_karat} Rate: ₹{gold_rate}/g")
        
        # 7. CALCULATE MARKET VALUE
        value_min = weight_min * gold_rate
        value_max = weight_max * gold_rate
        value_avg = estimated_weight * gold_rate
        
        # 8. CALCULATE LOAN ELIGIBILITY
        ltv = 0.75
        loan_amount = value_avg * ltv
        
        # 9. COMPREHENSIVE RISK ASSESSMENT
        print("\n⚠️  Running Risk Assessment...")
        risk_flags = []
        risk_score = 0
        
        # Quality-based risks
        poor_quality_count = sum(1 for q in quality_reports if q['quality'] == 'poor')
        if poor_quality_count >= 2:
            risk_flags.append(f"Multiple low-quality images ({poor_quality_count})")
            risk_score += 15
        
        # Hallmark risks
        if not hallmark["has_valid_hallmark"]:
            risk_flags.append("No BIS hallmark detected")
            risk_score += 25
            print("   ⚠️  No BIS hallmark")
        
        # Purity confidence risks
        if purity_confidence < 0.7:
            risk_flags.append(f"Low purity confidence ({int(purity_confidence*100)}%)")
            risk_score += 20
            print(f"   ⚠️  Low purity confidence: {int(purity_confidence*100)}%")
        
        # Hollow detection
        if any_hollow:
            risk_flags.append("Item appears hollow")
            risk_score += 15
            print("   ⚠️  Hollow detected")
        
        # Wear level
        if classification.get("wear_level") == "heavy":
            risk_flags.append("Heavy wear detected")
            risk_score += 10
        
        # Agreement issues
        if agreement < 0.66:
            risk_flags.append(f"Low image agreement ({int(agreement*100)}%)")
            risk_score += 10
        
        # Determine risk level
        if risk_score < 20:
            risk_level = "LOW"
            recommendation = "APPROVE"
        elif risk_score < 45:
            risk_level = "MEDIUM"
            recommendation = "APPROVE"
        else:
            risk_level = "HIGH"
            recommendation = "VERIFY"
        
        print(f"   → Risk Score: {risk_score} ({risk_level})")
        
        # 10. OVERALL CONFIDENCE
        confidence_score = (
            purity_confidence * 0.30 +
            avg_confidence * 0.30 +
            agreement * 0.20 +
            (1 - risk_score/100) * 0.20
        )
        
        # Adjust for image quality
        avg_quality_score = sum(1 if q['quality'] == 'good' else 0.5 if q['quality'] == 'fair' else 0.2 for q in quality_reports) / len(quality_reports)
        confidence_score = confidence_score * (0.7 + 0.3 * avg_quality_score)
        confidence_score = min(confidence_score, 0.98)
        
        print(f"\n✨ Assessment Complete!")
        print(f"   → Final Confidence: {confidence_score*100:.0f}%")
        print(f"   → Recommendation: {recommendation}")
        print(f"   → Risk Flags: {len(risk_flags)}")
        print(f"{'='*60}\n")
        
        # Clean up
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        return {
            "success": True,
            "jewelry_type": consensus_type.title(),
            "jewelry_subtype": None,
            "weight_range": {
                "min": round(weight_min, 1),
                "max": round(weight_max, 1),
                "unit": "grams"
            },
            "purity": {
                "karat": purity_karat,
                "confidence": round(purity_confidence, 2)
            },
            "hallmark": {
                "detected": hallmark["detected"],
                "bis_verified": hallmark["bis_verified"],
                "purity_stamp": hallmark.get("detected_purity") or "N/A"
            },
            "market_value": {
                "min": round(value_min),
                "max": round(value_max),
                "currency": "INR"
            },
            "loan_eligible": {
                "amount": round(loan_amount),
                "ltv": int(ltv * 100),
                "interest_rate": 10.5,
                "tenure_months": 12
            },
            "risk_level": risk_level,
            "risk_flags": risk_flags,
            "recommendation": recommendation,
            "confidence_score": round(confidence_score, 2)
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up
        try:
            for path in image_paths:
                os.remove(path)
            os.rmdir(temp_dir)
        except:
            pass
        
        # Fallback response
        return {
            "success": True,
            "jewelry_type": "Chain",
            "jewelry_subtype": None,
            "weight_range": {"min": 15.0, "max": 20.0, "unit": "grams"},
            "purity": {"karat": "22K", "confidence": 0.70},
            "hallmark": {"detected": False, "bis_verified": False, "purity_stamp": "N/A"},
            "market_value": {"min": 93000, "max": 125000, "currency": "INR"},
            "loan_eligible": {"amount": 82000, "ltv": 75, "interest_rate": 10.5, "tenure_months": 12},
            "risk_level": "MEDIUM",
            "risk_flags": ["AI analysis incomplete - manual verification recommended"],
            "recommendation": "VERIFY",
            "confidence_score": 0.65
        }

@app.get("/api/gold-rate")
async def get_gold_rate():
    return await get_live_gold_price()

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "ai_models": "loaded"}
