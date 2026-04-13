// frontend/app/assess/page.tsx

"use client";

import { useState } from "react";
import ImageUploader from "@/components/ImageUploader";
import AudioRecorder from "@/components/AudioRecorder";
import ResultsCard from "@/components/ResultsCard";

export default function AssessPage() {
  const [images, setImages] = useState<File[]>([]);
  const [audio, setAudio] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleSubmit = async () => {
    setLoading(true);
    
    const formData = new FormData();
    images.forEach((img) => formData.append("images", img));
    if (audio) formData.append("audio", audio);

    try {
      const response = await fetch("http://localhost:8000/api/assess", {
        method: "POST",
        body: formData,
      });
      
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Assessment failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-4xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">
          Gold Assessment
        </h1>

        {!results ? (
          <div className="space-y-8">
            {/* Image Upload Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <h2 className="text-xl font-semibold mb-4">
                📸 Upload Jewelry Photos (3-5 images)
              </h2>
              <ImageUploader onImagesChange={setImages} />
              <p className="text-sm text-gray-500 mt-2">
                {images.length} images selected
              </p>
            </div>

            {/* Audio Recording Section */}
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <h2 className="text-xl font-semibold mb-4">
                🎤 Record Tap Sound (Optional)
              </h2>
              <AudioRecorder onAudioReady={setAudio} />
            </div>

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={images.length < 3 || loading}
              className={`w-full py-4 rounded-lg text-white font-semibold text-lg
                ${images.length < 3 
                  ? "bg-gray-400 cursor-not-allowed" 
                  : "bg-amber-500 hover:bg-amber-600"
                } transition`}
            >
              {loading ? "Analyzing..." : "Assess Gold →"}
            </button>
          </div>
        ) : (
          <ResultsCard results={results} onReset={() => setResults(null)} />
        )}
      </div>
    </main>
  );
}