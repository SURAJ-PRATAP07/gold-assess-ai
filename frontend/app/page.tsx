// frontend/app/page.tsx

import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-amber-50 to-white">
      {/* Hero Section */}
      <div className="max-w-6xl mx-auto px-4 py-20">
        <div className="text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            GoldAssess AI
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Get your gold valued from home. Instant loan eligibility.
          </p>
          <Link
            href="/assess"
            className="bg-amber-500 text-white px-8 py-4 rounded-lg text-lg font-semibold hover:bg-amber-600 transition"
          >
            Start Assessment →
          </Link>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mt-20">
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <div className="text-3xl mb-4">📸</div>
            <h3 className="text-xl font-semibold mb-2">Upload Photos</h3>
            <p className="text-gray-600">
              Take clear photos from multiple angles
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <div className="text-3xl mb-4">🎤</div>
            <h3 className="text-xl font-semibold mb-2">Tap Test</h3>
            <p className="text-gray-600">
              Record the sound of tapping your gold
            </p>
          </div>
          <div className="bg-white p-6 rounded-xl shadow-sm">
            <div className="text-3xl mb-4">💰</div>
            <h3 className="text-xl font-semibold mb-2">Get Loan Offer</h3>
            <p className="text-gray-600">
              Instant valuation and loan eligibility
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}