// frontend/components/ResultsCard.tsx

"use client";

interface ResultsCardProps {
  results: any;
  onReset: () => void;
}

export default function ResultsCard({ results, onReset }: ResultsCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "APPROVE": return "bg-green-100 text-green-800";
      case "VERIFY": return "bg-yellow-100 text-yellow-800";
      case "REJECT": return "bg-red-100 text-red-800";
      default: return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-amber-500 to-amber-600 p-6 text-white">
        <h2 className="text-2xl font-bold">Assessment Complete</h2>
        <p className="opacity-90">Confidence: {(results.confidence_score * 100).toFixed(0)}%</p>
      </div>

      {/* Results */}
      <div className="p-6 space-y-6">
        {/* Jewelry Info */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-gray-500 text-sm">Jewelry Type</p>
            <p className="text-xl font-semibold">{results.jewelry_type}</p>
          </div>
          <div>
            <p className="text-gray-500 text-sm">Purity</p>
            <p className="text-xl font-semibold">{results.purity.karat}</p>
          </div>
        </div>

        {/* Weight */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-gray-500 text-sm mb-1">Estimated Weight</p>
          <p className="text-2xl font-bold">
            {results.weight_range.min} - {results.weight_range.max}g
          </p>
        </div>

        {/* Valuation */}
        <div className="bg-amber-50 p-4 rounded-lg">
          <p className="text-gray-500 text-sm mb-1">Market Value</p>
          <p className="text-2xl font-bold text-amber-700">
            ₹{results.market_value.min.toLocaleString()} - ₹{results.market_value.max.toLocaleString()}
          </p>
        </div>

        {/* Loan Offer */}
        <div className="border border-amber-200 p-4 rounded-lg">
          <p className="text-gray-500 text-sm mb-1">Loan Eligibility</p>
          <p className="text-2xl font-bold text-green-600">
            ₹{results.loan_eligible.amount.toLocaleString()}
          </p>
          <p className="text-sm text-gray-500">
            Interest Rate: {results.loan_eligible.interest_rate}% | LTV: {results.loan_eligible.ltv}%
          </p>
        </div>

        {/* Risk & Recommendation */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-700">Risk Level</span>
            <span className={`px-3 py-1 rounded-full text-sm font-medium
              ${results.risk_level === "LOW" ? "bg-green-100 text-green-800" : 
                results.risk_level === "MEDIUM" ? "bg-yellow-100 text-yellow-800" : 
                "bg-red-100 text-red-800"}`}>
              {results.risk_level}
            </span>
          </div>
          
          {results.risk_flags.length > 0 && (
            <div className="mt-2">
              <p className="text-sm text-gray-500 mb-1">Risk Flags:</p>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {results.risk_flags.map((flag: string, i: number) => (
                  <li key={i}>{flag}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Recommendation */}
        <div className={`p-4 rounded-lg ${getStatusColor(results.recommendation)}`}>
          <p className="font-semibold text-lg">
            Recommendation: {results.recommendation}
          </p>
        </div>

        {/* Reset Button */}
        <button
          onClick={onReset}
          className="w-full py-3 bg-gray-100 text-gray-700 rounded-lg font-semibold hover:bg-gray-200 transition"
        >
          New Assessment
        </button>
      </div>
    </div>
  );
}