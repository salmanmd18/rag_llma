import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const ask = async () => {
    setError("");
    setAnswer("");
    const q = question.trim();
    if (!q) return;
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setAnswer(data.answer || "(no answer)");
    } catch (e) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-2xl bg-white shadow-lg rounded-xl p-6">
        <h1 className="text-2xl md:text-3xl font-semibold mb-4">🩺 Healthcare RAG Q&amp;A</h1>
        <p className="text-gray-600 mb-6">
          Ask a question about PubMed abstracts and WHO fact sheets indexed in this app.
        </p>
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            className="flex-1 border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Enter your question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask()}
          />
          <button
            onClick={ask}
            className="bg-blue-600 hover:bg-blue-700 text-white rounded-md px-4 py-2 disabled:opacity-50"
            disabled={loading}
          >
            {loading ? "Asking..." : "Ask"}
          </button>
        </div>
        {error && (
          <div className="text-red-600 text-sm mb-4">Error: {error}</div>
        )}
        {answer && (
          <div>
            <h2 className="text-lg font-medium mb-2">Answer</h2>
            <div className="whitespace-pre-wrap bg-gray-100 border rounded-md p-3">{answer}</div>
          </div>
        )}
        <div className="mt-6 text-xs text-gray-500">
          Backend: {process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}
        </div>
      </div>
    </main>
  );
}
