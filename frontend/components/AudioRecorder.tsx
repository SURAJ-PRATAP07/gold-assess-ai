// frontend/components/AudioRecorder.tsx

"use client";

import { useState, useRef } from "react";
import { Mic, Square, Play, Trash2 } from "lucide-react";

interface Props {
  onAudioReady: (file: File | null) => void;
}

export default function AudioRecorder({ onAudioReady }: Props) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      
      mediaRecorder.current.ondataavailable = (e) => {
        chunks.current.push(e.data);
      };
      
      mediaRecorder.current.onstop = () => {
        const blob = new Blob(chunks.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        setAudioURL(url);
        setAudioBlob(blob);
        
        const file = new File([blob], "tap-test.webm", { type: "audio/webm" });
        onAudioReady(file);
        
        chunks.current = [];
      };
      
      mediaRecorder.current.start();
      setIsRecording(true);
    } catch (error) {
      alert("Could not access microphone");
    }
  };

  const stopRecording = () => {
    mediaRecorder.current?.stop();
    setIsRecording(false);
  };

  const deleteRecording = () => {
    setAudioURL(null);
    setAudioBlob(null);
    onAudioReady(null);
  };

  return (
    <div className="space-y-4">
      {!audioURL ? (
        <button
          onClick={isRecording ? stopRecording : startRecording}
          className={`w-full py-3 rounded-lg font-semibold flex items-center justify-center gap-2
            ${isRecording 
              ? "bg-red-500 hover:bg-red-600 text-white" 
              : "bg-gray-100 hover:bg-gray-200 text-gray-700"
            }`}
        >
          {isRecording ? (
            <>
              <Square className="w-5 h-5" /> Stop Recording
            </>
          ) : (
            <>
              <Mic className="w-5 h-5" /> Start Recording
            </>
          )}
        </button>
      ) : (
        <div className="flex gap-2">
          <audio src={audioURL} controls className="flex-1" />
          <button
            onClick={deleteRecording}
            className="px-4 py-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      )}
      <p className="text-xs text-gray-500">
        Tap the jewelry with another piece of gold and record for 3-5 seconds
      </p>
    </div>
  );
}