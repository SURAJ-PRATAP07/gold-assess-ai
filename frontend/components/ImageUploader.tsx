// frontend/components/ImageUploader.tsx

"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Camera } from "lucide-react";

interface Props {
  onImagesChange: (files: File[]) => void;
}

export default function ImageUploader({ onImagesChange }: Props) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    onImagesChange(acceptedFiles);
  }, [onImagesChange]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpeg", ".jpg", ".png"] },
    maxFiles: 5,
  });

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition
        ${isDragActive 
          ? "border-amber-500 bg-amber-50" 
          : "border-gray-300 hover:border-amber-400"
        }`}
    >
      <input {...getInputProps()} />
      <Camera className="w-12 h-12 mx-auto text-gray-400 mb-4" />
      {isDragActive ? (
        <p className="text-lg text-amber-600">Drop the images here...</p>
      ) : (
        <>
          <p className="text-lg text-gray-600 mb-2">
            Drag & drop jewelry photos here
          </p>
          <p className="text-sm text-gray-400">or click to browse</p>
          <p className="text-xs text-gray-400 mt-4">
            Upload 3-5 clear photos from different angles
          </p>
        </>
      )}
    </div>
  );
}