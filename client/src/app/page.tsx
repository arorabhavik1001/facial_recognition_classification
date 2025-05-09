"use client";
import React from 'react';
import { ImageUpload } from './components/ImageUpload';
import { ResultDisplay } from './components/ResultDisplay';
import { useAppContext, AppProvider } from './context/AppContext';
import { Header } from './components/Header';

export default function Home() {
  const {
    recognitionResult,
    isProcessing,
    errorMessage,
    processedImageUrl,
  } = useAppContext();

  return (
    <AppProvider>
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 container mx-auto px-4 py-8 max-w-6xl">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="flex flex-col">
              <h2 className="text-2xl font-semibold mb-4 text-cyan-300">Upload Image</h2>
              <div className="bg-gray-800 rounded-lg shadow-lg p-6 flex-1">
                <ImageUpload />
              </div>
            </div>

            <div className="flex flex-col">
              <h2 className="text-2xl font-semibold mb-4 text-cyan-300">Recognition Results</h2>
              <div className="bg-gray-800 rounded-lg shadow-lg p-6 flex-1">
                <ResultDisplay
                  result={recognitionResult}
                  isProcessing={isProcessing}
                  errorMessage={errorMessage}
                  processedImageUrl={processedImageUrl}
                />
              </div>
            </div>
          </div>
          <div className="mt-8 text-gray-400 text-sm">
            <p>
              Disclaimer: This model is trained only on the images of specific individuals. 
              If an image of a person outside the trained classes is provided, the model will 
              still attempt to classify it into one of the learned classes, as it is a traditional 
              classification model.
            </p>
          </div>
        </main>
        <footer className="bg-gray-800 py-4 text-center text-gray-400 text-sm">
          <div className="container mx-auto">
            Facial Recognition System &copy; {new Date().getFullYear()}
          </div>
        </footer>
      </div>
    </AppProvider>
  );
}