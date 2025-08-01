import React, { useState, useRef } from 'react';
import './App.css';
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Progress } from './components/ui/progress';
import { Badge } from './components/ui/badge';
import { Upload, Camera, Brain, Sparkles, TrendingUp, Eye, Target, Smile } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setError(null);
      setResults(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setAnalyzing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await axios.post(`${API}/analyze-face`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.response?.data?.detail || 'Failed to analyze image. Please try again.');
    } finally {
      setAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBadgeVariant = (score) => {
    if (score >= 80) return 'default';
    if (score >= 60) return 'secondary';
    return 'destructive';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-900">FaceScore AI</h1>
                <p className="text-sm text-slate-600">Advanced Facial Attractiveness Analysis</p>
              </div>
            </div>
            <Badge variant="outline" className="bg-gradient-to-r from-blue-50 to-purple-50 border-purple-200">
              <Sparkles className="h-3 w-3 mr-1" />
              AI Powered
            </Badge>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Upload Section */}
          <Card className="mb-8 border-slate-200 shadow-lg">
            <CardHeader className="text-center pb-6">
              <CardTitle className="text-2xl font-bold text-slate-900 mb-2">
                Upload Your Photo
              </CardTitle>
              <CardDescription className="text-slate-600 max-w-lg mx-auto">
                Upload a clear, front-facing selfie or portrait for comprehensive facial attractiveness analysis using AI.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!imagePreview ? (
                <div 
                  className="border-2 border-dashed border-slate-300 rounded-xl p-12 text-center hover:border-purple-400 transition-colors cursor-pointer bg-gradient-to-br from-slate-50 to-white"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="flex flex-col items-center space-y-4">
                    <div className="p-4 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full">
                      <Upload className="h-8 w-8 text-purple-600" />
                    </div>
                    <div>
                      <p className="text-lg font-medium text-slate-900 mb-1">Drop your image here</p>
                      <p className="text-slate-600">or click to browse files</p>
                    </div>
                    <Button variant="outline" className="mt-4">
                      <Camera className="h-4 w-4 mr-2" />
                      Choose Photo
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="relative">
                    <img
                      src={imagePreview}
                      alt="Selected"
                      className="w-full max-w-md mx-auto rounded-xl shadow-lg"
                    />
                  </div>
                  <div className="flex justify-center space-x-4">
                    <Button
                      onClick={analyzeImage}
                      disabled={analyzing}
                      className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700"
                    >
                      {analyzing ? (
                        <>
                          <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Brain className="h-4 w-4 mr-2" />
                          Analyze Face
                        </>
                      )}
                    </Button>
                    <Button variant="outline" onClick={resetAnalysis}>
                      Reset
                    </Button>
                  </div>
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />
            </CardContent>
          </Card>

          {/* Error Display */}
          {error && (
            <Card className="mb-8 border-red-200 bg-red-50">
              <CardContent className="pt-6">
                <div className="flex items-center space-x-2 text-red-700">
                  <Target className="h-5 w-5" />
                  <p className="font-medium">{error}</p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Results Section */}
          {results && (
            <div className="space-y-6">
              {/* Overall Score */}
              <Card className="border-slate-200 shadow-lg">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5 text-purple-600" />
                    <span>Overall Attractiveness Score</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center space-y-4">
                    <div className={`text-6xl font-bold ${getScoreColor(results.overall_score)}`}>
                      {results.overall_score}
                      <span className="text-2xl text-slate-600">/100</span>
                    </div>
                    <Badge variant={getScoreBadgeVariant(results.overall_score)} className="text-sm px-4 py-1">
                      {results.overall_score >= 80 ? 'Excellent' : 
                       results.overall_score >= 60 ? 'Good' : 'Average'}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {/* Detailed Scores */}
              <div className="grid md:grid-cols-2 gap-6">
                {/* Symmetry & Golden Ratio */}
                <Card className="border-slate-200 shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-lg">
                      <Eye className="h-5 w-5 text-blue-600" />
                      <span>Facial Analysis</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium text-slate-700">Symmetry Score</span>
                        <Badge variant="outline" className={getScoreColor(results.symmetry_score)}>
                          {results.symmetry_score}
                        </Badge>
                      </div>
                      <Progress value={results.symmetry_score} className="h-2" />
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium text-slate-700">Golden Ratio</span>
                        <Badge variant="outline" className={getScoreColor(results.golden_ratio_score)}>
                          {results.golden_ratio_score}
                        </Badge>
                      </div>
                      <Progress value={results.golden_ratio_score} className="h-2" />
                    </div>
                  </CardContent>
                </Card>

                {/* Feature Breakdown */}
                <Card className="border-slate-200 shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2 text-lg">
                      <Smile className="h-5 w-5 text-green-600" />
                      <span>Feature Breakdown</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {Object.entries(results.feature_breakdown).map(([feature, score]) => (
                      <div key={feature}>
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-medium text-slate-700 capitalize">
                            {feature.replace('_', ' ')}
                          </span>
                          <Badge variant="outline" className={getScoreColor(score)}>
                            {Math.round(score)}
                          </Badge>
                        </div>
                        <Progress value={score} className="h-2" />
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </div>

              {/* Analysis Text */}
              <Card className="border-slate-200 shadow-lg">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Brain className="h-5 w-5 text-purple-600" />
                    <span>AI Analysis</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-700 leading-relaxed">{results.analysis}</p>
                </CardContent>
              </Card>

              {/* Disclaimer */}
              <Card className="border-yellow-200 bg-yellow-50">
                <CardContent className="pt-6">
                  <p className="text-sm text-yellow-800">
                    <strong>Disclaimer:</strong> Beauty is subjective and cultural. This AI analysis is experimental 
                    and should be taken as entertainment only. Your worth is not determined by any algorithm.
                  </p>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white/50 backdrop-blur-md mt-16">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center text-slate-600">
            <p className="text-sm">
              Powered by advanced AI • Built with React & FastAPI • Open Source
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;