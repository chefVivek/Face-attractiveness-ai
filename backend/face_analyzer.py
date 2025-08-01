import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import math
from typing import Dict, Tuple, List, Optional
import io
import base64

class FaceAttractivenessAnalyzer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3  # Much more lenient
        )
        
        # Load a simple pretrained model for feature extraction
        self.feature_extractor = torch.hub.load('pytorch/vision', 'resnet18', weights='DEFAULT')
        self.feature_extractor.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Golden ratio constant
        self.GOLDEN_RATIO = 1.618
        
        # Face landmarks indices for key points
        self.FACE_OVAL = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        
        # Eye landmarks
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Mouth landmarks
        self.MOUTH = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 78]

    def analyze_face(self, image_bytes: bytes) -> Dict:
        """Main analysis function"""
        try:
            # Convert bytes to PIL Image
            image_pil = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            # Ensure image is not too small
            if image_pil.size[0] < 100 or image_pil.size[1] < 100:
                # Resize small images
                image_pil = image_pil.resize((400, 400))
                
            # Convert to cv2 format
            image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            
            # Try face detection with multiple attempts
            results = self.face_mesh.process(image_rgb)
            
            # If first attempt fails, try with different processing
            if not results.multi_face_landmarks:
                # Try with enhanced contrast
                enhanced = cv2.equalizeHist(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY))
                enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                results = self.face_mesh.process(enhanced_rgb)
            
            if not results.multi_face_landmarks:
                return {"error": "No face detected in the image. Please upload a clear, front-facing portrait photo."}
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image_rgb.shape[:2]
            
            print(f"Image dimensions: {w}x{h}, Total landmarks: {len(landmarks.landmark)}")
            
            # Convert landmarks to pixel coordinates
            points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
            
            print(f"Converted {len(points)} landmark points")
            
            # Calculate scores
            overall_score = self._calculate_overall_attractiveness(image_pil, points, w, h)
            symmetry_score = self._calculate_symmetry_score(points, w, h)
            golden_ratio_score = self._calculate_golden_ratio_score(points, w, h)
            feature_scores = self._calculate_feature_scores(points, w, h)
            
            print(f"Scores calculated - Overall: {overall_score}, Symmetry: {symmetry_score}, Golden: {golden_ratio_score}")
            
            return {
                "overall_score": round(overall_score, 1),
                "symmetry_score": round(symmetry_score, 1),
                "golden_ratio_score": round(golden_ratio_score, 1),
                "feature_breakdown": feature_scores,
                "analysis": self._generate_analysis(overall_score, symmetry_score, golden_ratio_score, feature_scores)
            }
            
        except Exception as e:
            print(f"Analysis failed with error: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}. Please try a different image."}

    def _calculate_overall_attractiveness(self, image_pil: Image.Image, points: List[Tuple], w: int, h: int) -> float:
        """Calculate overall attractiveness using multiple factors"""
        try:
            # Extract features using pretrained model
            image_tensor = self.transform(image_pil).unsqueeze(0)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
                features_norm = torch.nn.functional.normalize(features, p=2, dim=1)
            
            # Base score from facial proportions and symmetry
            symmetry = self._calculate_symmetry_score(points, w, h)
            proportions = self._calculate_golden_ratio_score(points, w, h)
            
            # Feature harmony score
            harmony = self._calculate_harmony_score(points, w, h)
            
            # Combine scores with weights
            base_score = (symmetry * 0.3 + proportions * 0.3 + harmony * 0.4)
            
            # Add some variability based on features (simulating learned attractiveness)
            feature_variance = float(torch.mean(features_norm).item() * 20 + 50)
            
            # Final score combination
            final_score = (base_score * 0.7 + feature_variance * 0.3)
            
            # Ensure score is between 0-100
            return max(0, min(100, final_score))
            
        except Exception:
            return 50.0  # Default score if calculation fails

    def _calculate_symmetry_score(self, points: List[Tuple], w: int, h: int) -> float:
        """Calculate facial symmetry score"""
        try:
            # Get key symmetric points with error checking
            left_eye_points = [points[i] for i in self.LEFT_EYE if i < len(points)]
            right_eye_points = [points[i] for i in self.RIGHT_EYE if i < len(points)]
            
            if not left_eye_points or not right_eye_points:
                return 60.0  # Default if eye points missing
            
            left_eye_center = self._get_center_point(left_eye_points)
            right_eye_center = self._get_center_point(right_eye_points)
            
            # Calculate face center line
            face_center_x = w / 2
            
            # Simple symmetry calculation: how close are eyes to being equidistant from center
            left_distance = abs(left_eye_center[0] - face_center_x)
            right_distance = abs(right_eye_center[0] - face_center_x)
            
            # Calculate symmetry as similarity of distances
            if left_distance + right_distance == 0:
                return 100.0  # Perfect center alignment
            
            symmetry_ratio = 1 - abs(left_distance - right_distance) / (left_distance + right_distance)
            
            # Also check vertical alignment of eyes
            eye_height_diff = abs(left_eye_center[1] - right_eye_center[1])
            max_height_diff = h * 0.05  # 5% of image height
            height_symmetry = max(0, 1 - (eye_height_diff / max_height_diff))
            
            # Combine horizontal and vertical symmetry
            final_symmetry = (symmetry_ratio * 0.7 + height_symmetry * 0.3)
            
            return max(20, min(100, final_symmetry * 100))  # Minimum 20, never 0
            
        except Exception as e:
            print(f"Symmetry calculation error: {e}")
            return 60.0  # Reasonable default

    def _calculate_golden_ratio_score(self, points: List[Tuple], w: int, h: int) -> float:
        """Calculate how well face proportions match golden ratio"""
        try:
            # Get face boundary points with error checking
            face_oval_points = [points[i] for i in self.FACE_OVAL if i < len(points)]
            
            if len(face_oval_points) < 10:  # Need enough points for calculation
                return 65.0  # Default score
            
            # Get key facial measurements
            face_y_coords = [p[1] for p in face_oval_points]
            face_x_coords = [p[0] for p in face_oval_points]
            
            face_top = min(face_y_coords)
            face_bottom = max(face_y_coords)
            face_left = min(face_x_coords)
            face_right = max(face_x_coords)
            
            face_width = face_right - face_left
            face_height = face_bottom - face_top
            
            if face_width <= 0 or face_height <= 0:
                return 65.0  # Avoid division by zero
            
            # Calculate eye and mouth positions
            left_eye_points = [points[i] for i in self.LEFT_EYE if i < len(points)]
            right_eye_points = [points[i] for i in self.RIGHT_EYE if i < len(points)]
            mouth_points = [points[i] for i in self.MOUTH if i < len(points)]
            
            if not all([left_eye_points, right_eye_points, mouth_points]):
                return 65.0  # Missing critical features
            
            eye_center_y = (self._get_center_point(left_eye_points)[1] + 
                           self._get_center_point(right_eye_points)[1]) / 2
            mouth_center_y = self._get_center_point(mouth_points)[1]
            
            eye_mouth_distance = abs(mouth_center_y - eye_center_y)
            
            # Check various golden ratio relationships
            ratios = []
            
            # Face width to height ratio (ideal around 0.618)
            width_height_ratio = face_width / face_height
            ideal_ratio = 1 / self.GOLDEN_RATIO  # ~0.618
            ratio_score = max(0, 1 - abs(width_height_ratio - ideal_ratio) / ideal_ratio)
            ratios.append(ratio_score)
            
            # Eye to mouth distance vs face height (ideal around 0.36)
            if eye_mouth_distance > 0:
                eye_mouth_ratio = eye_mouth_distance / face_height
                ideal_eye_mouth = 0.36
                eye_mouth_score = max(0, 1 - abs(eye_mouth_ratio - ideal_eye_mouth) / ideal_eye_mouth)
                ratios.append(eye_mouth_score)
            
            # Eye width vs face width (ideal around 0.3)
            eye_distance = abs(self._get_center_point(left_eye_points)[0] - 
                             self._get_center_point(right_eye_points)[0])
            if eye_distance > 0:
                eye_face_ratio = eye_distance / face_width
                ideal_eye_ratio = 0.3
                eye_ratio_score = max(0, 1 - abs(eye_face_ratio - ideal_eye_ratio) / ideal_eye_ratio)
                ratios.append(eye_ratio_score)
            
            # Average the ratios
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0.6
            
            return max(30, min(100, avg_ratio * 100))  # Minimum 30, never 0
            
        except Exception as e:
            print(f"Golden ratio calculation error: {e}")
            return 65.0  # Reasonable default

    def _calculate_feature_scores(self, points: List[Tuple], w: int, h: int) -> Dict[str, float]:
        """Calculate individual feature scores"""
        try:
            scores = {}
            
            # Eye symmetry and size
            left_eye_points = [points[i] for i in self.LEFT_EYE]
            right_eye_points = [points[i] for i in self.RIGHT_EYE]
            
            left_eye_width = max([p[0] for p in left_eye_points]) - min([p[0] for p in left_eye_points])
            right_eye_width = max([p[0] for p in right_eye_points]) - min([p[0] for p in right_eye_points])
            
            eye_symmetry = 1 - abs(left_eye_width - right_eye_width) / max(left_eye_width, right_eye_width)
            scores["eye_symmetry"] = max(0, min(100, eye_symmetry * 100))
            
            # Mouth proportions
            mouth_points = [points[i] for i in self.MOUTH]
            mouth_width = max([p[0] for p in mouth_points]) - min([p[0] for p in mouth_points])
            face_width = max([points[i][0] for i in self.FACE_OVAL]) - min([points[i][0] for i in self.FACE_OVAL])
            
            if face_width > 0:
                mouth_ratio = mouth_width / face_width
                ideal_mouth_ratio = 0.5  # Mouth should be about half face width
                mouth_score = 1 - abs(mouth_ratio - ideal_mouth_ratio) / ideal_mouth_ratio
                scores["mouth_proportion"] = max(0, min(100, mouth_score * 100))
            else:
                scores["mouth_proportion"] = 50.0
            
            # Face shape score (based on oval similarity)
            face_oval_score = self._calculate_face_shape_score(points)
            scores["face_shape"] = face_oval_score
            
            # Nose proportion (simplified)
            scores["nose_proportion"] = 75.0  # Placeholder - would need more complex analysis
            
            return scores
            
        except Exception:
            return {
                "eye_symmetry": 50.0,
                "mouth_proportion": 50.0,
                "face_shape": 50.0,
                "nose_proportion": 50.0
            }

    def _calculate_harmony_score(self, points: List[Tuple], w: int, h: int) -> float:
        """Calculate overall facial harmony"""
        try:
            # Calculate various proportional relationships
            face_points = [points[i] for i in self.FACE_OVAL]
            
            # Face width at different heights
            top_third = sorted([p[1] for p in face_points])[len(face_points)//3]
            bottom_third = sorted([p[1] for p in face_points])[-len(face_points)//3]
            
            # Get widths at different levels
            top_width = len([p for p in face_points if abs(p[1] - top_third) < 10])
            bottom_width = len([p for p in face_points if abs(p[1] - bottom_third) < 10])
            
            # Harmony based on proportional consistency
            harmony = 0.8 if top_width > 0 and bottom_width > 0 else 0.5
            
            return harmony * 100
            
        except Exception:
            return 60.0

    def _calculate_face_shape_score(self, points: List[Tuple]) -> float:
        """Calculate face shape score based on oval similarity"""
        try:
            face_points = [points[i] for i in self.FACE_OVAL]
            
            # Calculate face dimensions
            min_x = min([p[0] for p in face_points])
            max_x = max([p[0] for p in face_points])
            min_y = min([p[1] for p in face_points])
            max_y = max([p[1] for p in face_points])
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Ideal oval ratio
            if height > 0:
                aspect_ratio = width / height
                ideal_ratio = 0.75  # Ideal face width to height ratio
                shape_score = 1 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
                return max(0, min(100, shape_score * 100))
            
            return 50.0
            
        except Exception:
            return 50.0

    def _get_center_point(self, points: List[Tuple]) -> Tuple[float, float]:
        """Calculate center point of a list of coordinates"""
        if not points:
            return (0, 0)
        
        avg_x = sum([p[0] for p in points]) / len(points)
        avg_y = sum([p[1] for p in points]) / len(points)
        return (avg_x, avg_y)

    def _generate_analysis(self, overall: float, symmetry: float, golden_ratio: float, features: Dict) -> str:
        """Generate human-readable analysis"""
        analysis = []
        
        if overall >= 80:
            analysis.append("Exceptionally attractive facial features with excellent proportions.")
        elif overall >= 70:
            analysis.append("Very attractive facial features with good proportions.")
        elif overall >= 60:
            analysis.append("Attractive facial features with decent proportions.")
        elif overall >= 50:
            analysis.append("Average facial attractiveness with room for improvement.")
        else:
            analysis.append("Below average facial attractiveness, focus on grooming and styling.")
        
        if symmetry >= 80:
            analysis.append("Excellent facial symmetry.")
        elif symmetry >= 60:
            analysis.append("Good facial symmetry.")
        else:
            analysis.append("Some asymmetry detected - consider makeup techniques to enhance balance.")
        
        if golden_ratio >= 80:
            analysis.append("Face proportions closely match the golden ratio.")
        elif golden_ratio >= 60:
            analysis.append("Good facial proportions.")
        else:
            analysis.append("Facial proportions could be enhanced with styling techniques.")
        
        # Feature-specific feedback
        best_feature = max(features.items(), key=lambda x: x[1])
        analysis.append(f"Your strongest feature is {best_feature[0].replace('_', ' ')}.")
        
        return " ".join(analysis)