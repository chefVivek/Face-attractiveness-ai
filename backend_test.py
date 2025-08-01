import requests
import sys
import os
from datetime import datetime
import io
from PIL import Image
import numpy as np

class FaceAttractivenessAPITester:
    def __init__(self, base_url="https://638a81b3-bdc6-4ac9-abe9-c5ba03d98526.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else self.api_url
        headers = {}
        
        # Don't set Content-Type for multipart/form-data - requests will set it automatically
        if not files:
            headers['Content-Type'] = 'application/json'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files)
                else:
                    response = requests.post(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Non-dict response'}")
                except:
                    print(f"   Response: {response.text[:200]}...")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:500]}...")

            return success, response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_test_image(self, width=400, height=400):
        """Create a simple test image with a face-like pattern"""
        # Create a simple face-like image for testing
        img = Image.new('RGB', (width, height), color='white')
        pixels = img.load()
        
        # Draw a simple face pattern
        center_x, center_y = width // 2, height // 2
        
        # Face outline (circle)
        for x in range(width):
            for y in range(height):
                dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if 80 < dist < 120:  # Face outline
                    pixels[x, y] = (200, 180, 160)
                elif dist < 80:  # Face interior
                    pixels[x, y] = (220, 200, 180)
        
        # Eyes
        for eye_x in [center_x - 30, center_x + 30]:
            for x in range(eye_x - 10, eye_x + 10):
                for y in range(center_y - 20, center_y - 10):
                    if 0 <= x < width and 0 <= y < height:
                        pixels[x, y] = (50, 50, 50)
        
        # Mouth
        for x in range(center_x - 20, center_x + 20):
            for y in range(center_y + 20, center_y + 30):
                if 0 <= x < width and 0 <= y < height:
                    pixels[x, y] = (150, 100, 100)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    def test_root_endpoint(self):
        """Test the root API endpoint"""
        success, response = self.run_test(
            "Root API Endpoint",
            "GET",
            "",
            200
        )
        return success

    def test_analyze_face_with_valid_image(self):
        """Test face analysis with a valid image"""
        test_image = self.create_test_image()
        
        success, response = self.run_test(
            "Face Analysis - Valid Image",
            "POST",
            "analyze-face",
            200,
            files={'file': ('test_face.jpg', test_image, 'image/jpeg')}
        )
        
        if success and isinstance(response, dict):
            # Check required fields in response
            required_fields = ['overall_score', 'symmetry_score', 'golden_ratio_score', 'feature_breakdown', 'analysis']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            # Validate score ranges
            scores_valid = True
            for score_field in ['overall_score', 'symmetry_score', 'golden_ratio_score']:
                score = response.get(score_field, -1)
                if not (0 <= score <= 100):
                    print(f"⚠️  Invalid {score_field}: {score} (should be 0-100)")
                    scores_valid = False
            
            # Check feature breakdown
            if not isinstance(response.get('feature_breakdown'), dict):
                print("⚠️  feature_breakdown should be a dictionary")
                scores_valid = False
            
            # Check analysis text
            if not isinstance(response.get('analysis'), str) or len(response.get('analysis', '')) < 10:
                print("⚠️  analysis should be a meaningful text string")
                scores_valid = False
            
            if scores_valid:
                print("✅ Response structure and values are valid")
            
            return scores_valid
        
        return success

    def test_analyze_face_no_file(self):
        """Test face analysis without uploading a file"""
        success, response = self.run_test(
            "Face Analysis - No File",
            "POST",
            "analyze-face",
            422  # FastAPI returns 422 for validation errors
        )
        return success

    def test_analyze_face_invalid_file(self):
        """Test face analysis with invalid file type"""
        # Create a text file instead of image
        text_content = b"This is not an image file"
        
        success, response = self.run_test(
            "Face Analysis - Invalid File Type",
            "POST",
            "analyze-face",
            400,
            files={'file': ('test.txt', text_content, 'text/plain')}
        )
        return success

    def test_analyze_face_empty_file(self):
        """Test face analysis with empty file"""
        empty_content = b""
        
        success, response = self.run_test(
            "Face Analysis - Empty File",
            "POST",
            "analyze-face",
            400,
            files={'file': ('empty.jpg', empty_content, 'image/jpeg')}
        )
        return success

    def test_analysis_history(self):
        """Test getting analysis history"""
        success, response = self.run_test(
            "Analysis History",
            "GET",
            "analysis-history",
            200
        )
        
        if success and isinstance(response, list):
            print(f"✅ Retrieved {len(response)} analysis records")
        
        return success

    def test_status_endpoints(self):
        """Test status check endpoints"""
        # Test creating a status check
        test_data = {
            "client_name": f"test_client_{datetime.now().strftime('%H%M%S')}"
        }
        
        success1, response1 = self.run_test(
            "Create Status Check",
            "POST",
            "status",
            200,
            data=test_data
        )
        
        # Test getting status checks
        success2, response2 = self.run_test(
            "Get Status Checks",
            "GET",
            "status",
            200
        )
        
        return success1 and success2

def main():
    print("🚀 Starting Face Attractiveness API Tests")
    print("=" * 50)
    
    # Setup
    tester = FaceAttractivenessAPITester()
    
    # Run all tests
    test_results = []
    
    # Basic connectivity
    test_results.append(tester.test_root_endpoint())
    
    # Core face analysis functionality
    test_results.append(tester.test_analyze_face_with_valid_image())
    
    # Error handling tests
    test_results.append(tester.test_analyze_face_no_file())
    test_results.append(tester.test_analyze_face_invalid_file())
    test_results.append(tester.test_analyze_face_empty_file())
    
    # Additional endpoints
    test_results.append(tester.test_analysis_history())
    test_results.append(tester.test_status_endpoints())
    
    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed! Backend API is working correctly.")
        return 0
    else:
        failed_tests = tester.tests_run - tester.tests_passed
        print(f"⚠️  {failed_tests} test(s) failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())