# 🚗 Road Lane Detection  

## 🎯 Aim  
Develop a road lane detection system using Python and computer vision.  

## 🎯 Objective  
Accurately identify lane markings under various conditions (🌙 night, 🏎️ curves, 🌫️ fog, 🌧️ rain, 🦓 zebra crossings, 🏜️ sandy roads) for real-time lane detection in autonomous driving.  

## 🛠️ Methodology  
1️⃣ **Video Processing** 🎥: Capture and decode frames.  
2️⃣ **Preprocessing** 🖼️: Convert to grayscale, apply Gaussian blur, and detect edges using Canny.  
3️⃣ **ROI Selection** 🔍: Focus on lane areas.  
4️⃣ **Lane Detection** 📏: Use Hough Transform to identify lane lines.  
5️⃣ **Visualization** 🖍️: Overlay detected lanes on frames.  
6️⃣ **Real-time Processing & Evaluation** 🏎️: Assess accuracy for autonomous applications.  

## 🚀 Motivation  
- 🤖 **Autonomous Driving**: Enhances vehicle navigation.  
- 🚗 **ADAS Integration**: Provides lane departure warnings.  
- 🛡️ **Safety Improvement**: Reduces lane departure risks.  
- 🔬 **R&D**: Advances computer vision for self-driving tech.  

## 🎯 Expected Results  
Develop a lane detection system capable of processing both static images and dynamic video streams. The focus is on refining image-processing algorithms before advancing to video analysis.  

## 🏗️ Processing Steps  
1️⃣ **Grayscale Conversion** 🎨: Simplifies the image for better edge detection.  
2️⃣ **Gaussian Blur** 🌫️: Reduces noise and smooths intensity variations.  
3️⃣ **Canny Edge Detection** ✂️: Identifies significant intensity changes to highlight lane markings.  
4️⃣ **Zoomed Edge Detection** 🔍: Provides a closer view of detected edges.  
5️⃣ **Region Selection Mask** 🎯: Focuses on the road area where lane markings are typically present.  
6️⃣ **Final Output** 📸: Displays only lane markings within the selected region.  

---

## 📊 Evaluation and Enhancements  
Testing revealed challenges in detecting lanes on curved roads due to camera distortion, impacting accuracy. Enhancements include:  
✅ **Improved Curved Road Detection**: Optimized algorithms for accurate lane tracking.  
✅ **Directional Guidance**: Determines whether the vehicle should turn left or right based on lane curvature.  

---

## ⚠️ Challenges  
❌ **Curved Road Detection**: Non-linear lane patterns make tracking difficult.  
❌ **Lane Boundary Ambiguity**: Hard to distinguish merging or splitting lanes.  
❌ **Varying Lane Marking Patterns**: Road geometry impacts detection reliability.  
❌ **Limited Field of View**: Camera constraints affect lane visibility in curves.  

---

## 🏁 Curved Road Detection Algorithm  
1️⃣ **Calibration & Undistortion** 📷: Corrects lens distortion for accurate road representation.  
2️⃣ **Image Thresholding** 🎨: Uses Sobel edge detection and HLS color thresholding.  
3️⃣ **Lane Detection Pipeline** 🛣️:  
   - ✅ Corrects image distortion.  
   - ✅ Combines gradient and color thresholding for lane identification.  
   - ✅ Warps perspective for a bird’s-eye view.  
   - ✅ Uses a sliding window method to detect lanes.  
   - ✅ Draws lane boundaries and visualizes road conditions in real-time.  

---

## 🛑 Additional Challenges  
Despite improvements, challenges remain:  
🌫️ **Foggy & Rainy Weather** – Reduced lane visibility.  
🌙 **Night Driving** – Low lighting conditions.  
📸 **Improper Camera Angles** – Distorted lane perspective.  
🏜️ **Sandy Roads** – Faint or absent lane markings.  
🚶 **Zebra Crossings & Pedestrians** – Interference with lane tracking.  
🛣️ **Wide Lane Gaps** – Difficulty in continuous lane detection.  
🏙️ **Highway & City Roads** – Complex traffic and varying lane structures.  

---

## 🎯 Final Results  
✅ **Improved Curved Road Detection** – Enhanced accuracy in lane tracking.  
✅ **Directional Guidance** – Real-time navigation assistance.  

---

## 🔮 Future Scope  
Our system successfully detects lanes on straight and curved roads, integrating directional guidance.  
Future improvements focus on:  
🚗 Handling adverse weather conditions.  
🏙️ Urban environment adaptations.  
🚶 Better pedestrian & obstacle detection.  
These enhancements will make the system more robust and reliable for autonomous driving.  

---

## 🛠️ Installation
### 🔧 Requirements  
- Python 3.x 🐍  
- OpenCV 🖼️  
- NumPy 🔢  
- Matplotlib 📊  


