# Face Recognition Attendance System

I built this project to explore how computer vision and deep learning can be combined to automate everyday tasks—specifically, tracking attendance without manual sign-in sheets. 

Instead of relying on pre-packaged recognition APIs, this repository walks through the entire machine learning workflow from scratch: capturing a custom face dataset via webcam, preprocessing the images, training a Convolutional Neural Network (CNN) in Keras, and logging live entry and exit timestamps straight into an Excel spreadsheet.

---

## How It Works

The project is split into five sequential scripts so each stage of the pipeline is easy to follow and debug:

### 1. Data Collection (`img_collect.py`)
Opens your webcam and uses OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to detect faces in real time. Once a face is found, the script crops the bounding box and collects **100 sample frames** per person. At the end of the capture, it prompts for the person's name and saves the images as `<Name>_0.jpg` through `<Name>_99.jpg` inside an automatically created `images/` folder.

### 2. Preprocessing & Serialization (`image_to_array.py`)
Neural networks need uniform inputs, so this script loops through every photo in the `images/` folder, resizes each face to **100×100 pixels**, and converts it to **grayscale** to keep the model lightweight. It extracts the person's name from the filename as the target label and packs both the image arrays and labels into binary pickle files (`array/images.p` and `array/labels.p`) for fast loading.

### 3. Training the CNN (`model.py`)
Loads the pickled dataset and prepares the images by applying **Histogram Equalization** (`cv2.equalizeHist`) to handle varying room lighting, followed by pixel normalization (`0–1` scaling). Labels are encoded alphabetically using Scikit-Learn's `LabelEncoder`.

The custom Keras CNN architecture includes:
* **Conv2D (128 filters, 5×5)** + **MaxPooling2D (2×2)**
* **Conv2D (64 filters, 3×3)** + **MaxPooling2D (2×2)**
* **Flatten** $\rightarrow$ **Dense (64)** $\rightarrow$ **Dense (32)** $\rightarrow$ **Softmax Output Layer**

Trained over 21 epochs using the Adam optimizer (`lr=0.01`), the final weights are exported to `Final_Model.h5`.

### 4. Live Model Testing (`test.py`)
A quick sanity-check script that loads `Final_Model.h5` and runs real-time inference on your webcam feed, drawing bounding boxes and predicted names on screen before you hook it up to the Excel logger.

### 5. Automated Attendance Logger (`att.py`)
The main application. It runs live face recognition alongside an interactive OpenCV overlay and manages daily Excel sheets (`Attendance_YYYY-MM-DD.xlsx`) using `openpyxl`:
* **Press `S` (Start):** Creates today's Excel sheet (if it doesn't already exist) and logs the **Name**, **Date**, and **Start Time** the first time a recognized face appears on camera. Duplicate entries for the same person are automatically filtered out.
* **Press `E` (End):** Stamps the **End Time** for everyone logged in that session and saves the workbook.
* **Press `Q` (Quit):** Safely releases the webcam and closes all windows.

---

## Tech Stack & Libraries
* **Language:** Python 3
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning:** TensorFlow / Keras
* **Data Manipulation:** NumPy, Scikit-Learn, Pickle
* **Spreadsheet Automation:** OpenPyXL, Matplotlib (for dataset inspection)

---

## Running the Project Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/MayankSahoo-NITJSR28/Face-Recognition-Attendance-System.git](https://github.com/MayankSahoo-NITJSR28/Face-Recognition-Attendance-System.git)
   cd Face-Recognition-Attendance-System
