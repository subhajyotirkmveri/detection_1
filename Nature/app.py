import os
import cv2
import streamlit as st
import numpy as np  # Added NumPy import
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('/home/sysadm/Downloads/Nature/runs/detect/train/weights/best.pt')

# Decoding according to the .yaml file class names order
decoding_of_predictions = {0: 'squirrel', 1: 'butterfly'}

def main():
    st.title("Object Detection with YOLOv8")
    st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)      
    st.markdown("<p>🚀Welcome to the introduction page of our project! In this project, we will be exploring the YOLO (You Only Look Once) algorithm. YOLO is known for its ability to detect objects in an image in a single pass, making it a highly efficient and accurate object detection algorithm.🚀</p>", unsafe_allow_html=True)  
    st.markdown("<p>The latest version of YOLO, YOLOv8, released in January 2023 by Ultralytics, has introduced several modifications that have further improved its performance. 🌟</p>", unsafe_allow_html=True)
    st.write("🎯 Note : This project specifically focuses on classifying images containing butterflies and squirrels on any image 🎯")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Objects"):
            detect_objects(image)

def detect_objects(image):
    # Convert PIL image to numpy array
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model.predict(img_np, save=True, iou=0.5, save_txt=True, conf=0.25)

    # Define colors for different classes
    class_colors = {'squirrel': (255, 0, 0),    # Red for squirrel
                    'butterfly': (0, 0, 255)}  # Blue for butterfly

    for r in results:
        conf_list = r.boxes.conf.numpy().tolist()
        clss_list = r.boxes.cls.numpy().tolist()
        original_list = clss_list
        updated_list = []
        for element in original_list:
            updated_list.append(decoding_of_predictions[int(element)])

        bounding_boxes = r.boxes.xyxy.numpy()
        confidences = conf_list
        class_names = updated_list

        # Draw bounding boxes on the image
        for bbox, conf, cls in zip(bounding_boxes, confidences, class_names):
            x1, y1, x2, y2 = bbox.astype(int)
            
            # Get color for the bounding box based on the class
            box_color = class_colors.get(cls, (128, 128, 128))  # Default to grey if class color is not defined
            
            # Use the same color for both bounding box and text
            text_color = box_color
                
            cv2.rectangle(img_np, (x1, y1), (x2, y2), box_color, 2)
            
            # Adjust the position of the text to be at the top left corner of the bounding box
            text_position = (x1, y1 - 10) if y1 >= 20 else (x1, y1 + 20)
            
            # Display the label and confidence score on top of the bounding box
            label = f"{cls} {conf:.2f}"
            cv2.putText(img_np, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    # Convert numpy array back to PIL image
    img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

    st.image(img_pil, caption="Output Image", use_column_width=True)
    st.write("Feel free to upload another image to view its results.")



if __name__ == "__main__":
    main()

