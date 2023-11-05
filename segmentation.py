import streamlit as st
import cv2
import numpy as np

# Define segmentation functions
def color_segmentation(image, lower_bound, upper_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image

def edge_segmentation(image, low_threshold, high_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    segmented_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return segmented_image

def contour_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmented_image = cv2.drawContours(np.zeros_like(image), contours, -1, (0, 255, 0), 3)
    return segmented_image

def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    markers = cv2.connectedComponents(thresh)[1]
    markers = markers + 1
    markers[thresh == 255] = 0
    segmented_image = image.copy()
    cv2.watershed(segmented_image, markers)
    return segmented_image

# Streamlit UI
st.title("Image Segmentation App")

st.write("Name: Siddhi Dinesh Mankar")
st.write("Roll No: 251")
st.write("Division: C")

# Set the background color using HTML
st.markdown(
    """
    <style>
    .stApp {
        background-color: #87CEEB; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image is not None:
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Display original image
    st.image(image, channels="BGR", use_column_width=True, caption="Original Image")

    # Choose segmentation method
    method = st.selectbox("Select Segmentation Method", ["Color", "Edge", "Contour", "Watershed"])

    if method == "Color":
        lower_bound = np.array([30, 50, 50])
        upper_bound = np.array([90, 255, 255])
        segmented_image = color_segmentation(image, lower_bound, upper_bound)

    elif method == "Edge":
        low_threshold = 50
        high_threshold = 150
        segmented_image = edge_segmentation(image, low_threshold, high_threshold)

    elif method == "Contour":
        segmented_image = contour_segmentation(image)

    elif method == "Watershed":
        segmented_image = watershed_segmentation(image)

    # Display segmented image
    st.image(segmented_image, channels="BGR", use_column_width=True, caption="Segmented Image")
