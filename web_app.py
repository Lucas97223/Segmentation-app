import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import peanut_lib
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(layout="wide", page_title="Peanut Segmenter")

st.title("Peanut Segmentation Web App")

# --- Sidebar ---
st.sidebar.header("Data Input")

# Metadata Inputs
day_input = st.sidebar.text_input("Day", value="Day 1")
cond_input = st.sidebar.text_input("Condition", value="Control")
num_input = st.sidebar.text_input("Number", value="1")

# Image Upload
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Initialize Session State for Data
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Day", "Condition", "Number", "Seed", 
        "Area_cm2", "Area_px", "Scale_px_mm", "Filename", "Modified_by_App"
    ])

# --- Main Logic ---

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("Failed to load image.")
        st.stop()

    # State Management for Image Interaction
    # Reset state if new image uploaded
    if "current_image_name" not in st.session_state or st.session_state.current_image_name != uploaded_file.name:
        st.session_state.current_image_name = uploaded_file.name
        st.session_state.seeds = []
        st.session_state.manual_scale_pts = []
        st.session_state.boundary_lines = []
        st.session_state.boundary_click_queue = []
        st.session_state.last_click = None
        st.session_state.threshold = 127

    # --- Segmentation Controls ---
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.header("Controls")
        
        # Interaction Mode
        mode = st.radio("Interaction Mode", ["Select Seeds", "Draw Boundary", "Calibrate Scale"])
        
        st.subheader("Parameters")
        
        if mode == "Select Seeds":
            st.write("Click on seeds to segment them.")
            click_action = st.radio("Click Action", ["Add", "Remove"], horizontal=True)
            tolerance = st.slider("Color Tolerance", 0, 100, 50)
            
            if st.button("Clear Mask"):
                st.session_state.seeds = []
                st.rerun()
                
        elif mode == "Draw Boundary":
            st.write("Click two points to draw a barrier.")
            if st.button("Clear Boundaries"):
                st.session_state.boundary_lines = []
                st.session_state.boundary_click_queue = []
                st.rerun()

        # Morphology
        erode_iter = st.slider("Erosion (Shrink)", 0, 5, 0)
        dilate_iter = st.slider("Dilation (Grow)", 0, 5, 0)
        
        # Combined mask for display
        combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for mask in st.session_state.seeds:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            
        # Apply Morphology to display mask
        thresh_img = combined_mask.copy()
        if erode_iter > 0 or dilate_iter > 0:
            kernel = np.ones((3,3), np.uint8)
            if erode_iter > 0:
                thresh_img = cv2.erode(thresh_img, kernel, iterations=erode_iter)
            if dilate_iter > 0:
                thresh_img = cv2.dilate(thresh_img, kernel, iterations=dilate_iter)

        st.image(thresh_img, caption="Segmentation Mask", use_container_width=True)
        
        st.subheader("Scale Calibration")
        if mode == "Calibrate Scale":
            st.write("Click two points to define distance.")
        
        # Manual Scale Logic
        manual_scale_px_per_mm = 0.0
        
        if len(st.session_state.manual_scale_pts) == 2:
            p1 = np.array(st.session_state.manual_scale_pts[0])
            p2 = np.array(st.session_state.manual_scale_pts[1])
            dist_px = np.linalg.norm(p1 - p2)
            
            reference_cm = st.number_input("Reference Length (cm)", value=4.0)
            if reference_cm > 0:
                manual_scale_px_per_mm = dist_px / (reference_cm * 10)
                st.success(f"Scale: {manual_scale_px_per_mm:.2f} px/mm")
        elif mode == "Calibrate Scale":
            st.info("Waiting for points...")

        # Save Measurement Button
        if st.button("Save Measurement"):
            if manual_scale_px_per_mm <= 0:
                st.error("Please define the scale first.")
            else:
                new_rows = []
                vis_img = img.copy()
                overlay = vis_img.copy()
                
                for i, mask in enumerate(st.session_state.seeds):
                    # Apply morphology per seed
                    proc_mask = mask.copy()
                    if erode_iter > 0 or dilate_iter > 0:
                        kernel = np.ones((3,3), np.uint8)
                        if erode_iter > 0:
                            proc_mask = cv2.erode(proc_mask, kernel, iterations=erode_iter)
                        if dilate_iter > 0:
                            proc_mask = cv2.dilate(proc_mask, kernel, iterations=dilate_iter)
                    
                    # Find contours
                    seed_contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Calculate area
                    area_px = sum(cv2.contourArea(c) for c in seed_contours)
                    area_cm2 = area_px / ((manual_scale_px_per_mm * 10) ** 2)
                    
                    new_rows.append({
                        "Day": day_input,
                        "Condition": cond_input,
                        "Number": num_input,
                        "Seed": i,
                        "Area_cm2": area_cm2,
                        "Area_px": area_px,
                        "Scale_px_mm": manual_scale_px_per_mm,
                        "Filename": uploaded_file.name,
                        "Modified_by_App": True
                    })
                    
                    # Visualization
                    cv2.drawContours(overlay, seed_contours, -1, (0, 255, 0), -1)
                    cv2.drawContours(vis_img, seed_contours, -1, (0, 255, 0), 5)
                    
                    for c in seed_contours:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cv2.putText(vis_img, str(i), (cX - 20, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)

                if new_rows:
                    # Append to session state DataFrame
                    st.session_state.results_df = pd.concat([st.session_state.results_df, pd.DataFrame(new_rows)], ignore_index=True)
                    st.success(f"Added {len(new_rows)} seeds to results.")
                    
                    # Prepare Segmented Image for Download
                    cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0, vis_img)
                    if len(st.session_state.manual_scale_pts) == 2:
                         pt1 = tuple(map(int, st.session_state.manual_scale_pts[0]))
                         pt2 = tuple(map(int, st.session_state.manual_scale_pts[1]))
                         cv2.line(vis_img, pt1, pt2, (0, 0, 255), 8)
                    
                    is_success, buffer = cv2.imencode(".jpg", vis_img)
                    if is_success:
                        st.download_button(
                            label="Download Segmented Image",
                            data=io.BytesIO(buffer),
                            file_name=f"segmented_{uploaded_file.name}",
                            mime="image/jpeg"
                        )

    with col1:
        st.header("Image View")
        
        display_img = img.copy()
        overlay = display_img.copy()
        
        # Draw Seeds
        for i, mask in enumerate(st.session_state.seeds):
            proc_mask = mask.copy()
            if erode_iter > 0 or dilate_iter > 0:
                kernel = np.ones((3,3), np.uint8)
                if erode_iter > 0:
                    proc_mask = cv2.erode(proc_mask, kernel, iterations=erode_iter)
                if dilate_iter > 0:
                    proc_mask = cv2.dilate(proc_mask, kernel, iterations=dilate_iter)
                    
            seed_contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, seed_contours, -1, (0, 255, 0), -1)
            cv2.drawContours(display_img, seed_contours, -1, (0, 255, 0), 5)
            
            for c in seed_contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(display_img, str(i), (cX - 20, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)

        cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)
                
        # Draw Scale Line
        if len(st.session_state.manual_scale_pts) >= 1:
            for pt in st.session_state.manual_scale_pts:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
                
        if len(st.session_state.manual_scale_pts) == 2:
            pt1 = tuple(map(int, st.session_state.manual_scale_pts[0]))
            pt2 = tuple(map(int, st.session_state.manual_scale_pts[1]))
            cv2.line(display_img, pt1, pt2, (0, 0, 255), 8)
            
        # Draw Boundary Lines
        for line in st.session_state.boundary_lines:
            pt1 = tuple(map(int, line[0]))
            pt2 = tuple(map(int, line[1]))
            cv2.line(display_img, pt1, pt2, (0, 0, 0), 3)
            
        # Draw Pending Boundary Point
        for pt in st.session_state.boundary_click_queue:
            cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), -1)

        # Convert/Resize for Display
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        max_display_width = 800
        h, w = display_img_rgb.shape[:2]
        resize_factor = 1.0
        if w > max_display_width:
            resize_factor = w / max_display_width
            new_h = int(h / resize_factor)
            display_img_rgb = cv2.resize(display_img_rgb, (max_display_width, new_h))
        
        value = streamlit_image_coordinates(display_img_rgb, key="main_image")
        
        if value:
            click_coords = (value["x"], value["y"])
            
            if st.session_state.last_click != click_coords:
                st.session_state.last_click = click_coords
                
                # Map to original
                orig_x = value["x"] * resize_factor
                orig_y = value["y"] * resize_factor
                point = (orig_x, orig_y)
                
                if mode == "Calibrate Scale":
                    if len(st.session_state.manual_scale_pts) == 2:
                        st.session_state.manual_scale_pts = [point]
                    else:
                        st.session_state.manual_scale_pts.append(point)
                    st.rerun()
                    
                elif mode == "Draw Boundary":
                    st.session_state.boundary_click_queue.append(point)
                    if len(st.session_state.boundary_click_queue) == 2:
                        st.session_state.boundary_lines.append(tuple(st.session_state.boundary_click_queue))
                        st.session_state.boundary_click_queue = []
                    st.rerun()
                    
                elif mode == "Select Seeds":
                    seed_pt = (int(point[0]), int(point[1]))
                    
                    img_for_fill = img.copy()
                    for line in st.session_state.boundary_lines:
                        pt1 = tuple(map(int, line[0]))
                        pt2 = tuple(map(int, line[1]))
                        cv2.line(img_for_fill, pt1, pt2, (0, 0, 0), 3)
                    
                    new_region = peanut_lib.get_flood_fill_mask(img_for_fill, seed_pt, tolerance)
                    
                    target_idx = -1
                    for i, mask in enumerate(st.session_state.seeds):
                        if mask[seed_pt[1], seed_pt[0]] > 0:
                            target_idx = i
                            break
                    
                    if target_idx == -1:
                        for i, mask in enumerate(st.session_state.seeds):
                            overlap = cv2.bitwise_and(mask, new_region)
                            if cv2.countNonZero(overlap) > 0:
                                target_idx = i
                                break
                    
                    if click_action == "Add":
                        if target_idx != -1:
                            st.session_state.seeds[target_idx] = cv2.bitwise_or(st.session_state.seeds[target_idx], new_region)
                        else:
                            st.session_state.seeds.append(new_region)
                    else:
                        if target_idx != -1:
                            st.session_state.seeds[target_idx] = cv2.bitwise_and(st.session_state.seeds[target_idx], cv2.bitwise_not(new_region))
                    
                    st.rerun()

else:
    st.info("Please upload an image to start.")

# --- Results Download ---
st.sidebar.markdown("---")
st.sidebar.header("Results")

if not st.session_state.results_df.empty:
    st.sidebar.dataframe(st.session_state.results_df)
    
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="peanut_measurements.csv",
        mime="text/csv",
    )
