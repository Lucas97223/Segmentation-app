import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import segmentation_lib
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Page Config ---
st.set_page_config(
    layout="wide", 
    page_title="Universal Segmenter",
    page_icon="📐"
)

# --- Custom CSS for "Elegant" Look ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("Universal Segmentation Tool")
st.markdown("Analyze, segment, and measure objects in any image with AI-assisted tools.")

with st.expander("ℹ️ How it works"):
    st.markdown("""
    1.  **Upload** an image in the sidebar.
    2.  **Calibrate** the scale by clicking two points on a ruler (reference object).
    3.  **Segment** objects by clicking on them. The tool uses a flood-fill algorithm to find boundaries.
    4.  **Export** your data as a CSV file.
    """)

# --- Sidebar ---
st.sidebar.header("1. Project Settings")
object_name = st.sidebar.text_input("Object Name", value="Object", help="E.g., Seed, Cell, Leaf")
group_id = st.sidebar.text_input("Group / ID", value="Group A", help="Optional tag for this batch")

st.sidebar.header("2. Input")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Initialize Session State
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Label", "Group_ID", "Object_Index", 
        "Area_cm2", "Area_px", "Scale_px_mm", "Filename"
    ])

# --- Main Logic ---
if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None:
        st.error("Failed to load image.")
        st.stop()

    # Reset state if new image
    if "current_image_name" not in st.session_state or st.session_state.current_image_name != uploaded_file.name:
        st.session_state.current_image_name = uploaded_file.name
        st.session_state.seeds = [] # List of masks
        st.session_state.manual_scale_pts = []
        st.session_state.boundary_lines = []
        st.session_state.boundary_click_queue = []
        st.session_state.last_click = None

    # --- Layout ---
    col_tools, col_img = st.columns([1, 3])
    
    with col_tools:
        st.subheader("Tools")
        
        # Interaction Mode
        mode = st.radio("Mode", ["Select Objects", "Draw Boundary", "Calibrate Scale"])
        
        st.markdown("---")
        st.subheader("Settings")
        
        if mode == "Select Objects":
            st.info("Click on objects to add/remove.")
            click_action = st.radio("Action", ["Add", "Remove"], horizontal=True)
            tolerance = st.slider("Color Tolerance", 0, 100, 50, help="Higher value = broader color range")
            
            if st.button("Clear All Masks", type="primary"):
                st.session_state.seeds = []
                st.rerun()
                
        elif mode == "Draw Boundary":
            st.info("Click 2 points to draw a barrier line.")
            if st.button("Clear Boundaries"):
                st.session_state.boundary_lines = []
                st.session_state.boundary_click_queue = []
                st.rerun()
                
        elif mode == "Calibrate Scale":
            st.info("Click 2 points on your ruler.")
            reference_cm = st.number_input("Reference Length (cm)", value=1.0, min_value=0.1)

        # Morphology
        with st.expander("Advanced Options"):
            erode_iter = st.slider("Erosion (Shrink)", 0, 5, 0)
            dilate_iter = st.slider("Dilation (Grow)", 0, 5, 0)

        # Scale Logic
        manual_scale_px_per_mm = 0.0
        if len(st.session_state.manual_scale_pts) == 2:
            p1 = np.array(st.session_state.manual_scale_pts[0])
            p2 = np.array(st.session_state.manual_scale_pts[1])
            dist_px = np.linalg.norm(p1 - p2)
            if reference_cm > 0:
                manual_scale_px_per_mm = dist_px / (reference_cm * 10)
                st.success(f"Scale: {manual_scale_px_per_mm:.2f} px/mm")
        
        # Save Button
        st.markdown("---")
        if st.button("💾 Save to Results", use_container_width=True):
            if manual_scale_px_per_mm <= 0:
                st.warning("⚠️ Please calibrate scale first!")
            else:
                new_rows = []
                vis_img = img.copy()
                overlay = vis_img.copy()
                
                for i, mask in enumerate(st.session_state.seeds):
                    # Morphology
                    proc_mask = mask.copy()
                    if erode_iter > 0 or dilate_iter > 0:
                        kernel = np.ones((3,3), np.uint8)
                        if erode_iter > 0: proc_mask = cv2.erode(proc_mask, kernel, iterations=erode_iter)
                        if dilate_iter > 0: proc_mask = cv2.dilate(proc_mask, kernel, iterations=dilate_iter)
                    
                    # Contours
                    contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    area_px = sum(cv2.contourArea(c) for c in contours)
                    area_cm2 = area_px / ((manual_scale_px_per_mm * 10) ** 2)
                    
                    new_rows.append({
                        "Label": object_name,
                        "Group_ID": group_id,
                        "Object_Index": i + 1,
                        "Area_cm2": round(area_cm2, 4),
                        "Area_px": int(area_px),
                        "Scale_px_mm": round(manual_scale_px_per_mm, 2),
                        "Filename": uploaded_file.name
                    })
                    
                    # Visualization for Download
                    cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
                    cv2.drawContours(vis_img, contours, -1, (0, 255, 0), 2)
                    for c in contours:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                            cv2.putText(vis_img, str(i+1), (cX - 10, cY + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if new_rows:
                    st.session_state.results_df = pd.concat([st.session_state.results_df, pd.DataFrame(new_rows)], ignore_index=True)
                    st.toast(f"Saved {len(new_rows)} objects!", icon="✅")
                    
                    # Image Download
                    cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
                    is_success, buffer = cv2.imencode(".jpg", vis_img)
                    if is_success:
                        st.download_button(
                            label="Download Segmented Image",
                            data=io.BytesIO(buffer),
                            file_name=f"segmented_{uploaded_file.name}",
                            mime="image/jpeg"
                        )

    with col_img:
        # Live Stats
        m1, m2, m3 = st.columns(3)
        num_objects = len(st.session_state.seeds)
        m1.metric("Objects", num_objects)
        
        # Calculate live area if scale exists
        total_area = 0
        if manual_scale_px_per_mm > 0 and num_objects > 0:
            for mask in st.session_state.seeds:
                area_px = cv2.countNonZero(mask)
                total_area += area_px / ((manual_scale_px_per_mm * 10) ** 2)
            m2.metric("Total Area", f"{total_area:.2f} cm²")
            m3.metric("Avg Area", f"{total_area/num_objects:.2f} cm²")
        else:
            m2.metric("Total Area", "--")
            m3.metric("Avg Area", "--")

        # Image Processing for Display
        display_img = img.copy()
        overlay = display_img.copy()
        
        # Draw Masks
        for i, mask in enumerate(st.session_state.seeds):
            proc_mask = mask.copy()
            if erode_iter > 0 or dilate_iter > 0:
                kernel = np.ones((3,3), np.uint8)
                if erode_iter > 0: proc_mask = cv2.erode(proc_mask, kernel, iterations=erode_iter)
                if dilate_iter > 0: proc_mask = cv2.dilate(proc_mask, kernel, iterations=dilate_iter)
            
            contours, _ = cv2.findContours(proc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
            cv2.drawContours(display_img, contours, -1, (0, 255, 0), 2)
            
            for c in contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.putText(display_img, str(i+1), (cX - 10, cY + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
        
        # Draw Overlays (Scale/Boundary)
        if len(st.session_state.manual_scale_pts) >= 1:
            for pt in st.session_state.manual_scale_pts:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        if len(st.session_state.manual_scale_pts) == 2:
            cv2.line(display_img, tuple(map(int, st.session_state.manual_scale_pts[0])), tuple(map(int, st.session_state.manual_scale_pts[1])), (0, 0, 255), 2)
            
        for line in st.session_state.boundary_lines:
            cv2.line(display_img, tuple(map(int, line[0])), tuple(map(int, line[1])), (0, 0, 0), 2)
        for pt in st.session_state.boundary_click_queue:
            cv2.circle(display_img, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)

        # Resize for Streamlit Component
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        max_width = 800
        h, w = display_img_rgb.shape[:2]
        resize_factor = 1.0
        if w > max_width:
            resize_factor = w / max_width
            display_img_rgb = cv2.resize(display_img_rgb, (max_width, int(h / resize_factor)))
            
        # Interactive Component
        value = streamlit_image_coordinates(display_img_rgb, key="main_image")
        
        # Handle Clicks
        if value:
            click_coords = (value["x"], value["y"])
            if st.session_state.last_click != click_coords:
                st.session_state.last_click = click_coords
                point = (value["x"] * resize_factor, value["y"] * resize_factor)
                
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
                    
                elif mode == "Select Objects":
                    seed_pt = (int(point[0]), int(point[1]))
                    
                    # Prepare image with barriers
                    img_fill = img.copy()
                    for line in st.session_state.boundary_lines:
                        cv2.line(img_fill, tuple(map(int, line[0])), tuple(map(int, line[1])), (0,0,0), 3)
                        
                    new_mask = segmentation_lib.get_flood_fill_mask(img_fill, seed_pt, tolerance)
                    
                    # Check overlap
                    target_idx = -1
                    for i, m in enumerate(st.session_state.seeds):
                        if m[seed_pt[1], seed_pt[0]] > 0 or cv2.countNonZero(cv2.bitwise_and(m, new_mask)) > 0:
                            target_idx = i
                            break
                    
                    if click_action == "Add":
                        if target_idx != -1:
                            st.session_state.seeds[target_idx] = cv2.bitwise_or(st.session_state.seeds[target_idx], new_mask)
                        else:
                            st.session_state.seeds.append(new_mask)
                    else: # Remove
                        if target_idx != -1:
                            st.session_state.seeds[target_idx] = cv2.bitwise_and(st.session_state.seeds[target_idx], cv2.bitwise_not(new_mask))
                            if cv2.countNonZero(st.session_state.seeds[target_idx]) == 0:
                                st.session_state.seeds.pop(target_idx)
                    st.rerun()

else:
    st.info("👈 Upload an image in the sidebar to get started!")

# --- Footer / Export ---
st.sidebar.markdown("---")
st.sidebar.header("3. Export")
if not st.session_state.results_df.empty:
    st.sidebar.dataframe(st.session_state.results_df, height=150)
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV", csv, "segmentation_results.csv", "text/csv")
