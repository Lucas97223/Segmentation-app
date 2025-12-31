import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import zipfile
import segmentation_lib
from streamlit_image_coordinates import streamlit_image_coordinates

# --- Page Config ---
st.set_page_config(
    layout="wide", 
    page_title="SegmentAI",
    page_icon="📐"
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Import Inter Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Global Variables */
    :root {
        --bg-color: #0f172a;
        --text-color: #f8fafc;
        --accent-gradient: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    /* Main App Background */
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-color);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid var(--glass-border);
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: var(--text-color) !important;
    }
    
    /* Gradient Text Class */
    .gradient-text {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 2rem;
        font-weight: 600;
        border: 1px solid var(--glass-border);
        background-color: var(--glass-bg);
        color: var(--text-color);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--glass-bg);
        border-radius: 1rem;
        color: var(--text-color);
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: var(--text-color); }
    div[data-testid="stMetricLabel"] { color: var(--secondary-color); }
    
    /* --- Fixes for User Feedback --- */
    
    /* 1. Remove White Band (Header) */
    header[data-testid="stHeader"] {
        background: transparent;
    }
    
    /* 2. High Contrast Text (Radio Buttons, Markdown, Input Labels) */
    .stRadio label p, 
    .stMarkdown p, 
    .stTextInput label p, 
    .stNumberInput label p, 
    .stSelectbox label p, 
    .stSlider label p {
        color: var(--text-color) !important;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* 3. Info/Alert Boxes */
    .stAlert {
        background-color: rgba(59, 130, 246, 0.15); /* Subtle blue tint */
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: var(--text-color);
    }
    
    /* Ensure icons in alerts are visible */
    .stAlert > div {
        color: var(--text-color) !important;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<h1>📐 <span class="gradient-text">SegmentAI</span></h1>', unsafe_allow_html=True)
st.markdown("Advanced object segmentation, measurement, and analysis tool.")

# --- Sidebar ---
st.sidebar.header("1. Project & Image")
# No more global "Object Name" here, we do it per object
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# --- Session State Initialization ---
if "objects" not in st.session_state:
    st.session_state.objects = [] # List of dicts: {'id': int, 'name': str, 'mask': array, 'metrics': dict}
if "next_id" not in st.session_state:
    st.session_state.next_id = 1
if "scale_px_per_unit" not in st.session_state:
    st.session_state.scale_px_per_unit = None
if "unit_name" not in st.session_state:
    st.session_state.unit_name = "px"
if "barrier_thickness" not in st.session_state:
    st.session_state.barrier_thickness = 10
if "select_action" not in st.session_state:
    st.session_state.select_action = "add" # "add" or "remove"
if "pending_mask" not in st.session_state:
    st.session_state.pending_mask = None
if "pending_conflict_idx" not in st.session_state:
    st.session_state.pending_conflict_idx = None

# --- Main Logic ---
if uploaded_file is not None:
    # Read Image
    # Image loaded in state check below
    
    # Image loaded in state check below

    # Reset state if new image
    if "current_image_name" not in st.session_state or st.session_state.current_image_name != uploaded_file.name:
        st.session_state.current_image_name = uploaded_file.name
        
        # Load and store image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        st.session_state.working_image = cv2.imdecode(file_bytes, 1)
        
        st.session_state.objects = []
        st.session_state.next_id = 1
        st.session_state.manual_scale_pts = []
        st.session_state.boundary_polys = [] # List of lists of points
        st.session_state.current_boundary_points = [] # Points for current poly being drawn
        st.session_state.last_click = None
        st.session_state.scale_px_per_unit = None
        st.session_state.unit_name = "px"
        st.session_state.pending_mask = None
        st.session_state.pending_conflict_idx = None
        st.session_state.select_action = "add"
        
        # Zoom/View State
        st.session_state.zoom_level = 1.0
        st.session_state.viewport_origin = (0, 0) # (x, y) top-left

    img = st.session_state.working_image
    col_tools, col_img = st.columns([1, 3])
    
    with col_tools:
        st.subheader("Tools")
        
        # --- Conflict Resolution Dialog ---
        if st.session_state.pending_mask is not None:
            conflict_idx = st.session_state.pending_conflict_idx
            conflict_name = st.session_state.objects[conflict_idx]['name']
            
            st.warning(f"⚠️ Overlap detected with **{conflict_name}**!")
            
            col_merge, col_sep, col_cancel = st.columns(3)
            
            if col_merge.button("Merge"):
                # Merge into existing object
                st.session_state.objects[conflict_idx]['mask'] = cv2.bitwise_or(
                    st.session_state.objects[conflict_idx]['mask'], 
                    st.session_state.pending_mask
                )
                # Recalculate metrics for the merged object
                st.session_state.objects[conflict_idx]['metrics'] = segmentation_lib.calculate_metrics(
                    st.session_state.objects[conflict_idx]['mask'], 
                    st.session_state.scale_px_per_unit
                )
                
                st.session_state.pending_mask = None
                st.session_state.pending_conflict_idx = None
                st.rerun()
                
            if col_sep.button("Separate"):
                # Resolve overlap cleanly
                existing_mask = st.session_state.objects[conflict_idx]['mask']
                new_mask_candidate = st.session_state.pending_mask
                
                # Get non-overlapping versions
                resolved_existing, resolved_new = segmentation_lib.resolve_overlap(existing_mask, new_mask_candidate)
                
                # Update existing object
                st.session_state.objects[conflict_idx]['mask'] = resolved_existing
                st.session_state.objects[conflict_idx]['metrics'] = segmentation_lib.calculate_metrics(
                    resolved_existing, 
                    st.session_state.scale_px_per_unit
                )
                
                # Add new object
                metrics = segmentation_lib.calculate_metrics(resolved_new, st.session_state.scale_px_per_unit)
                new_obj = {
                    'id': st.session_state.next_id,
                    'name': f"Object {st.session_state.next_id}", 
                    'mask': resolved_new,
                    'metrics': metrics
                }
                st.session_state.objects.append(new_obj)
                st.session_state.next_id += 1
                
                st.session_state.pending_mask = None
                st.session_state.pending_conflict_idx = None
                st.rerun()
                
            if col_cancel.button("Cancel"):
                st.session_state.pending_mask = None
                st.session_state.pending_conflict_idx = None
                st.rerun()
                
        st.markdown("---") # Separator

        mode = st.radio("Mode", ["🖱️ Select Objects", "✏️ Draw Boundary", "📏 Calibrate Scale"])
        
        st.markdown("---")
        
        # --- Calibration Controls ---
        if mode == "📏 Calibrate Scale":
            st.info("Click 2 points on a known reference.")
            cal_val = st.number_input("Reference Value", value=1.0, min_value=0.01)
            cal_unit = st.selectbox("Unit", ["cm", "mm", "in", "px", "ft", "m"])
            
            if len(st.session_state.manual_scale_pts) == 2:
                p1 = np.array(st.session_state.manual_scale_pts[0])
                p2 = np.array(st.session_state.manual_scale_pts[1])
                dist_px = np.linalg.norm(p1 - p2)
                
                if cal_unit != "px":
                    scale = dist_px / cal_val
                    st.session_state.scale_px_per_unit = scale
                    st.session_state.unit_name = cal_unit
                    st.success(f"Scale: {scale:.2f} px/{cal_unit}")
                else:
                    st.session_state.scale_px_per_unit = None
                    st.session_state.unit_name = "px"
                    st.info("Scale set to pixels (1:1).")
            
            if st.button("Reset Calibration"):
                st.session_state.manual_scale_pts = []
                st.session_state.scale_px_per_unit = None
                st.session_state.unit_name = "px"
                st.rerun()

        # --- Segmentation Controls ---
        elif mode == "🖱️ Select Objects":
            st.info("Click to Add/Remove objects.")
            
            # Action Buttons
            col_a1, col_a2 = st.columns(2) # Or vertical? User said "buttons on top of eachother".
            # Let's do vertical as requested for Draw Boundary, assuming consistency.
            
            if st.button("➕ Add New Object", type="primary" if st.session_state.select_action == "add" else "secondary", use_container_width=True):
                st.session_state.select_action = "add"
                st.rerun()
            
            if st.button("🗑️ Remove Object", type="primary" if st.session_state.select_action == "remove" else "secondary", use_container_width=True):
                st.session_state.select_action = "remove"
                st.rerun()
            
            if st.session_state.select_action == "add":
                next_name = st.text_input("Name for next object", value=f"Object {st.session_state.next_id}")
            
            tolerance = st.slider("Color Tolerance", 0, 100, 50)
            
            st.markdown("---")
            if st.button("🗑️ Clear All Objects", type="primary", use_container_width=True):
                st.session_state.objects = []
                st.session_state.next_id = 1
                st.rerun()

        # --- Boundary Controls ---
        elif mode == "✏️ Draw Boundary":
            st.info("Click points to draw. Use buttons to finish.")
            
            st.subheader("Actions")
            if st.button("🏁 Finish Line", use_container_width=True):
                if len(st.session_state.current_boundary_points) > 1:
                    st.session_state.boundary_polys.append(st.session_state.current_boundary_points)
                    st.session_state.current_boundary_points = []
                    st.rerun()
            
            if st.button("🔄 Close Loop", use_container_width=True):
                if len(st.session_state.current_boundary_points) > 2:
                    # Append first point to end to close it visually/logically
                    pts = st.session_state.current_boundary_points
                    pts.append(pts[0])
                    st.session_state.boundary_polys.append(pts)
                    st.session_state.current_boundary_points = []
                    st.rerun()
            
            if st.button("↩️ Undo Point", use_container_width=True):
                if st.session_state.current_boundary_points:
                    st.session_state.current_boundary_points.pop()
                    st.rerun()
            
            st.markdown("---")
            st.session_state.barrier_thickness = st.slider(
                "Barrier Thickness", 1, 50, st.session_state.barrier_thickness
            )
            
            st.markdown("---")
            if st.button("🗑️ Clear All Boundaries", type="primary", use_container_width=True):
                st.session_state.boundary_polys = []
                st.session_state.current_boundary_points = []
                st.rerun()

        # --- Export ---
        st.markdown("---")
        st.subheader("Export")
        
        # Prepare DataFrame
        data = []
        for obj in st.session_state.objects:
            row = {
                "ID": obj['id'],
                "Name": obj['name'],
                **obj['metrics']
            }
            data.append(row)
        df = pd.DataFrame(data)
        
        if not df.empty:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "segmentation_results.csv", "text/csv")
            
            if st.button("Download ZIP (Images)"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for obj in st.session_state.objects:
                        mask = obj['mask']
                        x, y, w, h = cv2.boundingRect(mask)
                        crop = img[y:y+h, x:x+w]
                        mask_crop = mask[y:y+h, x:x+w]
                        crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                        crop_rgba[:, :, 3] = mask_crop
                        is_success, buffer = cv2.imencode(".png", crop_rgba)
                        if is_success:
                            zf.writestr(f"{obj['name']}_{obj['id']}.png", buffer.tobytes())
                
                st.download_button("Download ZIP", zip_buffer.getvalue(), "objects.zip", "application/zip")

    with col_img:
        # --- View Controls Bar ---
        col_v1, col_v2, col_v3, col_v4 = st.columns([1, 1, 2, 1])
        
        if col_v1.button("↺ Left"):
            # Rotate Image
            st.session_state.working_image = cv2.rotate(st.session_state.working_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = st.session_state.working_image.shape[:2]
            
            # Rotate Objects
            for obj in st.session_state.objects:
                obj['mask'] = cv2.rotate(obj['mask'], cv2.ROTATE_90_COUNTERCLOCKWISE)
                # Re-calc metrics (width/height swap)
                obj['metrics'] = segmentation_lib.calculate_metrics(obj['mask'], st.session_state.scale_px_per_unit)
            
            # Rotate Boundaries
            new_polys = []
            for poly in st.session_state.boundary_polys:
                new_poly = []
                for pt in poly:
                    x, y = pt
                    new_x = y
                    new_y = h - 1 - x
                    new_poly.append((new_x, new_y))
                new_polys.append(new_poly)
            st.session_state.boundary_polys = new_polys
            
            st.session_state.manual_scale_pts = []
            st.rerun()

        if col_v2.button("↻ Right"):
            # Rotate Image
            st.session_state.working_image = cv2.rotate(st.session_state.working_image, cv2.ROTATE_90_CLOCKWISE)
            h, w = st.session_state.working_image.shape[:2]
            
            # Rotate Objects
            for obj in st.session_state.objects:
                obj['mask'] = cv2.rotate(obj['mask'], cv2.ROTATE_90_CLOCKWISE)
                obj['metrics'] = segmentation_lib.calculate_metrics(obj['mask'], st.session_state.scale_px_per_unit)
            
            # Rotate Boundaries
            new_polys = []
            for poly in st.session_state.boundary_polys:
                new_poly = []
                for pt in poly:
                    x, y = pt
                    new_x = w - 1 - y
                    new_y = x
                    new_poly.append((new_x, new_y))
                new_polys.append(new_poly)
            st.session_state.boundary_polys = new_polys
            
            st.session_state.manual_scale_pts = []
            st.rerun()
        
        with col_v3:
            st.session_state.zoom_level = st.slider("Zoom", 1.0, 5.0, st.session_state.zoom_level, 0.1, label_visibility="collapsed")
        
        if col_v4.button("Reset"):
            st.session_state.zoom_level = 1.0
            st.session_state.viewport_origin = (0, 0)
            st.rerun()

        # --- Image Processing & Display ---
        # 1. Calculate Viewport
        h_full, w_full = img.shape[:2]
        view_w = int(w_full / st.session_state.zoom_level)
        view_h = int(h_full / st.session_state.zoom_level)
        
        # Ensure viewport is within bounds
        vx, vy = st.session_state.viewport_origin
        vx = max(0, min(vx, w_full - view_w))
        vy = max(0, min(vy, h_full - view_h))
        st.session_state.viewport_origin = (vx, vy)
        
        # Crop Image
        display_img = img[vy:vy+view_h, vx:vx+view_w].copy()
        overlay = display_img.copy()
        
        # Coordinate Helpers
        def global_to_screen(pt):
            return (pt[0] - vx, pt[1] - vy)
            
        def screen_to_global(pt):
            return (pt[0] + vx, pt[1] + vy)
            
        # Draw Objects (Offset by viewport)
        for obj in st.session_state.objects:
            mask = obj['mask']
            # We need to draw contours on the cropped overlay.
            # Efficient way: Crop the mask, find contours on crop.
            mask_crop = mask[vy:vy+view_h, vx:vx+view_w]
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Color based on ID (simple hash)
            np.random.seed(obj['id'])
            color = np.random.randint(0, 255, 3).tolist()
            
            cv2.drawContours(overlay, contours, -1, color, -1)
            cv2.drawContours(display_img, contours, -1, color, 2)
            
            # Label
            for c in contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.putText(display_img, str(obj['id']), (cX - 10, cY + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)
        
        # Draw Calibration
        if len(st.session_state.manual_scale_pts) >= 1:
            for pt in st.session_state.manual_scale_pts:
                s_pt = global_to_screen(pt)
                cv2.circle(display_img, (int(s_pt[0]), int(s_pt[1])), 5, (0, 0, 255), -1)
        if len(st.session_state.manual_scale_pts) == 2:
            s_p1 = global_to_screen(st.session_state.manual_scale_pts[0])
            s_p2 = global_to_screen(st.session_state.manual_scale_pts[1])
            cv2.line(display_img, tuple(map(int, s_p1)), tuple(map(int, s_p2)), (0, 0, 255), 2)
            
            # Draw label
            mid_x = int((s_p1[0] + s_p2[0]) / 2)
            mid_y = int((s_p1[1] + s_p2[1]) / 2)
            if st.session_state.scale_px_per_unit:
                label = f"{cal_val} {cal_unit}"
                cv2.putText(display_img, label, (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw Boundaries (Polylines)
        # 1. Completed Polys
        for poly in st.session_state.boundary_polys:
            # Shift points
            shifted_poly = [global_to_screen(pt) for pt in poly]
            pts = np.array(shifted_poly, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], False, (0, 0, 0), st.session_state.barrier_thickness)
            
        # 2. Current Poly being drawn
        if st.session_state.current_boundary_points:
            shifted_poly = [global_to_screen(pt) for pt in st.session_state.current_boundary_points]
            pts = np.array(shifted_poly, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], False, (255, 0, 0), 2)
            for pt in shifted_poly:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1)

        # Pan Controls (Overlay on top or separate?)
        # Let's put Pan controls ABOVE the image if zoomed
        if st.session_state.zoom_level > 1.0:
            col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns([1, 1, 1, 1, 6])
            step = int(min(view_w, view_h) * 0.2)
            
            if col_p2.button("⬆️"):
                st.session_state.viewport_origin = (vx, max(0, vy - step))
                st.rerun()
            if col_p1.button("⬅️"):
                st.session_state.viewport_origin = (max(0, vx - step), vy)
                st.rerun()
            if col_p3.button("➡️"):
                st.session_state.viewport_origin = (min(w_full - view_w, vx + step), vy)
                st.rerun()
            if col_p2.button("⬇️"): # Reusing col_p2 for vertical stack effect? No, Streamlit columns are horizontal.
                # Actually, standard arrow layout is hard with columns.
                # Let's just do Left/Up/Down/Right in a row.
                pass
            
            # Better Pan Layout:
            #  [ ^ ]
            # [<] [>]
            #  [ v ]
            # This is hard. Let's just do [Left] [Up] [Down] [Right]
            
        # Resize for Streamlit Display
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        max_width = 800
        h_disp, w_disp = display_img_rgb.shape[:2]
        resize_factor = 1.0
        if w_disp > max_width:
            resize_factor = w_disp / max_width
            display_img_rgb = cv2.resize(display_img_rgb, (max_width, int(h_disp / resize_factor)))
            
        value = streamlit_image_coordinates(display_img_rgb, key="main_image")
        
        # --- Click Handling ---
        if value:
            click_coords = (value["x"], value["y"])
            if st.session_state.last_click != click_coords:
                st.session_state.last_click = click_coords
                
                # 1. Map click to Viewport (Screen) coordinates
                screen_x = value["x"] * resize_factor
                screen_y = value["y"] * resize_factor
                
                # 2. Map Viewport to Global coordinates
                global_pt = screen_to_global((screen_x, screen_y))
                point = global_pt # Use this for all logic
                
                if mode == "📏 Calibrate Scale":
                    if len(st.session_state.manual_scale_pts) == 2:
                        st.session_state.manual_scale_pts = [point]
                    else:
                        st.session_state.manual_scale_pts.append(point)
                    st.rerun()
                    
                elif mode == "✏️ Draw Boundary":
                    st.session_state.current_boundary_points.append(point)
                    st.rerun()
                    
                elif mode == "🖱️ Select Objects":
                    seed_pt = (int(point[0]), int(point[1]))
                    
                    if st.session_state.select_action == "remove":
                        # Find which object mask contains the point
                        to_remove_idx = -1
                        for i, obj in enumerate(st.session_state.objects):
                            # Check bounds first
                            h_m, w_m = obj['mask'].shape
                            if 0 <= seed_pt[1] < h_m and 0 <= seed_pt[0] < w_m:
                                if obj['mask'][seed_pt[1], seed_pt[0]] > 0:
                                    to_remove_idx = i
                                    break
                        
                        if to_remove_idx != -1:
                            # Use Smart Cut
                            original_mask = st.session_state.objects[to_remove_idx]['mask']
                            new_mask = segmentation_lib.smart_cut(
                                original_mask, 
                                st.session_state.boundary_polys, 
                                seed_pt,
                                thickness=st.session_state.barrier_thickness
                            )
                            
                            if cv2.countNonZero(new_mask) == 0:
                                # Fully removed
                                st.session_state.objects.pop(to_remove_idx)
                            else:
                                # Update mask and metrics
                                st.session_state.objects[to_remove_idx]['mask'] = new_mask
                                st.session_state.objects[to_remove_idx]['metrics'] = segmentation_lib.calculate_metrics(
                                    new_mask, st.session_state.scale_px_per_unit
                                )
                            st.rerun()
                            
                    else: # Add New Object
                        # Pass boundaries to flood fill for hard barriers
                        new_mask = segmentation_lib.get_flood_fill_mask(
                            img, 
                            seed_pt, 
                            tolerance, 
                            boundaries=st.session_state.boundary_polys,
                            thickness=st.session_state.barrier_thickness
                        )
                        
                        # Check for overlap
                        target_idx = -1
                        for i, obj in enumerate(st.session_state.objects):
                            # Check intersection
                            overlap = cv2.bitwise_and(obj['mask'], new_mask)
                            if cv2.countNonZero(overlap) > 0:
                                target_idx = i
                                break
                        
                        if target_idx != -1:
                            # Conflict detected!
                            st.session_state.pending_mask = new_mask
                            st.session_state.pending_conflict_idx = target_idx
                            st.rerun()
                        else:
                            # No spatial conflict. Check if name exists for merging.
                            existing_obj_idx = -1
                            for i, obj in enumerate(st.session_state.objects):
                                if obj['name'] == next_name:
                                    existing_obj_idx = i
                                    break
                            
                            if existing_obj_idx != -1:
                                # Merge into existing object by name
                                target_obj = st.session_state.objects[existing_obj_idx]
                                merged_mask = cv2.bitwise_or(target_obj['mask'], new_mask)
                                
                                target_obj['mask'] = merged_mask
                                target_obj['metrics'] = segmentation_lib.calculate_metrics(merged_mask, st.session_state.scale_px_per_unit)
                                st.toast(f"Merged into {next_name}", icon="🔀")
                                st.rerun()
                            else:
                                # No conflict, unique name, add normally
                                metrics = segmentation_lib.calculate_metrics(new_mask, st.session_state.scale_px_per_unit)
                                new_obj = {
                                    'id': st.session_state.next_id,
                                    'name': next_name,
                                    'mask': new_mask,
                                    'metrics': metrics
                                }
                                st.session_state.objects.append(new_obj)
                                st.session_state.next_id += 1
                                st.rerun()

    # --- Live Results Table ---
    st.markdown("### Results")
    if not df.empty:
        # Allow editing Name
        edited_df = st.data_editor(
            df, 
            key="data_editor",
            column_config={
                "ID": st.column_config.NumberColumn(disabled=True),
                "Name": st.column_config.TextColumn("Object Name"),
            },
            disabled=["ID", "Area (px)", "Perimeter (px)", "Width (px)", "Height (px)", "Circularity", "Aspect Ratio", "Area (unit²)", "Perimeter (unit)", "Width (unit)", "Height (unit)"],
            hide_index=True
        )
        
        # Sync name changes and handle merges
        if not edited_df.equals(df):
            # 1. Update names
            for index, row in edited_df.iterrows():
                obj_id = row['ID']
                new_name = row['Name']
                for obj in st.session_state.objects:
                    if obj['id'] == obj_id:
                        obj['name'] = new_name
                        break
            
            # 2. Check for duplicates and merge
            unique_objects = {} # name -> obj
            objects_to_keep = []
            
            for obj in st.session_state.objects:
                name = obj['name']
                if name in unique_objects:
                    # Merge into existing
                    target_obj = unique_objects[name]
                    merged_mask = cv2.bitwise_or(target_obj['mask'], obj['mask'])
                    
                    target_obj['mask'] = merged_mask
                    target_obj['metrics'] = segmentation_lib.calculate_metrics(merged_mask, st.session_state.scale_px_per_unit)
                    # The current 'obj' is dropped (merged)
                else:
                    unique_objects[name] = obj
                    objects_to_keep.append(obj)
            
            if len(objects_to_keep) != len(st.session_state.objects):
                st.session_state.objects = objects_to_keep
                st.toast("Merged objects with same name", icon="🔀")
            
            st.rerun()

else:
    st.info("👈 Upload an image to start.")
