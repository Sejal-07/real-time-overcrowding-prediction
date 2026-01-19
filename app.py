import streamlit as st
import requests
import cv2
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# Page configuration with light theme
st.set_page_config(
    page_title="Nagpur Bus Crowd Predictor",
    page_icon="🚍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS with better contrast
st.markdown(
    """
<style>
    .main-header {
        color: #FFFFFF;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1E3A8A 0%, #4F46E5 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4F46E5;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        color: #1F2937;
    }
    .recommendation-box {
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        color: #1F2937;
    }
    .wait {
        background-color: #FEF3C7;
        border: 2px solid #D97706;
        color: #92400E;
    }
    .board {
        background-color: #D1FAE5;
        border: 2px solid #059669;
        color: #065F46;
    }
    .crowd-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 0.2rem;
    }
    .low-crowd { background-color: #10B981; }
    .medium-crowd { background-color: #F59E0B; color: #000; }
    .high-crowd { background-color: #EF4444; }
    
    /* Fix text visibility */
    .stSelectbox, .stRadio, .stMultiselect, .stTextInput {
        color: #1F2937 !important;
    }
    
    /* Fix button text */
    .stButton > button {
        color: #FFFFFF !important;
        background-color: #4F46E5;
    }
    
    .stButton > button:hover {
        background-color: #4338CA;
    }
    
    /* Fix metric text */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #1F2937 !important;
    }
    
    /* Fix dataframe text */
    .dataframe {
        color: #1F2937 !important;
    }
    
    /* Section headers */
    h3 {
        color: #1E3A8A !important;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem !important;
    }
    
    /* Info boxes */
    .stAlert {
        color: #1F2937 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "scan_data" not in st.session_state:
    st.session_state.scan_data = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "selected_time" not in st.session_state:
    st.session_state.selected_time = "08:15"
if "future_minutes" not in st.session_state:
    st.session_state.future_minutes = [
        10,
        20,
        30,
        60,
        90,
    ]  # Extended options up to 90 minutes


def decode_qr_from_image(image):
    """Decode QR code from image using OpenCV"""
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Initialize QR Code detector
        qr_detector = cv2.QRCodeDetector()

        # Detect and decode QR code
        data, vertices, _ = qr_detector.detectAndDecode(img_cv)

        if data and vertices is not None:
            # Draw bounding box on the image
            pts = vertices.astype(int)
            cv2.polylines(img_cv, [pts], True, (0, 255, 0), 3)

            # Convert back to RGB for display
            img_with_box = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img_with_box), data

    except Exception as e:
        st.error(f"QR decoding error: {str(e)}")

    return image, None


def get_predictions(route_id, stop_name, current_time, future_minutes):
    """Get predictions from API"""
    prediction_request = {
        "route_id": route_id,
        "stop_name": stop_name,
        "current_time": current_time,
        "future_minutes": future_minutes,
    }

    try:
        response = requests.post(
            "http://localhost:8000/predict-crowd/", json=prediction_request, timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error("API returned an error")
            return None

    except Exception as e:
        st.error(f"API Error: {str(e)}")

        # Return demo data for testing
        current_hour = int(current_time.split(":")[0]) if ":" in current_time else 8
        demo_occupancy = (
            85.0
            if 7 <= current_hour <= 10
            else 65.0 if 17 <= current_hour <= 20 else 40.0
        )
        demo_crowd = (
            "HIGH"
            if demo_occupancy > 70
            else "MEDIUM" if demo_occupancy > 30 else "LOW"
        )

        return {
            "current_prediction": {
                "crowd_level": demo_crowd,
                "occupancy_percent": demo_occupancy,
                "comfort": (
                    "crowded"
                    if demo_crowd == "HIGH"
                    else "moderate" if demo_crowd == "MEDIUM" else "comfortable"
                ),
            },
            "future_predictions": {
                "10": {"crowd_level": "MEDIUM", "occupancy_percent": 55.0},
                "20": {"crowd_level": "LOW", "occupancy_percent": 25.0},
                "30": {"crowd_level": "LOW", "occupancy_percent": 20.0},
                "60": {"crowd_level": "LOW", "occupancy_percent": 15.0},
                "90": {"crowd_level": "LOW", "occupancy_percent": 10.0},
            },
            "recommendation": "WAIT 20 minutes for LOW crowd (25% occupied vs current 85%)",
            "route_id": route_id,
            "stop_name": stop_name,
            "current_time": current_time,
            "note": "Demo data - API offline",
        }


# Header
st.markdown(
    "<h1 class='main-header'>🚍 Nagpur Smart Bus Crowd Prediction System</h1>",
    unsafe_allow_html=True,
)

# Two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📸 QR Ticket Scanner")

    # Camera input using Streamlit's camera_input
    camera_option = st.radio(
        "Select input method:",
        ["📱 Use Device Camera", "📁 Upload QR Image"],
        horizontal=True,
    )

    if camera_option == "📱 Use Device Camera":
        camera_img = st.camera_input("Point camera at QR code", key="camera")

        if camera_img:
            image = Image.open(camera_img)
            processed_image, qr_data = decode_qr_from_image(image)

            # Display image with proper width
            st.image(processed_image, caption="Scanned QR Code")

            if qr_data:
                # Parse QR data
                parts = qr_data.split("|")
                if len(parts) >= 3:
                    st.session_state.scan_data = {
                        "route_id": int(parts[0]),
                        "stop_name": parts[1],
                        "scan_time": parts[2],
                        "status": "success",
                    }

                    st.success(f"✅ QR Code Detected!")
                    st.info(
                        f"**Route {parts[0]}** at **{parts[1]}**, Time: **{parts[2]}**"
                    )

                    # Auto-set the time from QR
                    st.session_state.selected_time = parts[2]
                else:
                    st.error("Invalid QR format. Expected: route_id|stop_name|time")
            else:
                st.warning("No QR code detected. Please try again.")

    else:  # Upload QR Image
        uploaded_file = st.file_uploader(
            "Upload QR code image", type=["png", "jpg", "jpeg"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            processed_image, qr_data = decode_qr_from_image(image)

            # Display image with proper width
            st.image(processed_image, caption="Uploaded QR Code")

            if qr_data:
                # Parse QR data
                parts = qr_data.split("|")
                if len(parts) >= 3:
                    st.session_state.scan_data = {
                        "route_id": int(parts[0]),
                        "stop_name": parts[1],
                        "scan_time": parts[2],
                        "status": "success",
                    }

                    st.success(f"✅ QR Code Detected!")
                    st.info(
                        f"**Route {parts[0]}** at **{parts[1]}**, Time: **{parts[2]}**"
                    )

                    # Auto-set the time from QR
                    st.session_state.selected_time = parts[2]
                else:
                    st.error("Invalid QR format. Expected: route_id|stop_name|time")
            else:
                st.warning("No QR code found in image")

    # Get Predictions button for QR scanning
    if st.session_state.scan_data:
        if st.button(
            "🔮 Get Predictions from QR", type="primary", use_container_width=True
        ):
            scan_data = st.session_state.scan_data

            # Use extended future minutes for better options
            future_minutes = st.session_state.future_minutes

            # Get predictions
            st.session_state.predictions = get_predictions(
                scan_data["route_id"],
                scan_data["stop_name"],
                scan_data["scan_time"],
                future_minutes,
            )
            st.rerun()

    st.markdown("---")
    st.markdown("### 📝 Manual Input (Alternative)")

    # Manual input fields
    route_id = st.selectbox("Select Route", [61, 45, 33], index=0, key="manual_route")

    # Nagpur stops
    nagpur_stops = [
        "Burdi",
        "Panchasheel Square",
        "Rahate Colony",
        "Ajni Square",
        "Chatrapati Square",
        "Ujjwal Nagar",
        "Airport",
        "Chinchbhavan",
        "Khapri",
        "Ashokwan",
        "Mohgaon",
        "Butibori",
    ]
    stop_name = st.selectbox(
        "Select Bus Stop", nagpur_stops, index=0, key="manual_stop"
    )

    # Time selection - CHANGED TO DROPDOWN
    st.markdown("#### Select Current Time")

    # Create time options exactly as in your screenshot
    time_options = ["07:30", "07:45", "08:00", "08:15", "08:30", "08:45", "09:00"]

    # Use selectbox for time selection
    selected_time = st.selectbox(
        "Click to select time",
        time_options,
        index=(
            time_options.index(st.session_state.selected_time)
            if st.session_state.selected_time in time_options
            else 3
        ),
        key="time_select",
    )

    # Update session state with selected time
    st.session_state.selected_time = selected_time

    # Show selected time (optional - for display only)
    st.text_input(
        "Selected Time",
        value=st.session_state.selected_time,
        key="time_display",
        disabled=True,
    )

    # Extended future time options (up to 90 minutes)
    st.markdown("#### ⏱️ Predict Next Buses (up to 90 minutes)")
    future_minutes = st.multiselect(
        "Select time intervals for prediction:",
        [5, 10, 15, 20, 30, 45, 60, 75, 90],
        default=[10, 20, 30, 60, 90],
        format_func=lambda x: f"{x} minutes",
        key="future_select",
    )

    # Manual predictions button
    if st.button(
        "📊 Get Manual Predictions", type="secondary", use_container_width=True
    ):
        # Get predictions
        st.session_state.predictions = get_predictions(
            route_id, stop_name, selected_time, future_minutes
        )
        st.rerun()

with col2:
    st.markdown("### 📊 Crowd Predictions Dashboard")

    if st.session_state.predictions:
        predictions = st.session_state.predictions

        # Current bus status
        st.markdown("#### 🚌 Current Bus Status")

        current = predictions["current_prediction"]
        crowd_level = current["crowd_level"]

        # Color coding
        if crowd_level == "LOW":
            crowd_color = "low-crowd"
            comfort_emoji = "✅"
        elif crowd_level == "MEDIUM":
            crowd_color = "medium-crowd"
            comfort_emoji = "⚠️"
        else:
            crowd_color = "high-crowd"
            comfort_emoji = "❌"

        # Display current status
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Crowd Level", crowd_level)
        with col_b:
            st.metric("Occupancy", f"{current['occupancy_percent']}%")
        with col_c:
            st.metric("Comfort", current["comfort"].title())

        # Route and time info
        st.markdown(
            f"""
        <div class='card'>
            <span class='crowd-indicator {crowd_color}'>{comfort_emoji} {crowd_level}</span>
            <p><b>Route {predictions['route_id']}</b> at <b>{predictions['stop_name']}</b>, {predictions['current_time']}</p>
            <p>Bus capacity: 50 seats | Current occupancy: {current['occupancy_percent']}%</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Future predictions table
        st.markdown("---")
        st.markdown("#### 🔮 Next Bus Predictions")

        future_data = []
        for minutes, data in predictions["future_predictions"].items():
            # Calculate expected arrival time
            try:
                current_time_obj = datetime.strptime(
                    predictions["current_time"], "%H:%M"
                )
                arrival_time = (
                    current_time_obj + timedelta(minutes=int(minutes))
                ).strftime("%H:%M")
            except:
                arrival_time = "N/A"

            future_data.append(
                {
                    "Wait Time": f"+{minutes} min",
                    "Arrival At": arrival_time,
                    "Crowd Level": data["crowd_level"],
                    "Occupancy %": f"{data['occupancy_percent']}%",
                    "Improvement": f"{(current['occupancy_percent'] - data['occupancy_percent']):+.1f}%",
                    "Comfort": (
                        "✅"
                        if data["crowd_level"] == "LOW"
                        else "⚠️" if data["crowd_level"] == "MEDIUM" else "❌"
                    ),
                }
            )

        df = pd.DataFrame(future_data)
        st.dataframe(df, hide_index=True)

        # Visualization
        st.markdown("---")
        st.markdown("#### 📈 Occupancy Comparison")

        # Create comparison chart
        times = ["Current"] + [
            f"+{k} min" for k in predictions["future_predictions"].keys()
        ]
        occupancy_values = [current["occupancy_percent"]] + [
            v["occupancy_percent"] for v in predictions["future_predictions"].values()
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=times,
                    y=occupancy_values,
                    marker_color=[
                        "#EF4444" if x > 70 else "#F59E0B" if x > 30 else "#10B981"
                        for x in occupancy_values
                    ],
                    text=[f"{x}%" for x in occupancy_values],
                    textposition="auto",
                    hovertemplate="<b>%{x}</b><br>Occupancy: %{y}%<extra></extra>",
                    textfont=dict(
                        color="#FFFFFF",  # White text for better visibility
                        size=14,
                        family="Arial, sans-serif",
                        weight="bold",
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Bus Occupancy Comparison",
            xaxis_title="Time",
            yaxis_title="Occupancy %",
            yaxis_range=[0, 100],
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#1F2937", size=12),
            showlegend=False,
        )

        # Add threshold lines with improved text visibility
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            annotation_text="Low Crowd",
            annotation_position="right",
            annotation_font=dict(color="green", size=12, weight="bold"),
        )
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            annotation_text="High Crowd",
            annotation_position="right",
            annotation_font=dict(color="red", size=12, weight="bold"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recommendation
        st.markdown("---")
        st.markdown("#### 🎯 Smart Recommendation")

        recommendation = predictions.get("recommendation", "BOARD current bus")

        if "WAIT" in recommendation.upper() or "CONSIDER" in recommendation.upper():
            st.markdown(
                f"""
            <div class='recommendation-box wait'>
                ⏳ {recommendation}
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Show best alternative
            best_option = None
            best_improvement = 0
            for minutes, data in predictions["future_predictions"].items():
                improvement = current["occupancy_percent"] - data["occupancy_percent"]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_option = (minutes, data)

            if best_option and best_improvement > 0:
                minutes, data = best_option
                current_time_obj = datetime.strptime(
                    predictions["current_time"], "%H:%M"
                )
                arrival_time = (
                    current_time_obj + timedelta(minutes=int(minutes))
                ).strftime("%H:%M")

                st.info(
                    f"""
                **Best Alternative:** Bus arriving at **{arrival_time}** ({minutes} minutes from now)
                - **Crowd Level:** {data['crowd_level']} (vs current {crowd_level})
                - **Occupancy:** {data['occupancy_percent']}% (vs current {current['occupancy_percent']}%)
                - **Improvement:** {best_improvement:.1f}% less crowded
                """
                )
        else:
            st.markdown(
                f"""
            <div class='recommendation-box board'>
                🚌 {recommendation}
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.info(
                f"""
            **Why board now?**
            - Current occupancy ({current['occupancy_percent']}%) is already reasonable
            - Future buses offer minimal improvement
            - You'll reach your destination sooner
            """
            )

        # Nagpur-specific context
        st.markdown("---")
        st.markdown("#### 🏙️ Nagpur City Context")

        context_info = ""
        stop_name = predictions["stop_name"]

        try:
            current_hour = int(predictions["current_time"].split(":")[0])
        except:
            current_hour = 8  # Default

        if stop_name in ["Airport", "MIHAN", "Chinchbhavan"]:
            context_info = "**Industrial/Airport Zone:** Morning (7-9 AM) and evening (5-7 PM) peaks due to shift workers."
        elif stop_name in ["Ajni Square", "Chatrapati Square", "Panchasheel Square"]:
            context_info = "**CBD Area:** Peak hours 8-10 AM (office arrivals) and 6-8 PM (departures)."
        elif stop_name in ["Butibori", "Mohgaon", "Khapri"]:
            context_info = (
                "**Industrial Area:** Busiest during shift changes (6 AM, 2 PM, 10 PM)."
            )
        elif stop_name in ["Burdi", "Rahate Colony", "Ujjwal Nagar"]:
            context_info = "**Residential Area:** Higher crowd during commute hours (7-9 AM, 6-8 PM)."
        else:
            context_info = "**General Area:** Moderate traffic throughout the day."

        # Add time-specific advice
        if 7 <= current_hour <= 10:
            context_info += (
                " **Morning peak** - consider waiting 20-30 minutes for reduced crowd."
            )
        elif 17 <= current_hour <= 20:
            context_info += " **Evening peak** - buses fill quickly, but later buses (after 8 PM) are less crowded."
        elif 11 <= current_hour <= 16:
            context_info += " **Off-peak hours** - generally comfortable throughout."
        elif current_hour < 7 or current_hour > 20:
            context_info += " **Early morning/late night** - minimal crowd expected."

        st.markdown(
            f"""
        <div class='card'>
            <b style='color: #1E3A8A;'>📍 {stop_name} Insights:</b><br>
            <span style='color: #4B5563;'>{context_info}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Reset button
        col_reset1, col_reset2, col_reset3 = st.columns([1, 2, 1])
        with col_reset2:
            if st.button("🔄 Start New Prediction", use_container_width=True):
                st.session_state.predictions = None
                st.session_state.scan_data = None
                st.rerun()

    else:
        st.info(
            "👈 Start camera and scan a QR ticket, or use manual input to get predictions"
        )

        # Quick info card
        st.markdown(
            f"""
        <div class='card'>
            <h4 style='color: #1E3A8A;'>🚍 How to use:</h4>
            <ol style='color: #4B5563;'>
                <li><b>Option 1 (QR Scan):</b> Use device camera or upload QR image</li>
                <li><b>Option 2 (Manual):</b> Select route, stop, and time below</li>
                <li>Select future times to check (up to 90 minutes ahead)</li>
                <li>Click <b>Get Predictions</b> to see crowd levels</li>
            </ol>
            <p style='color: #4B5563;'><b>QR Format:</b> route_id|stop_name|time (e.g., 61|Ajni Square|07:30)</p>
            <p style='color: #4B5563;'><b>Nagpur Routes:</b> 61 (CBD→Airport→MIHAN), 45, 33</p>
            <p style='color: #4B5563;'><b>Tip:</b> System recommends waiting if future buses are significantly less crowded</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #6B7280;'>
    <p>🚍 <b style='color: #1E3A8A;'>Nagpur Smart City Initiative</b> - Real-time Public Transport Optimization System</p>
    <p>Predicts crowd levels for Nagpur city buses and helps passengers make informed BOARD/WAIT decisions</p>
</div>
""",
    unsafe_allow_html=True,
)
