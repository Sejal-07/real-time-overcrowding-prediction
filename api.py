from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from datetime import datetime
import uvicorn
from models.data_processor import NagpurCrowdPredictor
import traceback

app = FastAPI(title="Nagpur Bus Crowd Prediction API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = NagpurCrowdPredictor()


class PredictionRequest(BaseModel):
    route_id: int
    stop_name: str
    current_time: str
    future_minutes: list[int] = [5, 10, 15, 30]


@app.get("/")
async def root():
    return {
        "message": "Nagpur Smart Bus Crowd Prediction API",
        "status": "active",
        "city": "Nagpur, Maharashtra",
        "endpoints": {
            "predict_crowd": "POST /predict-crowd/",
            "nagpur_stops": "GET /nagpur-stops/{route_id}",
            "health": "GET /health",
            "test_prediction": "GET /test-prediction/",
            "scan_qr": "POST /scan-qr/",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/test-prediction/")
async def test_prediction():
    """Test endpoint to verify predictions are working"""
    try:
        # Test with common Nagpur scenario
        test_route = 61
        test_stop = "Ajni Square"
        test_hour = 8
        test_day = 1

        print(
            f"🧪 Testing prediction for Route {test_route}, {test_stop}, {test_hour}:00"
        )

        # Get current prediction
        current_crowd, current_occ = predictor.predict_current_crowd(
            test_route, test_stop, test_hour, test_day
        )

        # Get future predictions
        future_results = {}
        future_tuples = {}  # For recommendation
        for minutes in [10, 20, 30]:
            crowd, occ = predictor.predict_future_crowd(
                test_route, test_stop, test_hour, test_day, minutes
            )
            future_results[str(minutes)] = {
                "crowd_level": crowd,
                "occupancy_percent": occ,
            }
            future_tuples[str(minutes)] = (crowd, occ)

        # Get recommendation - FIXED: pass current_occupancy and future tuples
        recommendation = predictor.get_recommendation(
            current_crowd, current_occ, future_tuples
        )

        return {
            "test_scenario": {
                "route": test_route,
                "stop": test_stop,
                "hour": test_hour,
                "day": test_day,
            },
            "current_prediction": {
                "crowd_level": current_crowd,
                "occupancy_percent": current_occ,
            },
            "future_predictions": future_results,
            "recommendation": recommendation,
            "status": "success",
            "message": "Prediction test completed successfully",
        }

    except Exception as e:
        error_details = traceback.format_exc()
        return {"status": "error", "error": str(e), "details": error_details}


@app.post("/scan-qr/")
async def scan_qr(file: UploadFile = File(...)):
    """Scan QR ticket from uploaded image"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "error", "message": "Failed to read image"}

        # Initialize QR Code detector
        qr_detector = cv2.QRCodeDetector()

        # Detect and decode QR code
        data, vertices, _ = qr_detector.detectAndDecode(img)

        if data:
            # Parse QR data (assuming format: route_id|stop_name|timestamp)
            parts = data.split("|")
            if len(parts) >= 3:
                try:
                    route_id = int(parts[0])
                    stop_name = parts[1]
                    scan_time = parts[2]

                    # Validate time format
                    try:
                        datetime.strptime(scan_time, "%H:%M")
                    except:
                        # If time is invalid, use current time
                        scan_time = datetime.now().strftime("%H:%M")

                    return {
                        "status": "success",
                        "route_id": route_id,
                        "stop_name": stop_name,
                        "scan_time": scan_time,
                        "qr_data": data,
                        "message": "QR code scanned successfully",
                    }
                except ValueError:
                    return {"status": "error", "message": "Invalid route ID in QR code"}
            else:
                return {"status": "error", "message": f"Invalid QR format. Got: {data}"}
        else:
            return {"status": "error", "message": "No QR code found in image"}

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"QR scanning error: {error_details}")
        return {"status": "error", "message": f"Error scanning QR: {str(e)}"}


@app.post("/predict-crowd/")
async def predict_crowd(request: PredictionRequest):
    """Get current and future crowd predictions"""
    try:
        print(f"📊 Received prediction request: {request}")

        # Parse current time
        try:
            if ":" in request.current_time:
                current_time_obj = datetime.strptime(request.current_time, "%H:%M")
                current_hour = current_time_obj.hour
                current_minute = current_time_obj.minute
            else:
                # If just hour is provided
                current_hour = int(request.current_time)
                current_minute = 0
                request.current_time = f"{current_hour:02d}:{current_minute:02d}"
        except Exception as time_error:
            print(f"⚠️ Time parsing error: {time_error}")
            # Use current system time
            now = datetime.now()
            current_hour = now.hour
            current_minute = now.minute
            request.current_time = f"{current_hour:02d}:{current_minute:02d}"

        # Get day of week (1=Monday, 7=Sunday)
        day_of_week = datetime.now().weekday() + 1

        print(
            f"🔍 Parameters: hour={current_hour}, day={day_of_week}, route={request.route_id}, stop={request.stop_name}"
        )

        # Get current prediction
        try:
            current_crowd, current_occupancy = predictor.predict_current_crowd(
                request.route_id, request.stop_name, current_hour, day_of_week
            )
            print(f"✅ Current prediction: {current_crowd}, {current_occupancy}%")
        except Exception as pred_error:
            print(f"❌ Current prediction error: {pred_error}")
            # Fallback values
            current_crowd = "MEDIUM"
            current_occupancy = 50.0

        # Get future predictions
        future_predictions = {}
        future_tuples = {}  # Store as tuples for recommendation
        for minutes in request.future_minutes:
            try:
                crowd, occupancy = predictor.predict_future_crowd(
                    request.route_id,
                    request.stop_name,
                    current_hour,
                    day_of_week,
                    minutes,
                )
                future_predictions[str(minutes)] = {
                    "crowd_level": crowd,
                    "occupancy_percent": round(occupancy, 1),
                }
                future_tuples[str(minutes)] = (crowd, occupancy)
                print(f"✅ Future prediction +{minutes}min: {crowd}, {occupancy}%")
            except Exception as future_error:
                print(f"⚠️ Future prediction error for +{minutes}min: {future_error}")
                # Default future prediction
                future_predictions[str(minutes)] = {
                    "crowd_level": "MEDIUM",
                    "occupancy_percent": 50.0,
                }
                future_tuples[str(minutes)] = ("MEDIUM", 50.0)

        # Get recommendation - FIXED: pass current_occupancy and future tuples
        try:
            recommendation = predictor.get_recommendation(
                current_crowd, current_occupancy, future_tuples
            )
        except Exception as rec_error:
            print(f"❌ Recommendation error: {rec_error}")
            recommendation = "BOARD current bus"

        # Determine comfort level
        if current_crowd == "LOW":
            comfort = "comfortable"
        elif current_crowd == "MEDIUM":
            comfort = "moderate"
        else:
            comfort = "crowded"

        response = {
            "current_prediction": {
                "crowd_level": current_crowd,
                "occupancy_percent": round(current_occupancy, 1),
                "comfort": comfort,
            },
            "future_predictions": future_predictions,
            "recommendation": recommendation,
            "route_id": request.route_id,
            "stop_name": request.stop_name,
            "current_time": request.current_time,
            "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        print(f"✅ Successfully generated predictions")
        return response

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"❌ Critical error in predict-crowd: {str(e)}")
        print(f"📋 Error details:\n{error_details}")

        # Return demo data for testing
        return {
            "current_prediction": {
                "crowd_level": "MEDIUM",
                "occupancy_percent": 50.0,
                "comfort": "moderate",
            },
            "future_predictions": {
                "10": {"crowd_level": "LOW", "occupancy_percent": 25.0},
                "20": {"crowd_level": "LOW", "occupancy_percent": 20.0},
                "30": {"crowd_level": "MEDIUM", "occupancy_percent": 45.0},
            },
            "recommendation": "WAIT 20 minutes for LOW crowd (20% occupied vs current 50%)",
            "route_id": request.route_id,
            "stop_name": request.stop_name,
            "current_time": request.current_time,
            "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Demo data (API error occurred)",
        }


@app.get("/nagpur-stops/{route_id}")
async def get_nagpur_stops(route_id: int):
    """Get Nagpur-specific bus stops for a route"""
    try:
        if hasattr(predictor, "data") and predictor.data is not None:
            # Filter stops for the given route
            route_data = predictor.data[predictor.data["route_id"] == route_id]
            if len(route_data) > 0:
                stops = route_data["stop_name"].unique().tolist()
            else:
                # Default Nagpur stops if route not found
                stops = [
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
        else:
            # Default stops if data not loaded
            stops = [
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

        return {"route_id": route_id, "stops": stops, "total_stops": len(stops)}

    except Exception as e:
        print(f"Error getting stops: {e}")
        return {
            "route_id": route_id,
            "stops": [
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
            ],
            "total_stops": 12,
            "note": "Using default Nagpur stops",
        }


@app.get("/bus-times/")
async def get_bus_times():
    """Get common bus times for Nagpur buses"""
    times = []
    for hour in range(4, 22):  # 4 AM to 10 PM
        for minute in [0, 30]:
            if hour == 21 and minute == 30:  # Skip 21:30
                continue
            times.append(f"{hour:02d}:{minute:02d}")

    return {
        "times": times,
        "peak_hours": ["07:00", "08:00", "09:00", "17:00", "18:00", "19:00"],
        "off_peak": ["11:00", "12:00", "13:00", "14:00", "15:00"],
        "note": "Nagpur bus timings (4:00 AM to 9:00 PM)",
    }


if __name__ == "__main__":
    print("🚀 Starting Nagpur Bus Crowd Prediction API...")
    print("🔗 API will be available at: http://127.0.0.1:8000")
    print("📊 Health check: http://127.0.0.1:8000/health")
    print("🧪 Test prediction: http://127.0.0.1:8000/test-prediction/")
    uvicorn.run(app, host="127.0.0.1", port=8000)
