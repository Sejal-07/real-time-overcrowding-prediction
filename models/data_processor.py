import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

warnings.filterwarnings("ignore")


class NagpurCrowdPredictor:
    def __init__(self, bus_capacity=50):
        self.bus_capacity = bus_capacity
        self.data = None
        self.model = None
        self.model_columns = None
        self.load_nagpur_data()
        self.train_model()

    def load_nagpur_data(self):
        """Load Nagpur-specific bus data"""
        try:
            os.makedirs("data", exist_ok=True)
            self.data = pd.read_csv("data/nagpur_bus_data.csv")
            print(f"✅ Loaded Nagpur bus data: {len(self.data)} records")
        except Exception as e:
            print(f"⚠️ Could not load data file: {e}")
            print("⚠️ Creating synthetic Nagpur data...")
            self.create_synthetic_nagpur_data()

    def create_synthetic_nagpur_data(self):
        """Generate realistic Nagpur bus data with varied crowd levels"""
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

        data = []

        for day in [1, 2, 3, 4, 5, 6, 7]:  # Days 1-7
            for hour in range(5, 22):  # 5 AM to 10 PM
                for route in [61, 45, 33]:
                    occupancy = 0  # Reset occupancy for each route-hour combination

                    for stop_idx, stop in enumerate(nagpur_stops):
                        # Realistic patterns for Nagpur
                        # Morning peak (7-10 AM) - HIGH boarding at CBD stops
                        if 7 <= hour <= 10:
                            if stop in ["Burdi", "Rahate Colony", "Ujjwal Nagar"]:
                                # Residential areas - high boarding in morning
                                boarding = np.random.randint(15, 25)
                                alighting = np.random.randint(3, 8)
                            elif stop in [
                                "Ajni Square",
                                "Chatrapati Square",
                                "Panchasheel Square",
                            ]:
                                # CBD areas - mixed
                                boarding = np.random.randint(8, 15)
                                alighting = np.random.randint(10, 20)
                            elif stop == "Airport":
                                # Airport - some arrivals in morning
                                boarding = np.random.randint(5, 12)
                                alighting = np.random.randint(8, 15)
                            else:
                                boarding = np.random.randint(8, 18)
                                alighting = np.random.randint(5, 12)

                        # Evening peak (5-8 PM) - HIGH alighting at residential stops
                        elif 17 <= hour <= 20:
                            if stop in ["Burdi", "Rahate Colony", "Ujjwal Nagar"]:
                                # Residential areas - high alighting in evening
                                boarding = np.random.randint(3, 8)
                                alighting = np.random.randint(15, 25)
                            elif stop in [
                                "Ajni Square",
                                "Chatrapati Square",
                                "Panchasheel Square",
                            ]:
                                # CBD areas - high boarding in evening
                                boarding = np.random.randint(10, 20)
                                alighting = np.random.randint(8, 15)
                            elif stop == "Airport":
                                # Airport - evening flights
                                boarding = np.random.randint(8, 15)
                                alighting = np.random.randint(5, 12)
                            else:
                                boarding = np.random.randint(5, 12)
                                alighting = np.random.randint(8, 18)

                        # Off-peak hours
                        else:
                            boarding = np.random.randint(3, 10)
                            alighting = np.random.randint(3, 10)

                        # Calculate occupancy
                        occupancy = occupancy + boarding - alighting
                        occupancy = max(0, min(self.bus_capacity, occupancy))

                        # Determine crowd level
                        occupancy_pct = (occupancy / self.bus_capacity) * 100
                        if occupancy_pct < 30:
                            crowd = "LOW"
                        elif occupancy_pct < 70:
                            crowd = "MEDIUM"
                        else:
                            crowd = "HIGH"

                        data.append(
                            {
                                "day": day,
                                "route_id": route,
                                "stop_name": stop,
                                "hour": hour,
                                "boarding_count": boarding,
                                "alighting_count": alighting,
                                "occupancy": occupancy,
                                "crowd_level": crowd,
                            }
                        )

        self.data = pd.DataFrame(data)
        os.makedirs("data", exist_ok=True)
        self.data.to_csv("data/nagpur_bus_data.csv", index=False)
        print("✅ Generated synthetic Nagpur bus data with realistic patterns")
        print(
            f"📊 Crowd distribution: {self.data['crowd_level'].value_counts().to_dict()}"
        )

    def train_model(self):
        """Train ML model for crowd prediction"""
        if self.data is None:
            self.load_nagpur_data()

        print("🔧 Training ML model...")

        # Prepare features
        X = self.data[
            ["day", "route_id", "hour", "boarding_count", "alighting_count"]
        ].copy()

        # One-hot encode stop names (Nagpur-specific)
        stops_encoded = pd.get_dummies(self.data["stop_name"], prefix="stop")
        X = pd.concat([X, stops_encoded], axis=1)

        # Store the feature columns for prediction
        self.model_columns = X.columns.tolist()

        # Target variable
        y = self.data["crowd_level"]

        # Check class distribution
        print(f"📊 Training data class distribution:")
        print(y.value_counts())

        # Train Random Forest with balanced class weights
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
        )
        self.model.fit(X, y)

        # Check feature importance
        feature_importance = pd.DataFrame(
            {
                "feature": self.model_columns,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print("🔝 Top 10 important features:")
        print(feature_importance.head(10))

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Save model
        joblib.dump(self.model, "models/crowd_model.joblib")
        print("✅ ML model trained and saved")
        print(f"📊 Model features: {len(self.model_columns)} columns")

    def predict_current_crowd(self, route_id, stop_name, current_hour, day_of_week):
        """Predict crowd level for current bus"""
        try:
            if self.model is None:
                print("⚠️ Model not loaded, using fallback prediction")
                return self._get_fallback_prediction(current_hour, stop_name)

            # Create feature vector with realistic values
            # Estimate boarding/alighting based on time and stop
            if 7 <= current_hour <= 10:  # Morning peak
                if stop_name in ["Burdi", "Rahate Colony", "Ujjwal Nagar"]:
                    boarding = np.random.randint(15, 20)
                    alighting = np.random.randint(3, 6)
                else:
                    boarding = np.random.randint(8, 15)
                    alighting = np.random.randint(5, 10)
            elif 17 <= current_hour <= 20:  # Evening peak
                if stop_name in ["Burdi", "Rahate Colony", "Ujjwal Nagar"]:
                    boarding = np.random.randint(3, 6)
                    alighting = np.random.randint(15, 20)
                else:
                    boarding = np.random.randint(10, 18)
                    alighting = np.random.randint(8, 15)
            else:  # Off-peak
                boarding = np.random.randint(4, 8)
                alighting = np.random.randint(4, 8)

            features_dict = {
                "day": [day_of_week],
                "route_id": [route_id],
                "hour": [current_hour],
                "boarding_count": [boarding],
                "alighting_count": [alighting],
            }

            # Add stop encoding for all stops
            all_stops = [
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

            for stop in all_stops:
                features_dict[f"stop_{stop}"] = [1 if stop == stop_name else 0]

            features = pd.DataFrame(features_dict)

            # Ensure all required columns are present
            missing_cols = set(self.model_columns) - set(features.columns)
            for col in missing_cols:
                features[col] = 0

            # Reorder columns to match training
            features = features[self.model_columns]

            # Predict
            prediction = self.model.predict(features)[0]

            # Calculate realistic occupancy
            occupancy_pct = self._calculate_realistic_occupancy(
                current_hour, route_id, stop_name, day_of_week
            )

            # Adjust prediction based on calculated occupancy if needed
            if occupancy_pct < 30 and prediction != "LOW":
                prediction = "LOW"
            elif occupancy_pct >= 70 and prediction != "HIGH":
                prediction = "HIGH"
            elif 30 <= occupancy_pct < 70 and prediction != "MEDIUM":
                prediction = "MEDIUM"

            print(
                f"🔍 Prediction for {stop_name} at {current_hour}:00: {prediction} ({occupancy_pct:.1f}%)"
            )

            return prediction, min(100, max(0, occupancy_pct))

        except Exception as e:
            print(f"❌ Error in predict_current_crowd: {e}")
            # Return fallback prediction
            return self._get_fallback_prediction(current_hour, stop_name)

    def _get_fallback_prediction(self, current_hour, stop_name):
        """Get fallback prediction based on time of day and stop"""
        # Base prediction
        if 7 <= current_hour <= 10:
            if stop_name in ["Burdi", "Rahate Colony", "Ujjwal Nagar"]:
                return "HIGH", 85.0
            else:
                return "MEDIUM", 65.0
        elif 17 <= current_hour <= 20:
            if stop_name in ["Burdi", "Rahate Colony", "Ujjwal Nagar"]:
                return "MEDIUM", 60.0
            else:
                return "HIGH", 75.0
        else:
            if stop_name == "Airport":
                return "MEDIUM", 55.0
            else:
                return "LOW", 35.0

    def _calculate_realistic_occupancy(self, hour, route_id, stop_name, day_of_week):
        """Calculate realistic occupancy percentage based on multiple factors"""
        try:
            # Base occupancy based on time of day
            if 7 <= hour <= 10:  # Morning peak
                base = np.random.uniform(60, 90)
            elif 17 <= hour <= 20:  # Evening peak
                base = np.random.uniform(55, 85)
            elif 11 <= hour <= 16:  # Mid-day
                base = np.random.uniform(30, 60)
            else:  # Early morning / late night
                base = np.random.uniform(15, 40)

            # Adjust based on route (Route 61 is busiest)
            if route_id == 61:
                base = min(95, base * 1.25)
            elif route_id == 45:
                base = base * 0.9
            else:
                base = base * 0.8

            # Adjust based on stop type
            residential_stops = ["Burdi", "Rahate Colony", "Ujjwal Nagar"]
            cbd_stops = ["Ajni Square", "Chatrapati Square", "Panchasheel Square"]
            industrial_stops = ["Butibori", "Mohgaon", "Khapri"]

            if stop_name in residential_stops:
                if 7 <= hour <= 10:
                    base = min(95, base * 1.3)  # High in morning
                elif 17 <= hour <= 20:
                    base = base * 0.8  # Lower in evening (people getting off)
            elif stop_name in cbd_stops:
                if 8 <= hour <= 10:
                    base = min(95, base * 1.2)  # High in morning arrival
                elif 17 <= hour <= 19:
                    base = min(95, base * 1.3)  # High in evening departure
            elif stop_name == "Airport":
                base = base * 1.1  # Always busy

            # Adjust based on day of week (weekends are less busy)
            if day_of_week >= 6:  # Saturday (6) or Sunday (7)
                base = base * 0.7

            # Add some randomness
            base = base * np.random.uniform(0.9, 1.1)

            return min(100, max(0, base))

        except:
            return 50.0  # Default

    def predict_future_crowd(
        self, route_id, stop_name, current_hour, day_of_week, minutes_ahead
    ):
        """Predict crowd for future bus"""
        try:
            # Calculate future hour
            future_hour = current_hour + (minutes_ahead // 60)
            future_minute = minutes_ahead % 60

            if future_hour >= 24:
                future_hour -= 24
                # Next day - adjust day of week
                day_of_week = day_of_week + 1 if day_of_week < 7 else 1

            print(
                f"🔮 Future prediction: +{minutes_ahead}min → {future_hour}:{future_minute:02d}"
            )

            # Get prediction for future time
            future_crowd, future_occupancy = self.predict_current_crowd(
                route_id, stop_name, future_hour, day_of_week
            )

            # Adjust for the fact that later buses tend to be less crowded
            # (people spread out over time)
            if minutes_ahead >= 30:
                reduction_factor = 1.0 - (
                    minutes_ahead / 180
                )  # Up to 66% reduction for 3-hour wait
                future_occupancy = future_occupancy * max(0.34, reduction_factor)

                # Recalculate crowd level based on adjusted occupancy
                if future_occupancy < 30:
                    future_crowd = "LOW"
                elif future_occupancy < 70:
                    future_crowd = "MEDIUM"
                else:
                    future_crowd = "HIGH"

            print(
                f"✅ Future +{minutes_ahead}min: {future_crowd} ({future_occupancy:.1f}%)"
            )

            return future_crowd, min(100, max(0, future_occupancy))

        except Exception as e:
            print(f"❌ Error in predict_future_crowd: {e}")
            # Return reasonable defaults
            if minutes_ahead <= 15:
                return "MEDIUM", 55.0
            elif minutes_ahead <= 30:
                return "MEDIUM", 45.0
            elif minutes_ahead <= 60:
                return "LOW", 35.0
            else:
                return "LOW", 25.0

    def get_recommendation(self, current_crowd, current_occupancy, future_predictions):
        """Generate BOARD vs WAIT recommendation based on occupancy percentage"""
        try:
            print(
                f"🤔 Generating recommendation: Current {current_crowd} ({current_occupancy}%)"
            )

            # If current occupancy is already LOW, always recommend BOARD
            if current_crowd == "LOW" and current_occupancy < 40:
                return f"BOARD current bus ({current_occupancy:.0f}% occupied - Already comfortable)"

            # Find future buses with significantly lower occupancy
            better_options = []
            for minutes, (crowd, occupancy) in future_predictions.items():
                improvement = current_occupancy - occupancy
                if improvement > 15:  # At least 15% improvement
                    better_options.append((minutes, crowd, occupancy, improvement))

            # Sort by highest improvement (most comfortable)
            better_options.sort(key=lambda x: x[3], reverse=True)

            if better_options:
                best_minutes, best_crowd, best_occupancy, best_improvement = (
                    better_options[0]
                )

                # Format recommendation
                if best_improvement >= 30:
                    strength = "Highly recommended"
                elif best_improvement >= 20:
                    strength = "Recommended"
                else:
                    strength = "Consider"

                return f"{strength}: WAIT {best_minutes} minutes for {best_crowd.lower()} crowd ({best_occupancy:.0f}% occupied vs current {current_occupancy:.0f}%)"
            else:
                # Check if any future bus is at least better
                for minutes, (crowd, occupancy) in future_predictions.items():
                    if occupancy < current_occupancy:
                        return f"BOARD current bus (Future buses offer minimal improvement)"

                return "BOARD current bus (No better options available)"

        except Exception as e:
            print(f"❌ Error in get_recommendation: {e}")
            return "BOARD current bus"
