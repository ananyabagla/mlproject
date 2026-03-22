from flask import Flask, render_template, request
import logging
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    logging.info("Navigated to Home Page")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    logging.info("!!! POST REQUEST RECEIVED AT /predict !!!")
    try:
        # Log all incoming form data for debugging
        logging.info(f"Form Data: {request.form.to_dict()}")

        data = CustomData(
            Hours_Studied=float(request.form.get('Hours_Studied')),
            Attendance=float(request.form.get('Attendance')),
            Parental_Involvement=request.form.get('Parental_Involvement'),
            Access_to_Resources=request.form.get('Access_to_Resources'),
            Extracurricular_Activities=request.form.get('Extracurricular_Activities'),
            Sleep_Hours=float(request.form.get('Sleep_Hours')),
            Previous_Scores=float(request.form.get('Previous_Scores')),
            Motivation_Level=request.form.get('Motivation_Level'),
            Internet_Access=request.form.get('Internet_Access'),
            Tutoring_Sessions=float(request.form.get('Tutoring_Sessions')),
            Family_Income=request.form.get('Family_Income'),
            Teacher_Quality=request.form.get('Teacher_Quality'),
            School_Type=request.form.get('School_Type'),
            Peer_Influence=request.form.get('Peer_Influence'),
            Physical_Activity=float(request.form.get('Physical_Activity')),
            Learning_Disabilities=request.form.get('Learning_Disabilities'),
            Parental_Education_Level=request.form.get('Parental_Education_Level'),
            Distance_from_Home=request.form.get('Distance_from_Home'),
            Gender=request.form.get('Gender')
        )

        pred_df = data.get_data_as_dataframe()
        logging.info("DataFrame ready. Initializing PredictPipeline...")

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        logging.info(f"Prediction result obtained: {results}")

        msg = f"Predicted Score: {results:.2f}"
        return render_template('index.html', prediction_text=msg)

    except Exception as e:
        logging.error(f"DETAILED ERROR IN app.py: {str(e)}")
        # This will show the error directly on your web page
        return render_template('index.html', prediction_text=f"ERROR OCCURRED: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)