from flask import Flask, request, render_template, jsonify

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html", form_data={})


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["GET", "POST"])
def predict_churn():
    if request.method == "GET":
        return render_template("index.html", form_data={})

    data = CustomData(
        gender=request.form.get("gender"),
        SeniorCitizen=int(request.form.get("SeniorCitizen")),
        Partner=request.form.get("Partner"),
        Dependents=request.form.get("Dependents"),
        tenure=int(request.form.get("tenure")),
        PhoneService=request.form.get("PhoneService"),
        MultipleLines=request.form.get("MultipleLines"),
        InternetService=request.form.get("InternetService"),
        OnlineSecurity=request.form.get("OnlineSecurity"),
        OnlineBackup=request.form.get("OnlineBackup"),
        DeviceProtection=request.form.get("DeviceProtection"),
        TechSupport=request.form.get("TechSupport"),
        StreamingTV=request.form.get("StreamingTV"),
        StreamingMovies=request.form.get("StreamingMovies"),
        Contract=request.form.get("Contract"),
        PaperlessBilling=request.form.get("PaperlessBilling"),
        PaymentMethod=request.form.get("PaymentMethod"),
        MonthlyCharges=float(request.form.get("MonthlyCharges")),
        TotalCharges=float(request.form.get("TotalCharges")),
    )

    pred_df = data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    pred_class, pred_proba = predict_pipeline.predict(pred_df)

    probability_value = round(pred_proba * 100, 1)

    # Three risk bands drive both the gauge color and the verdict copy.
    if probability_value < 30:
        risk_level, risk_color, risk_label = "low", "#0EA5A0", "Low risk"
    elif probability_value < 60:
        risk_level, risk_color, risk_label = "medium", "#F59E0B", "Medium risk"
    else:
        risk_level, risk_color, risk_label = "high", "#DC2626", "High risk"

    result = "Likely to Churn" if pred_class == 1 else "Likely to Stay"

    # request.form is echoed straight back so every field the user just
    # filled in stays populated instead of resetting on submit.
    return render_template(
        "index.html",
        form_data=request.form,
        result=result,
        probability=f"{probability_value}%",
        probability_value=probability_value,
        risk_level=risk_level,
        risk_color=risk_color,
        risk_label=risk_label,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)