from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load the ML model
model = pickle.load(open("stress_ml_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/basic_question", methods=["GET", "POST"])
def basic_question():
    if request.method == "POST":
        return redirect(url_for("inputs"))  # Redirect to inputs.html after submission
    return render_template("basic_question.html")

@app.route("/inputs", methods=["GET", "POST"])
def inputs():
    stress_level = None  # Default hidden state
    if request.method == "POST":
        try:
            acc_x = float(request.form["acc_x"])
            acc_y = float(request.form["acc_y"])
            acc_z = float(request.form["acc_z"])
            body_temp = float(request.form["body_temp"])
            resp_rate = float(request.form["resp_rate"])
            eda = float(request.form["eda"])
            emg = float(request.form["emg"])
            ecg = float(request.form["ecg"])

            # Prepare input features
            features = np.array([acc_x, acc_y, acc_z, body_temp , eda , ecg , emg , resp_rate])
            features = features.reshape(1,8)
            stress_level = model.predict(features)[0]  # Get prediction

        except Exception as e:
            print(f"Error: {e}")

    return render_template("inputs.html", stress_level=stress_level)

@app.route('/low')
def low():
    return render_template("low.html")

@app.route('/medium')
def medium():
    return render_template("medium.html")

@app.route('/high')
def high():
    return render_template("high.html")

if __name__ == "__main__":
    app.run(debug=True)
