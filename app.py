from flask import Flask, render_template, request
import pandas as pd

# Import the trained model directly from the Python script
from linear_regression_model import lm

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    avg_session = float(request.form['avg_session'])
    time_app = float(request.form['time_app'])
    time_website = float(request.form['time_website'])
    length_membership = float(request.form['length_membership'])
    
    # Prepare data
    new_customer = pd.DataFrame({
        "Avg. Session Length":[avg_session],
        "Time on App":[time_app],
        "Time on Website":[time_website],
        "Length of Membership":[length_membership]
    })
    
    # Make prediction
    prediction = lm.predict(new_customer)[0]
    
    return f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: Arial, sans-serif;
    ">
        <div style="
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            text-align: center;
            max-width: 500px;
            margin: 20px;
        ">
            <h1 style="
                color: #333;
                font-size: 24px;
                margin-bottom: 20px;
                font-weight: normal;
            ">Prediction Result</h1>
            
            <h2 style="
                color: #667eea;
                font-size: 36px;
                margin: 20px 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
                border-radius: 15px;
                border-left: 5px solid #667eea;
            ">
                $ {prediction:.2f}
            </h2>
            
            <p style="
                color: #666;
                font-size: 16px;
                margin-top: 20px;
                line-height: 1.5;
            ">
                Yearly Amount Spent
            </p>
            
            <a href="/" style="
                display: inline-block;
                margin-top: 30px;
                padding: 12px 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
                transition: transform 0.3s ease;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            " onmouseover="this.style.transform='scale(1.05)'" 
               onmouseout="this.style.transform='scale(1)'">
                Make Another Prediction
            </a>
        </div>
    </div>
    """

if __name__ == "__main__":
    app.run(debug=True, port=8000)