from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('log_reg_forest_prevention_model.pkl', 'rb'))

@app.route('/', methods=["GET"])
def index():
    return render_template('Forest.html')

@app.route('/predict', methods=["POST"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_input = [np.array(input_features)]
    prediction = model.predict_proba(final_input)
    output = '{:.2f}'.format(prediction[0][1])

    if float(output) >= 0.5:
        message = f"🔥 Your Forest is in **Danger**! Probability of fire: {output}"
    else:
        message = f"🌳 Your Forest is **Safe**. Probability of fire: {output}"

    return render_template('Forest.html', pred=message)

if __name__ == "__main__":
    app.run(debug=True)
