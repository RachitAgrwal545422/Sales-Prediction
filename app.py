# import All the necessary python packages
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle as pk


# Initiate the app
app = Flask(__name__)

# Take the first request and render the home page


@app.route("/")
def index():
    return render_template("home.html")

# Render the submit request


@app.route('/predict', methods=['POST'])
def predict():
    item_weight = float(request.form['item_weight'])
    item_fat_content = float(request.form['item_fat_content'])
    item_visibility = float(request.form['item_visibility'])
    item_type = float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year = float(
        request.form['outlet_establishment_year'])
    outlet_size = float(request.form['outlet_size'])
    outlet_location_type = float(request.form['outlet_location_type'])
    outlet_type = float(request.form['outlet_type'])
    X = np.array([[item_weight, item_fat_content, item_visibility, item_type, item_mrp,
                 outlet_establishment_year, outlet_size, outlet_location_type, outlet_type]])

    sc = pk.load(open('models/standard.pkl', 'rb'))
    X_std = sc.transform(X)
    model = pk.load(open('models/gridSearchForest.sav', 'rb'))
    Y_pred = model.predict(X_std)
    prediction = float(Y_pred)
    # Redirect to the result page
    return redirect(url_for('result', prediction=prediction))

# Render the result page


@app.route('/result')
def result():
    # Get the prediction from the URL parameter
    prediction = request.args.get('prediction')

    # Render the result template and pass the prediction as a variable
    return render_template('result.html', prediction=prediction)


# Basic template to run the application
if __name__ == "__main__":
    app.run(debug=True, port=7384)
