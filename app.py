import pickle 
from flask import Flask, request, app, jsonify, url_for, render_template 
import numpy as np 
import pandas as pd 

app = Flask(__name__)

# load the model 
regmodel = pickle.load(open("regmodel.pkl" ,'rb'))
scalar = pickle.load(open("scaling.pkl" ,'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output)
    
    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    # 1) Tentukan urutan fitur yang benar
    if hasattr(scalar, "feature_names_in_"):
        feature_order = list(scalar.feature_names_in_)  # paling aman: ambil dari scaler tersimpan
    else:
        # fallback: urutan standar Boston Housing (pastikan cocok dengan training-mu!)
        feature_order = [
            "CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS",
            "RAD","TAX","PTRATIO","B","LSTAT"
        ]

    # 2) Ambil nilai dari form BERDASARKAN NAMA FIELD yang sama dengan feature_order
    form = request.form
    try:
        row = [float(form[f]) for f in feature_order]
    except KeyError as e:
        return f"Missing form field: {e.args[0]}", 400
    except ValueError as e:
        return f"Invalid numeric value: {e}", 400

    # 3) Bungkus sebagai DataFrame dengan kolom yang tepat
    X_df = pd.DataFrame([row], columns=feature_order)

    # 4) Transform & prediksi
    final_input = scalar.transform(X_df)
    print(final_input)
    output = regmodel.predict(final_input)[0]

    return render_template("home.html", prediction_text=f"The House Price is {output}")

if __name__ == "__main__":
    app.run(debug =True )