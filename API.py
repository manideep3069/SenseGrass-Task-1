# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd

# # Load the trained model
# model = joblib.load(
#     "G:\My Drive\Colab Notebooks\SenseGrass\Assignment_Wine_data\model.pkl")

# # Load the wine dataset
# df = pd.read_csv(
#     "G:\My Drive\Colab Notebooks\SenseGrass\Assignment_Wine_data\OSX_DS_assignment.csv")

# # Initialize the Flask application
# app = Flask(__name__)

# # Define the endpoint for prediction


# @app.route("/", methods=["GET","POST"])
# def predict():
#     # Parse the request data
#     data = request.get_json()

#     # Get the review from the request data
#     review_title = data["review_title"]
#     review_description = data["review_description"]
#     designation = data["designation"]
#     points = data["points"]
#     price = data["price"]
#     province = data["province"]
#     region_1 = data["region_1"]
#     region_2 = data["region_2"]
#     winery = data["winery"]
#     variety = data["variety"]
#     country = data["country"]

#     # Create a DataFrame from the request data
#     new_data = pd.DataFrame({
#         "review_title": [review_title],
#         "review_description": [review_description],
#         "designation": [designation],
#         "points": [points],
#         "price": [price],
#         "province": [province],
#         "region_1": [region_1],
#         "region_2": [region_2],
#         "winery": [winery],
#         "variety": [variety],
#         "country": [country]
#     })

#     # Preprocess the data
#     new_data["review_description"] = new_data["review_description"].str.lower()
#     new_data["designation"] = new_data["designation"].fillna("")
#     new_data["region_1"] = new_data["region_1"].fillna("")
#     new_data["region_2"] = new_data["region_2"].fillna("")
#     new_data["winery"] = new_data["winery"].fillna("")
#     new_data["variety"] = new_data["variety"].fillna("")
#     new_data["text"] = new_data["review_title"] + " " + new_data["review_description"] + " " + \
#         new_data["designation"] + " " + new_data["region_1"] + " " + new_data["region_2"] + " " + \
#         new_data["winery"] + " " + new_data["variety"] + \
#         " " + new_data["country"]
#     new_data = new_data.drop(["review_title", "review_description", "designation", "region_1", "region_2",
#                               "winery", "variety", "country"], axis=1)

#     # Make a prediction
#     prediction = model.predict(new_data)[0]

#     # Return the prediction as JSON
#     return jsonify({"prediction": prediction})


# # Run the application
# if __name__ == "__main__":
#     app.run(port=12345, debug=True)

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

# Load the trained model
model = joblib.load(
    "S:\SenseGrass\SenseGrass\model.pkl")

# Load the wine data
data = pd.read_csv(
    "S:\SenseGrass\SenseGrass\OSX_DS_assignment.csv")

# Initialize the Flask application
app = Flask(__name__, template_folder='S:\SenseGrass\SenseGrass\emplates')

# Define the endpoint for the web page


@app.route("/")
def index():
    return render_template("S:\SenseGrass\SenseGrass\emplates\index.html")

# Define the endpoint for prediction


@app.route("/predict", methods=["POST"])
def predict():
    # Parse the request data
    data = request.form.to_dict()

    # Get the input values from the request data
    user_name = data["user_name"]
    country = data["country"]
    review_title = data["review_title"]
    review_description = data["review_description"]
    designation = data["designation"]
    points = float(data["points"])
    price = float(data["price"])
    province = data["province"]
    region_1 = data["region_1"]
    region_2 = data["region_2"]
    winery = data["winery"]
    variety = data["variety"]

    # Create a dictionary with the input values
    input_data = {
        "user_name": user_name,
        "country": country,
        "review_title": review_title,
        "review_description": review_description,
        "designation": designation,
        "points": points,
        "price": price,
        "province": province,
        "region_1": region_1,
        "region_2": region_2,
        "winery": winery,
        "variety": variety
    }

    # Create a DataFrame with the input data
    input_df = pd.DataFrame([input_data])

    # Make a prediction using the model
    prediction = model.predict(input_df)[0]

    # Return the prediction as JSON
    return jsonify({"prediction": prediction})


# Run the application
if __name__ == "__main__":
    app.run(debug=True)
