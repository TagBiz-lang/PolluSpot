from flask import Flask, render_template, request, redirect, url_for, session, flash
import time
import os
import random
import csv
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Flask application
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Data and model
data = pd.read_csv("data.csv")
model = DecisionTreeClassifier()

# Constants
INTERACTIONS_FILE = "interactions.csv"
IMAGE_FOLDER = "static/images"

def get_image_filename(name):
    """The name of the image file is found with the extension."""
    possible_extensions = ['.jpg', '.jpeg', '.jfif', '.png']
    for ext in possible_extensions:
        filename = f"{name}{ext}"
        if os.path.exists(os.path.join(IMAGE_FOLDER, filename)):
            return filename
    raise FileNotFoundError(f"Image not found: {name}")

def get_training_images():
    """It randomly selects 5 training images from the range 1-25 and 51-75."""
    valid_images = data[(data["Name"].astype(str).isin([str(i) for i in range(1, 26)]) |
                        data["Name"].astype(str).isin([str(i) for i in range(51, 76)]))]
    images = valid_images.sample(6).to_dict(orient="records")
    for image in images:
        image["Name"] = get_image_filename(str(image["Name"]))
    return images

def get_test_image():
    """Randomly selects a test image from range 26-50."""
    valid_images = data[data["Name"].astype(str).isin([str(i) for i in range(26, 51)])]
    test_image = valid_images.sample(1).iloc[0]
    return str(test_image["Name"]), test_image[["Smoke", "Garbage", "DirtyWater", "Dull", "Grass", "Tree"]].values





def log_training(image_names, features, user_labels, test_image_name, test_features, prediction):
    """Logs training and prediction processes as a single line."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    elapsed_time = round(time.time() - session.get("start_time", time.time()), 2)  # Calculate time spent

    with open(INTERACTIONS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        training_data = ';'.join(
            [f"{name}:{','.join(map(str, feat))}:{user_labels[name]}" for name, feat in zip(image_names, features)]
        )
        test_data = f"{test_image_name}:{','.join(map(str, test_features))}"
        writer.writerow([timestamp, session.get("fname"), session.get("linitial"), session.get("grade"),
                         training_data, test_data, elapsed_time, prediction])



# From here on, the necessary parts for creating the web page



@app.route("/", methods=["GET", "POST"])
def index():
    """Home Page."""
    if request.method == "POST":
        session["fname"] = request.form["fname"]
        session["linitial"] = request.form["linitial"]
        session["grade"] = request.form["grade"]
        return redirect(url_for("training"))
    return render_template("index.html")




@app.route("/training", methods=["GET", "POST"])
def training():
    images = get_training_images()  # We choose 6 images for training
    selections = {}

    if request.method == "POST":
        # We get the image names and labels selected by the user
        image_names = request.form.getlist("image_names")
        selections = {name: int(request.form.get(f"label_{name}")) for name in image_names}

        # We get the features of each image from the form
        features = [
            [
                int(value) if value else 0
                for value in request.form.get(f"features_{name}", "").split(',')
            ]
            for name in image_names
        ]

        # We train the model based on user labels
        labels = [selections[name] for name in image_names]
        model.fit(features, labels)

        # Test image and prediction process
        test_image_name, test_features = get_test_image()
        prediction = model.predict([test_features])[0]
        prediction_label = "Polluted" if prediction == 1 else "Unpolluted"

        # We log training data
        log_training(image_names, features, selections, test_image_name, test_features, prediction_label)

        #We want to activate the End button after training and prediction has been completed
        session["trained_once"] = True

        # We preserve existing features and extensions when re-rendering the page
        return render_template(
            "training.html",
            images=[{
                "Name": f"{name}.{get_image_filename(name).split('.')[-1]}",  # Combining name and extension
                "Smoke": feat[0],
                "Garbage": feat[1],
                "DirtyWater": feat[2],
                "Dull": feat[3],
                "Grass": feat[4],
                "Tree": feat[5]
            } for name, feat in zip(image_names, features)],
            selections=selections,
            prediction=prediction_label,
            test_image=get_image_filename(test_image_name),
            trained_once=session.get("trained_once", False)
        )

    return render_template("training.html", images=images, selections=selections, trained_once=session.get("trained_once",
                                                                                                            False))





@app.route("/retry", methods=["POST"])
def retry():
    # We get the image names and properties from the form
    image_names = request.form.getlist("image_names")
    features = [
        [
            int(value) if value else 0
            for value in request.form.get(f"features_{name}", "").split(',')
        ][:6]  # Ensure exactly six features
        for name in image_names
    ]

    selections = {
        name: int(request.form.get(f"label_{name}")) for name in image_names
    }

    # New test image and prediction
    test_image_name, test_features = get_test_image()
    prediction = model.predict([test_features])[0]
    prediction_label = "Polluted" if prediction == 1 else "Unpolluted"

    # Logging for retry
    log_training(image_names, features, selections, test_image_name, test_features, prediction_label)

    # We re-render the page keeping the same images and features
    return render_template(
        "training.html",
        images=[{
            "Name": f"{name}.{get_image_filename(name).split('.')[-1]}",  # Combining name and extension
            "Smoke": feat[0],
            "Garbage": feat[1],
            "DirtyWater": feat[2],
            "Dull": feat[3],
            "Grass": feat[4],
            "Tree": feat[5]
        } for name, feat in zip(image_names, features)],
        selections=selections,
        prediction=prediction_label,
        test_image=get_image_filename(test_image_name),
    )




if __name__ == "__main__":
    if not os.path.exists(INTERACTIONS_FILE):
        with open(INTERACTIONS_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "FirstName", "LastInitial", "Grade",
                             "TrainingData", "TestData", "Prediction"])
    app.run(debug=True)


