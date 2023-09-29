import requests
from flask import render_template

import app


@app.route('/display_predictions', methods=['GET'])
def display_predictions():
    # Make an internal request to the /get_predictions endpoint
    response = requests.get('http://localhost:5000/get_predictions')

    if response.status_code == 200:
        data = response.json()
        predictions = data['predictions']

        # Now you have access to the predictions and can use them in your template
        return render_template('display_predictions.html', predictions=predictions)

    return 'Failed to fetch predictions from the model.'

if __name__ == '__main__':
    app.run(debug=True)
