from flask import Flask,request

app = Flask(__name__)


@app.route("/")
def health():
    return "server is up"

@app.route("/submit",methods=['GET','POST'])
def get_submited_image():
    if request.method == 'POST':
        if 'my_image' not in request.files:
            return 'no file part', 400
    # Logic for handling POST request (e.g., process form submission)
        file = request.files['my_image']
        if file.filename == '':
            return 'No slected file', 400
        if file:
            return f'File {file.filename} uploaded and recieved'
        return 'Form submitted successfully!'
    else:
    # Logic for handling GET request (e.g., display the form)
        return 'Please log in with a form.'

if __name__ == '__main__':
    app.run()
    