from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # 獲取表單數據
        username = request.form.get('username')
        email = request.form.get('email')
        try:
            print(request.json)
        except:
            pass
        # 在此處處理表單數據
        return f"Received: Username={username}, Email={email}"

    return render_template('form.html')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)