from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# In-memory users (for demo only)
users = {}
# with open('data.pkl', 'wb') as f:
#     pickle.dump(users, f)

# Load models (Make sure these pickle files are in your project folder)
with open('linear_regression_model.pkl', 'rb') as f:
    linear_model = pickle.load(f)

# with open('label_encoder.pkl', 'rb') as f:
#     label_encoder = pickle.load(f)

# # AQI bucket mapping for display
# def get_aqi_category(aqi_bucket_encoded):
#     categories = label_encoder.classes_
#     if 0 <= aqi_bucket_encoded < len(categories):
#         return categories[aqi_bucket_encoded]
#     else:
#         return "Unknown"
    
    
def get_aqi_category(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists. Please login.')
            return redirect(url_for('login'))
        users[username] = password
        with open('data.pkl', 'wb') as f:
            pickle.dump(users, f)
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) == password:
            session['username'] = username
            flash(f'Welcome, {username}!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'username' not in session:
        flash('Please login first.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Collect input features as floats
            features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2',
                        'O3', 'Benzene', 'Toluene', 'Xylene']
            input_data = []
            for feat in features:
                val = float(request.form[feat])
                input_data.append(val)
            
            # Model expects input shape (1, 12)
            X = np.array(input_data).reshape(1, -1)
            
            # Predict AQI (continuous)
            predicted_aqi = linear_model.predict(X)[0]
            predicted_aqi = round(predicted_aqi, 2)
            
            # Decide AQI bucket based on thresholds (you can customize this)
            # Or if you want to predict bucket, you should train a classifier. 
            # Here we map AQI value to categories manually for demo:
            if predicted_aqi <= 50:
                aqi_bucket = 'Good'
            elif predicted_aqi <= 100:
                aqi_bucket = 'Moderate'
            elif predicted_aqi <= 150:
                aqi_bucket = 'Poor'
            elif predicted_aqi <= 200:
                aqi_bucket = 'Unhealthy'
            elif predicted_aqi <= 300:
                aqi_bucket = 'Very Unhealthy'
            else:
                aqi_bucket = 'Hazardous'
            
            return render_template('result.html', aqi=predicted_aqi, aqi_bucket=aqi_bucket)
        
        except Exception as e:
            flash(f"Error in input or prediction: {str(e)}")
            return redirect(url_for('dashboard'))

    return render_template('dashboard.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.')
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
