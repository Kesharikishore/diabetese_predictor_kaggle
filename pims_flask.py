from flask import Flask,render_template,request
import sklearn
import pickle
import numpy as np
app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
	if request.method=='POST':
		pregnency=int(request.form['Pregnancies'])
		glucose=int(request.form['Glucose'])
		BloodPressure=float(request.form['BloodPressure'])
		SkinThickness=float(request.form['SkinThickness'])
		Insulin=float(request.form['Insulin'])
		BMI=float(request.form['BMI'])
		DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
		Age=request.form['Age']
		with open('my_model','rb') as f:
				model=pickle.load(f)
		k=list(model.predict(np.array([[pregnency,glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])))
		if k[0]==1:
			return render_template('home.html',data=['sorry you may have diabetes','red'])
		else:
			return render_template('home.html',data=['congratulations you dont have diabetes','green'])
	else :
		return render_template('home.html')
    
@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
	app.run(debug=True)