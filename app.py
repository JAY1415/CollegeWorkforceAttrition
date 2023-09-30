import numpy as np
import static as static
from flask import Flask, request, jsonify, render_template
import pickle

from College_Workforce_Attrition import df, df1
from flask import send_from_directory

app = Flask(__name__,template_folder='templates',static_url_path='/static')
app.config['STATIC_FOLDER'] = 'static'
#model = pickle.load(open('FlaskProject/Model/MIModel.pkl', 'rb'))

@app.route('/')
def home():
    # Access data from the GET request
      # Use 'Default Data' if 'data' is not provided

    return render_template('index.html')

@app.route('/login')
def login():
    # Access data from the GET request
      # Use 'Default Data' if 'data' is not provided

    return render_template('Login/login.html')

@app.route('/departments')
def departments():
    return render_template('dashboard/department.html')


@app.route('/employee',)
def employee():
    try:
        # Verify that df1 is a valid DataFrame
        if 'df1' not in globals() or df1 is None:
            return "Error: df1 is not defined or is empty.", 500  # Return an error message and status code 500

        selected_col = ['Department', 'Qualification', 'Speciality', 'Salary', 'Designation', 'YearOfJoining',
                        'WorkExperince', 'NoOfCollegesWorked', 'YearSinceLastPromotion', 'DistanceFromHome',
                        'DepartmentOvertime', 'Gender']

        # Check if all columns in selected_col exist in df1
        if not all(col in df1.columns for col in selected_col):
            return "Error: One or more columns in selected_col do not exist in df1.", 400  # Return an error message and status code 400

        selected_data = df1[selected_col]
        json_data = selected_data.to_dict(orient='records')

        # Render an HTML template with the JSON data
        return render_template('dashboard/employee.html', data=json_data)
    except Exception as e:
        return str(e), 500
   #return render_template('dashboard/employee.html')

@app.route('/visualization')
def visualization():

    return render_template('dashboard/visualization.html')

@app.route('/plot1')
def plot1():
    return send_from_directory(app.config['STATIC_FOLDER'],'plot1.png')
@app.route('/plot2')
def plot2():
    return send_from_directory(app.config['STATIC_FOLDER'],'plot2.png')
@app.route('/plot3')
def plot3():
    return send_from_directory(app.config['STATIC_FOLDER'],'plot3.png')

@app.route('/analytics')
def analytics():
    quitEmployee = df[df['IntentionToQuit'] == 1]
    count_of_quit_employee = len(quitEmployee)
    total_employees = len(df)
    attrition_rate = (count_of_quit_employee / total_employees) * 100
    if attrition_rate> 20:
        severity ='High'
    else:
        severity='Low'

    condition = df['IntentionToQuit'] == 1
    comp_employees = df['New Department'] == 1
    comp_quit_emp=df[condition][comp_employees]
    count_of_quit_comp_employee=len(comp_quit_emp);

    count_comp_employees=len(comp_employees);
    comp_attrition_rate=(count_of_quit_comp_employee/count_comp_employees)*100;
    if comp_attrition_rate> 20:
        comp_severity ='High'
    else:
        comp_severity='Low'

    csbs_employees = df['New Department'] == 2
    csbs_quit_emp = df[condition][csbs_employees]
    count_of_quit_csbs_employee = len(csbs_quit_emp);
    count_csbs_employees = len(csbs_employees);
    csbs_attrition_rate = (count_of_quit_csbs_employee / count_csbs_employees) * 100;
    if csbs_attrition_rate > 20:
        csbs_severity = 'High'
    else:
        csbs_severity = 'Low'

    return render_template('dashboard/analytics.html', attrition_rate=attrition_rate,severity=severity,comp_attrition_rate=comp_attrition_rate,comp_severity=comp_severity,csbs_attrition_rate=csbs_attrition_rate,csbs_severity=csbs_severity)
if __name__ == "__main__":
    app.run(debug=True)