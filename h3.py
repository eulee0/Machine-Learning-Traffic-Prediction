from flask import Flask, request, render_template

from sklearn.linear_model import Ridge

app = Flask(__name__)

data_set = list()
data_out = list()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training', methods=['POST'])
def training():
    road = int(request.form['road_id'])
    direction = int(request.form['direction'])
    day = int(request.form['day'])
    time = int(request.form['time'])
    status = int(request.form['status'])

    data_set.append([road, direction, day, time])
    data_out.append([status])

    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    clf = Ridge(alpha=1.0)
    clf.fit(data_set, data_out)

    road = int(request.form['road_id'])
    direction = int(request.form['direction'])
    day = int(request.form['day'])
    time = int(request.form['time'])

    answer = clf.predict([road, direction, day, time])
    return '[[' + str(road) + ', ' + str(direction) + ', ' + str(day) + ', ' + str(time) + \
           ']] => ' + str(answer)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
