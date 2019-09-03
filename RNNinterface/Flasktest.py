
import math
from flask import Flask, render_template, request
from test import pred


def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("interface.html", n="0%")


@app.route('/', methods=['GET', 'POST'])
def getvalue():
    textF = request.form['text1']
    if textF != "":
        return render_template("interface.html", n=(str(truncate(float(pred([textF]))*100, 1))+'%'), t=textF)
    else:
        return render_template("interface.html", n="0%")


if __name__ == '__main__':
    app.run(debug=True)

