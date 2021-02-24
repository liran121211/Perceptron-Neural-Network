from matplotlib import pyplot as plt
import numpy as np
import random as rnd
from matplotlib.animation import FuncAnimation as FA
import time
'''
docstring:
Point Class (X, Y) and Label (classify as 1 or -1)
Perceptron Class (weights [], learning rate (C constant))

Learn Model: y= (weight * x) + C
Sign Function: (return 1 or -1 based on input (n)
Guess Function: multiply weight[i] by input[i] in every neuron and send to Sign Function to determine -1 or 1
Train Function: receive ([p.x, p.y] and p.label) and try to guess the p.label of the object bt adjusting the weights of the perceptron

Simplify User Interface with:
Draw Function - Use MetaPlotLib to visualize the points on the graph
Setup Function - Initialize staring values/objects
'''


def setup(length):
    for index in range(length): # make 100 points array
        points.append(Point(rnd.uniform(-10,10), rnd.uniform(-10,10)))
    print('\n{0} Points (X,Y) Initialized!, Starting...\n'.format(length))

    #For Debug
    '''inputs = initial_inputs #Inputs of the neuron, (X0, X1)
    guess = P.guess(inputs) #guess 1/-1
    print(guess)'''

def draw():
    plt.cla() #Clean Graph for each train
    for index in range(0,len(points)): #Plot first data of all (-1/1 points)
        if(points[index].label == 1):
            plt.plot(points[index].x, points[index].y, 'o',c='red')
        else:
            plt.plot (points[index].x, points[index].y, 'o', c='blue')

        x = np.linspace(-10, 10, 100) #y=mx+n line
        y=x
        plt.plot(x,y, linestyle='solid') #plot linear line in the middle
        plt.xlabel('Y Axis')
        plt.ylabel('Y Axis')
        plt.title('First Neuron System!')

    x_axis_before_plot = []
    y_axis_before_plot = []
    x_axis_to_plot = []
    y_axis_to_plot = []
    bad_guess_x_axis = []
    bad_guess_y_axis = []
    count_fails = 0 #how many points failed to guess

    for point in points:
        P.train([point.x, point.y], point.label)
        guess = P.guess([point.x, point.y])

        if (guess == point.label):
            x_axis_before_plot.append(point.x)
            y_axis_before_plot.append(point.y)
        else:
            count_fails += 1
            bad_guess_x_axis.append(point.x)
            bad_guess_y_axis.append(point.y)
            plt.plot(point.x, point.y, 'o', c='black', markeredgewidth=3)

    plt.plot(4, 2, label="Classifier: 1", color = 'blue')
    plt.plot(4, 2, label="Classifier: -1", color = 'red')
    plt.plot(4, 2, label="Right Guess (^_^)", color = 'green')
    plt.plot(4, 2, label="Bad Guess (@_@)", color = 'black')
    plt.legend(loc="upper left")


    def animate(index): #iterator for animation
        if (index<len(x_axis_before_plot)):
            x_axis_to_plot.append(x_axis_before_plot[index])
            y_axis_to_plot.append(y_axis_before_plot[index])
        else:
            print('Neuron failed to learn [{0}] points. Retraining <(#_#)>....'.format(count_fails))
            if (count_fails > 0):
                draw() #Recurrsive call untill all points has been learned!
            else:
                print('\nlearned successfully! Exits in 5 seconds...')
                time.sleep(5)
                exit()

        plt.plot(x_axis_to_plot, y_axis_to_plot, 'o' ,c='green',  markeredgewidth=3)

    fig = plt.gcf() #Call Current Figure (Graph)
    fig.set_size_inches(7,7) #Set size in inches
    plt.get_current_fig_manager().window.wm_geometry("+1000+10")
    animation_function = FA(plt.gcf(), animate, interval=milliseconds)
    plt.tight_layout()
    plt.show()

def sign(n): #Activision Function - If n is positive return 1 else -1
    if n >= 0:
        return 1;
    else:
        return -1

class Point ():
    def __init__(self, x, y):#Constructor
        self.x = x
        self.y = y
        if (x> y):
            self.label = 1
        else:
            self.label = -1

    def __str__(self):
        return ('Point ({0}, {1}) ->Label: {2}'.format(self.x, self.y, self.label))

class Perceptron():
    def __init__(self,countWeights, intercept_C): #Constructor
        self.learning_rate = intercept_C
        self.weights = []
        for index in range(countWeights): #Initialize Weights randomly
            self.weights.append(rnd.uniform(-1, 1))

    def __str__(self):
        return ('Perceptron (Weights ({0}) | Learning Rate: {1})'.format(self.weights, self.learning_rate))

    def guess(self, point_x_y):
        sum = 0.0
        for index in range(len(self.weights)):
            sum += self.weights[index] * point_x_y[index]
        guessed_value = sign(sum)
        return guessed_value

    def train(self, point_x_y, target):
        calculated_guess = self.guess(point_x_y)
        error = target - calculated_guess

        #Tune all the weights
        for index in range(len(self.weights)):
            self.weights[index] += error * point_x_y[index] * self.learning_rate


points = []

if (input('Normal Mode [1] / Debug Mode [2]: ') == '1'):
    learning_rate = rnd.uniform(0.001, 0.005)
    weights = 2
    initial_inputs = [rnd.uniform(-29, -30), rnd.uniform(-98, -100)]
    num_of_points = rnd.randint(30, 30)
    milliseconds = 50  # time for each iteration in the graph (milliseconds/1000 - for seconds)
    P = Perceptron(weights, learning_rate)

    print('Parameters were set radomally:\n'
          ' 1) Learning Rate: {0}\n'
          ' 2) Weights: {1}\n'
          ' 3) Initial Inputs: {2}\n'
          ' 4) Number of Points: {3}\n'
          ' 5) Seconds: {4}\n'.format(learning_rate, weights, initial_inputs, num_of_points, milliseconds/1000))

    setup(num_of_points)
    draw()
else:
    first_input = 0.0
    second_input = 0.0
    weights = 2
    print('ReadMe: In this panel you can adjust the values of the neuron \n'
          ' 1) Learning Rate ---> How fast the machine will learn to find all correct point  (label) guess \n'
          ' 2) Weights ---> The coefficient of the point (X,Y) value, This system support 2D point, This is unchangeable value (2) \n'
          ' 3) Initial Inputs ---> An array that contain (X,Y) value of the point, used for Y= mX +b \n'
          ' 4) Number of Points ---> You can decide how many points you wish to train \n'
          ' 5) Milliseconds ---> How fast would you like for this algorithm to run \n')
    learning_rate = float(input('Learning Rate: '))
    first_input = float(input('First [n,] parameter: '))
    second_input = float(input('Second [,n] parameter: '))
    num_of_points = int(input('Number of Points: '))
    milliseconds = int(input('Accelaration (seconds * 1000): '))
    P = Perceptron(2, learning_rate)

    setup(int(num_of_points))
    draw()


