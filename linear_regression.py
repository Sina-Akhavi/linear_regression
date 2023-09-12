import numpy as np
import matplotlib.pyplot as plt
from utilities.utils import *

# ------------------ initialization ------------------

x_train, y_train = load_data()

# print(f'type of x_train is {type(x_train)}')
# print(f'type of y_train is {type(y_train)}')

# number_items = 6
# print(f'the first {number_items}th Xs: {x_train[:6]}')
# print(f'the first {number_items}th Ys: {y_train[:6]}')
# print(f'number of x_train is {x_train.shape}')
# print(f'number of y_train is {y_train.shape}')

# ------------------ plotting ------------------

# plt.scatter(x_train, y_train, c='r', marker='x')
# plt.axis([0, 30, 0, 40])
# plt.title('population vs. profits')
# plt.xlabel('population times 10,000')
# plt.ylabel('profit in $10,000')

# plt.show()

# ------------------ cost function ------------------
def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    cost = cost / (2 * m)

    return cost


# ---------------- test the correctness ----------------

# initial_w = 2
# initial_b = 1

# cost = compute_cost(x_train, y_train, initial_w, initial_b)
# print(type(cost))
# print(f'Cost at initial w (zeros): {cost:.3f}')

# ------------------ compute gradients ------------------
def compute_gradients(x, y, w, b):
    
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] +  b
        
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += f_wb - y[i]
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

# ---------------- test gra dients ----------------
# test 1

# initial_w = 0
# initial_b = 0

# tmp_dj_dw, tmp_dj_db = compute_gradients(x_train, y_train, initial_w, initial_b)
# print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

# test 2

# test_w = 0.2
# test_b = 0.2
# tmp_dj_dw, tmp_dj_db = compute_gradients(x_train, y_train, test_w, test_b)

# print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)

# ----------------------- gradient descent -----------------------

def gradient_descent(x, y, w, b, learning_rate, iter_num, compute_gradients, compute_cost):

    j_history = []
    for i in range(iter_num):

        dj_dw, dj_db = compute_gradients(x, y, w, b)

        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

        j_history.append(compute_cost(x, y, w, b))
        if i % 150 == 0:
            cost = compute_cost(x, y, w, b)
            print(f'iteration {i}:           if w={w} and b={b}, then cost={cost}')
            

    return w, b, j_history


# ----------------------- test gradient descent -----------------------


initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w, b, j_history = gradient_descent(x_train ,y_train, initial_w, initial_b, alpha, iterations
                       , compute_gradients, compute_cost)
print("w,b found by gradient descent:", w, b)

fig, ax = plt.subplots(1, 2)
separator = 20
ax[0].plot(range(separator), j_history[:separator])
ax[0].set_title('learning curve')
ax[0].set_xlabel('iterations')
ax[0].set_ylabel('cost')
ax[0].set_ylim(5, 7)
ax[0].set_xlim(0, 20)

ax[1].plot(range(separator, iterations), j_history[separator:])
ax[1].set_title('learning curve')
ax[1].set_xlabel('iterations')
ax[1].set_ylabel('cost')
ax[1].set_ylim(4, 7)
ax[1].set_xlim(separator, iterations)
plt.show()

# ------------------ plotting the linear fit ------------------
m = x_train.shape[0]
predictions = np.zeros(m)

for i in range(m):
    predictions[i] = w * x_train[i] + b



plt.scatter(x_train, y_train, c='r', marker='x')
plt.axis([0, 30, 0, 40])
plt.title('population vs. profits')
plt.xlabel('population times 10,000')
plt.ylabel('profit in $10,000')

plt.plot(x_train, predictions, c='b')
plt.axis([3, 25, 0, 25])

plt.show()

