from functions.diff import numerical_gradient


def gradient_decent(f, initial_x, learning_rate=0.01, num_step=100):
    x = initial_x.copy()
    for i in range(num_step):
        grad = numerical_gradient(f, x)
        x -= learning_rate * grad
    return x
