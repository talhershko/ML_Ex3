import numpy as np
from scipy.special import softmax


ReLU = lambda x: np.maximum(0, x)

def read_data():
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    return train_x, train_y, test_x


def train_dev_split(train_x, train_y):
    '''
    Split the original train file to 80% train and 20% validation data.
    '''
    # Number of samples
    N = len(train_x)
    # Shuffle data and labels in the same way
    arr = np.arange(N)
    # np.random.seed(seed=42)
    np.random.shuffle(arr)
    train_x = train_x[arr]
    train_y = train_y[arr]
    # Split the shuffled data to 80% train and 20% validation
    stop_index = int(N*0.8)
    train_data = list(zip(train_x[:stop_index], train_y[:stop_index]))
    dev_data = list(zip(train_x[stop_index:], train_y[stop_index:]))
    return train_data, dev_data


def forward(x, params):
    """
    Compute all layers from input to output and put results in a list.
    """
    # We would like to compute all x_i before x_n (for computing gradients later)
    # Put all previous x_i in the stack x_list
    # Start with input x
    x_list = [np.array(x)]
    for layer in range(int(len(params) / 2)):
        # Get layer params
        W, b = params[2 * layer], params[2 * layer + 1]
        # Compute next layer input - ReLU(x*W + b)
        # Prevent activation in linear classifier
        # Prevent activation for output layer
        if len(params) / 2 == 1 or layer == len(params) / 2 - 1:
            next_x = np.dot(x_list[-1], W) + b
        else:
            next_x = ReLU(np.dot(x_list[-1], W) + b)
        # Add current layer output to the stack
        x_list.append(next_x)
    # Softmax last output x_n
    x_list[-1] = softmax(x_list[-1])
    return x_list


def backprop(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    we should not have gW2 and gb2.)
    """
    # Hard cross-entropy loss (Same as NLLL ? todo: check!)
    probs = forward(x, params)[-1]
    loss = -np.log(probs[y])
    # Mark x_i+1 = RELU(x_i * W_i + b_i)
    # Last is x_n+1 = x_n * W_n + b_n (before softmax)
    # As in mlp1 we know d(loss)/d(x_n+1)
    dl_dx = probs.copy()
    dl_dx[y] -= 1
    # Now dx_n+1/dx_n = W_n
    # dx_n+1/dW_n = x_n
    # dx_n+1/db_n = 1
    # Compute all x_i before x_n+1
    hid_x = forward(x, params)
    # Drop the output (we do not need it for gradient computation)
    hid_x.pop()
    # Set gradient list
    grads = []
    # Set params stack
    p = params.copy()
    # It is convenient to work with the reversed stack
    p.reverse()
    # Compute last gradients as in mlp1
    gbn = dl_dx
    gWn = np.dot(np.transpose([hid_x.pop()]), np.transpose(np.transpose([dl_dx])))
    # Add the gradients (reversed) to grads list
    grads.extend([gbn, gWn])
    # Compute back layers gradients
    for k in range(int(len(params) / 2) - 1):
        # Current x and parameters needed for computing
        xi = hid_x.pop()
        b_next, W_next, bi, Wi = p[0:4]
        # Mark m_i = x_i * W_i + b_i
        mi = np.dot(xi, Wi) + bi
        # Mark h_i = ReLU(m_i) = x_i+1
        hi = ReLU(mi)
        # dh_i/dm_i = ReLU'(m_i)
        dhi_dmi = (mi>=0).astype(int)
        # dm_i/dW_i = x_i
        # dm_i/db_i = 1
        # Apply the chain rule (use known gradients)
        gbi = np.dot(W_next, grads[-2]) * dhi_dmi
        gWi = np.dot(np.transpose([xi]), np.transpose(np.transpose([gbi])))
        # Add the gradients (reversed) to grads list
        grads.extend([gbi, gWi])
        # Drop next layer parameters
        p.pop(0)
        p.pop(0)
    # Reverse back the gradient list
    grads.reverse()
    return loss, grads


def init_net(dims):
    '''
    Initialize neural network parameters (assuming ReLU activation)
    '''
    # Set parameters list
    params = []
    for layer in range(1, len(dims)):
        # He et al's suggestion
        W = np.random.randn(dims[layer-1], dims[layer]) * np.sqrt(2 / dims[layer-1])
        b = np.random.randn(dims[layer]) * np.sqrt(2 / dims[layer-1])
        # Add new params to the list
        params.extend([W, b])
    return params


def train(train_data, dev_data, params, epochs, learning_rate):
    for epoch in range(epochs):
        # total loss for epoch
        epoch_loss = 0
        # Shuffle data for SGD
        np.random.shuffle(train_data)
        for x, y in train_data:
            y = int(y)
            # Compute loss and the gradients
            loss, grads = backprop(x, y, params)
            epoch_loss += loss
            # update the parameters according to the gradients
            # and the learning rate.
            for i, param in enumerate(params):
                param -= learning_rate * grads[i]
        # Compute and print accuracy
        train_loss = epoch_loss / len(train_data)
        train_acc = evaluate(train_data, params)
        dev_acc = evaluate(dev_data, params)
        print('epoch:', epoch + 1, 'train_loss:', train_loss,
              'train_acc:', train_acc, 'dev_acc:', dev_acc)
    return params


def evaluate(data, params):
    '''
    Compute validation set accuracy using trained parameters
    '''
    # Initialization
    accuracy = 0
    # Number of samples
    N = len(data)
    for x, y in data:
        y = int(y)
        # Calculate the class scores
        probs = forward(x, params)[-1]
        # Find argmax
        y_hat = np.argmax(probs)
        # Correct classification
        if y_hat == y:
            accuracy += 1
    # Divide by number of samples
    accuracy /= N
    return accuracy


def predict(test_x, params, file):
    '''
    Predict test set labels using trained parameters
    '''
    with open(file, 'w+') as file:
        for x in test_x:
            # Calculate the class scores
            probs = forward(x, params)[-1]
            # Find argmax
            y_hat = np.argmax(probs)
            # Write to file
            file.write(str(y_hat) + '\n')


def main():
    # Read data
    train_x, train_y, test_x = read_data()
    input_size = len(train_x[0])
    # Train dev split
    train_data, dev_data = train_dev_split(train_x, train_y)
    # Initialize net params
    # params = init_net([input_size, 300, 50, 10])
    # params = init_net([input_size, 128, 10])
    # params = init_net([input_size, 256, 128, 100, 10])
    params = init_net([input_size, 1000, 500, 100, 10])  # best with 81.9% on dev
    # todo: find best hyper-parameters
    # train params
    params = train(train_data, dev_data, params, epochs=5, learning_rate=1e-5)
    predict(test_x, params, 'test_y')
    pass


if __name__ == '__main__':
    main()
    # check()


# USE THIS FOR GRADIENT CHECKS (see grad_check.py)


def check():
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W1, b1, W2, b2, W3, b3 = init_net([3, 4, 8, 5])

    def _loss_and_W1_grad(W1):
        global b1, W2, b2, W3, b3
        loss, grads = backprop([-7.3, 5, 0], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[0]

    def _loss_and_b1_grad(b1):
        global W1, W2, b2, W3, b3
        loss, grads = backprop([-9, 22, 3.2], 2, [W1, b1, W2, b2, W3, b3])
        return loss, grads[1]

    def _loss_and_W2_grad(W2):
        global W1, b1, b2, W3, b3
        loss, grads = backprop([-1, 7, 4], 1, [W1, b1, W2, b2, W3, b3])
        return loss, grads[2]

    def _loss_and_b2_grad(b2):
        global W1, b1, W2, W3, b3
        loss, grads = backprop([1, 2, 3], 0, [W1, b1, W2, b2, W3, b3])
        return loss, grads[3]

    def _loss_and_W3_grad(W3):
        global W1, b1, W2, b2, b3
        loss, grads = backprop([-1, 78, 4], 1, [W1, b1, W2, b2, W3, b3])
        return loss, grads[4]

    def _loss_and_b3_grad(b3):
        global W1, b1, W2, b2, W3
        loss, grads = backprop([1, 2, 7.25], 3, [W1, b1, W2, b2, W3, b3])
        return loss, grads[5]

    for _ in range(10):
        W1 = np.random.randn(W1.shape[0], W1.shape[1])
        b1 = np.random.randn(b1.shape[0])
        W2 = np.random.randn(W2.shape[0], W2.shape[1])
        b2 = np.random.randn(b2.shape[0])
        W3 = np.random.randn(W3.shape[0], W3.shape[1])
        b3 = np.random.randn(b3.shape[0])
        gradient_check(_loss_and_b1_grad, b1)
        gradient_check(_loss_and_W1_grad, W1)
        gradient_check(_loss_and_b2_grad, b2)
        gradient_check(_loss_and_W2_grad, W2)
        gradient_check(_loss_and_b3_grad, b3)
        gradient_check(_loss_and_W3_grad, W3)
