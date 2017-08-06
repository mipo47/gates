from gates.gate_net import *


def get_accuracy(net, dataset, is_one_hot):
    # validation accuracy
    softmax = Softmax(net)
    values = softmax.forward(dataset.inputs)
    predicted = np.argmax(values, axis=1)
    outputs = np.argmax(dataset.outputs, axis=1) if is_one_hot else dataset.outputs
    correct = np.sum(outputs == predicted)
    return correct / dataset.length()


# TODO: try weight regularization, cross validation, dropout
def train_model(data_info, train, validation, test, explore_steps=1000, validate_steps=10, show_progress=0.99, display_L2_loss=False):
    is_classification = data_info['loss'] == SoftmaxLoss

    # ------- START CREATE NETWORK ---------------------------------

    # input layer
    net = Gate(size=train.input_count())

    # default activation function for hidden layers
    activation = data_info['activation'] if 'activation' in data_info else Tanh

    # hidden layers
    for layer_size in data_info['layers']:
        net = l = Layer(net, layer_size, activation)

    # last/output layer
    output_count = data_info['output_count'] if 'output_count' in data_info else train.output_count()
    net = Layer(net, output_count,
                None if is_classification else Sigmoid)  # no activation is needed for classification

    # ------- END CREATE NETWORK ------------------------------------

    # choose loss function to measure network efficiency
    loss = data_info['loss'](net)
    if 'output_count' in data_info:
        loss.is_one_hot = False

    # always display L2 loss score, even for softmax classification
    display_loss = loss \
        if data_info['loss'] == LossL2 or not display_L2_loss \
        else LossL2(Softmax(net))

    # save weights of best scored network here
    checkpoint = Checkpoint(net)

    # select learning optimization algorithm
    optimizer = \
        data_info['optimizer'](data_info['learning_rate']) \
            if 'optimizer' in data_info \
            else Adam()

    shown_loss = backup_loss = valid_loss = 9999999
    shown_step = step = 0
    batch_size = data_info["batch_size"] if "batch_size" in data_info else 1024
    while True:
        # train in mini batches
        inputs, outputs = train.get_batch(batch_size)
        train_loss = loss.train(optimizer, inputs, outputs)

        # check validation score each 10-th step
        if (step + 1) % validate_steps == 0 or step == 0:
            # convert loss to L2 loss if needed
            if display_loss != loss:
                train_loss = display_loss.validate(train.inputs, train.outputs)

            valid_loss = display_loss.validate(validation.inputs, validation.outputs)
            sum_loss = valid_loss  # if display_loss == loss else loss.validate(validation.inputs, validation.outputs)

            if sum_loss < backup_loss:
                # save weights with best validation score
                backup_loss = sum_loss
                checkpoint.backup()

                # stop training if we have very low score
                if sum_loss < 1e-09: break

            # end training when loss is not improved enough for 'explore_steps' steps
            elif step - shown_step > explore_steps:
                print("decided to end training")
                print(step + 1, "train/validation loss", train_loss, valid_loss)
                break

        # show losses if our score improved more than 1%
        if step == 0 or sum_loss < shown_loss * show_progress:
            if is_classification:
                accuracy = get_accuracy(net, validation, loss.is_one_hot)
                print(step + 1, "train/validation/accuracy", train_loss, valid_loss, accuracy)
            else:
                print(step + 1, "train/validation loss", train_loss, valid_loss)

            shown_step = step
            shown_loss = sum_loss

        step += 1

    # restore best scored network weights
    checkpoint.restore()

    # show final score
    train_loss = display_loss.validate(train.inputs, train.outputs)
    valid_loss = display_loss.validate(validation.inputs, validation.outputs)
    test_loss = display_loss.validate(test.inputs, test.outputs)
    print("train/validation/test:", [train_loss, valid_loss, test_loss])

    if is_classification:
        accuracy = get_accuracy(net, test, loss.is_one_hot)
        print("test accuracy", accuracy)

    return checkpoint