import numpy as np
from random import sample, choice
from src.SGD_variance_reduction.size import size


"""
The stochastic optimizer for: SGD, SVRG, SARAH
SGD, SVRG and SARAH for quadratic loss and Gaussian input data
first create the updates for each iteration, 
then optimizes via different schemes of iteration loop
"""


class stochastic_optimizer(object):
    def __init__(
        self,
        # the loss function class (contains grad info)
        function,
        # training and test samples
        training_sample_x,
        training_sample_y,
        test_sample_x,
        test_sample_y,
    ):
        self.function = function
        self.training_sample_x = training_sample_x
        self.training_sample_y = training_sample_y
        self.test_sample_x = test_sample_x
        self.test_sample_y = test_sample_y

    # the SGD estimator update = the change of parameter via stochastic gradients
    def SGD_update(self, w, lr, batchsize):
        # detect the size of the training set
        trainingsize = size(self.training_sample_x, self.training_sample_y)
        # randomly choose the index set that forms the mini-batch
        batch_index = sample(list(range(0, trainingsize)), batchsize)
        # from the mini-batch index set select the corresponding training samples (x, y)
        batch_x = []
        batch_y = []
        for i in range(batchsize):
            batch_x.append(self.training_sample_x[batch_index[i]])
            batch_y.append(self.training_sample_y[batch_index[i]])
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        # calculate the stochastic gradient updates
        grad = self.function.average(w, batch_x, batch_y, self.function.grad)
        update = -lr * grad
        return update

    # the SGD optimizer, iterates a certain number of steps to update the weights
    def SGD_optimizer(self, w_init, steps, lr, batchsize):
        w_current = w_init
        trajectory_w = []
        loss_list = []
        test_error_list = []
        for i in range(steps):
            # record the current model weights w
            trajectory_w.append(w_current)
            # calculate the test error for the current model weights w
            test_error = self.function.value(
                w_current, self.test_sample_x, self.test_sample_y
            )
            test_error_list.append(test_error)
            # calculate the training error (loss) for the current model weights w
            loss_list.append(
                self.function.average(
                    w_current,
                    self.training_sample_x,
                    self.training_sample_y,
                    self.function.value,
                )
            )
            # update w via stochastic optimization
            w = w_current + self.SGD_update(w_current, lr, batchsize)
            w_current = w
        return trajectory_w, loss_list, test_error_list

    # the SVRG estimator update =
    #   the inner loop update via variance-reduced stochastic gradients
    #  w_checkpoint is the checkpoint w value recorded,
    #   i.e., the w-tilde in the Algorithm in the SVRG paper
    #   (Johnson-Zhang, NIPS 2013)
    #  grad_checkpoint is the checkpoint grad value recorded,
    #   i.e. mu_tilde = grad P(w_checkpoint)
    def SVRG_update(self, w, lr, w_checkpoint, grad_checkpoint):
        # sample one random index from the set [0,...,training_size-1]
        trainingsize = size(self.training_sample_x, self.training_sample_y)
        index = choice(list(range(0, trainingsize)))
        # return the variance-reduced stochastic gradient
        grad_1 = self.function.grad(
            w, self.training_sample_x[index], self.training_sample_y[index]
        )
        grad_2 = self.function.grad(
            w_checkpoint, self.training_sample_x[index], self.training_sample_y[index]
        )
        grad = grad_1 - grad_2 + grad_checkpoint
        update = -lr * grad
        return update

    # the SVRG optimizer, iterates a certain number of epochs,
    #   with each epoch under a cetain length (epochlength=m),
    #   to get the updated weights
    def SVRG_optimizer(self, w_init, epochs, epochlength, lr):
        w_checkpoint = w_init
        trajectory_w = []
        loss_list = []
        test_error_list = []
        for s in range(epochs):
            # grad_checkpoint is the gradient value of
            #   the empirical loss at the checkpoint,
            #  i.e., the mu-tilde
            grad_checkpoint = self.function.average(
                w_checkpoint,
                self.training_sample_x,
                self.training_sample_y,
                self.function.grad,
            )
            # record the current model weights w
            trajectory_w.append(w_checkpoint)
            # calculate the test error for the current model weights w
            test_error = self.function.value(
                w_checkpoint, self.test_sample_x, self.test_sample_y
            )
            test_error_list.append(test_error)
            # calculate the training error (loss) for the current model weights w
            loss_list.append(
                self.function.average(
                    w_checkpoint,
                    self.training_sample_x,
                    self.training_sample_y,
                    self.function.value,
                )
            )
            # the inner loop list of w is initialized,
            #   will fill in w_1,...,w_m (m=epochlength)
            w_innerloop_list = []
            # start the inner loop at w_checkpoint
            w_current = w_checkpoint
            for i in range(epochlength):
                w_next = w_current + self.SVRG_update(
                    w_current, lr, w_checkpoint, grad_checkpoint
                )
                w_innerloop_list.append(w_next)
                w_current = w_next
            # update the checkpoint by randomly select from
            #   the list in the inner loop w_1,...,w_m (m=epochlength)
            w_checkpoint = choice(list(w_innerloop_list))
        return trajectory_w, loss_list, test_error_list

    # the SARAH estimator update =
    #   the inner loop update via variance-reduced stochastic gradients
    def SARAH_update(self, w_current, w_previous, v_previous):
        # sample one random index from the set [0,...,training_size-1]
        trainingsize = size(self.training_sample_x, self.training_sample_y)
        index = choice(list(range(0, trainingsize)))
        # return the SARAH version of the variance-reduced stochastic gradient
        grad_1 = self.function.grad(
            w_current, self.training_sample_x[index], self.training_sample_y[index]
        )
        grad_2 = self.function.grad(
            w_previous, self.training_sample_x[index], self.training_sample_y[index]
        )
        v_current = grad_1 - grad_2 + v_previous
        return v_current

    # the SARAH optimizer, iterates a certain number of epochs,
    #   with each epoch under a cetain length (epochlength=m),
    #   to get the updated weights
    def SARAH_optimizer(self, w_init, epochs, epochlength, lr):
        w_checkpoint = w_init
        trajectory_w = []
        loss_list = []
        test_error_list = []
        for s in range(epochs):
            # record the current model weights w
            trajectory_w.append(w_checkpoint)
            # calculate the test error for the current model weights w
            test_error = self.function.value(
                w_checkpoint, self.test_sample_x, self.test_sample_y
            )
            test_error_list.append(test_error)
            # calculate the training error (loss) for the current model weights w
            loss_list.append(
                self.function.average(
                    w_checkpoint,
                    self.training_sample_x,
                    self.training_sample_y,
                    self.function.value,
                )
            )
            # the inner loop list of w is initialized,
            #   will fill in w_0,...,w_m (m=epochlength)
            w_innerloop_list = []
            # start the inner loop at w_checkpoint
            w_previous = w_checkpoint
            w_innerloop_list.append(w_previous)
            # v_{t-1}
            v_previous = self.function.average(
                w_checkpoint,
                self.training_sample_x,
                self.training_sample_y,
                self.function.grad,
            )
            w_current = w_previous - lr * v_previous
            for i in range(epochlength):
                w_innerloop_list.append(w_current)
                v_current = self.SARAH_update(w_current, w_previous, v_previous)
                w_next = w_current - lr * v_current
                v_previous = v_current
                w_previous = w_current
                w_current = w_next
            # update the checkpoint by randomly select from the list
            #   in the inner loop w_1,...,w_m (m=epochlength)
            w_checkpoint = choice(list(w_innerloop_list))
        return trajectory_w, loss_list, test_error_list
