import argparse
import csv
import logging.config
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.special


class SFNN(object):
    def __init__(self, X, Y, h1=16, h2=16, h3=16, h4=16, Nh=4, M=30, lr=0.1):
        '''Initialize the Stochastic Feedforward Neural Network.'''
        # seed random number generator
        np.random.seed(0)

        # read training examples
        with open(X) as f, open(Y) as g:
            self._X = [x for x in csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)]
            self._Y = [y for y in csv.reader(g, quoting=csv.QUOTE_NONNUMERIC)]

        # D-dimensional data pairs
        self._D = len(self.X[0])
        log.debug('%d-dimensional data pairs', self._D)

        # set number of training patterns
        self._N = len(self.X)
        log.debug('%d training patterns', self._N)

        # initialize residual variance to variance of y
        # self._var = np.var(self.Y)

        self.variances = []
        max_x = np.max(self.X)
        min_x = np.min(self.X)
        self.target_variance(min_x, max_x)

        print "self.variance"

        print self.variances

        print "why didn;t you print"

        # initialize residual variance to variance of y
        self._var = np.var(self.Y)

        log.debug('sigma^2 = %f', self._var)

        # set hidden layer sizes
        self._H = h1, h2, h3, h4

        # initialize model
        self._W1 = self._init_weights(h1, self._D)
        self._W2 = self._init_weights(h2, h1)
        self._W3 = self._init_weights(h3, h2)
        self._W4 = self._init_weights(h4, h3)
        self._W5 = self._init_weights(self._D, h4)
        
        # initialize bias
        self._Bh1 = np.zeros((h1))
        self._Bh2 = np.zeros((h2))
        self._Bh3 = np.zeros((h3))
        self._Bh4 = np.zeros((h4))
        self._By  = np.zeros((len(self._Y[0])))

        # set number of stochastic nodes in hybrid layers
        self._Nh = Nh

        # set number of importance samples
        self._M = M

        # set learning rate
        self._lr = lr

        # set activation function
        self._f = lambda x: scipy.special.expit(x)
        self._f_prime = lambda sigma_x: sigma_x * (1.0 - sigma_x)

    @property
    def X(self):
        return self._X
    

    @property
    def Y(self):
        return self._Y

    def _init_weights(self, fanout, fanin):
        '''Use normalized initialization [Glorot'10].'''
        r = 4 * math.sqrt(6) / math.sqrt(fanin + fanout)
        return np.random.uniform(-r, r, (fanout, fanin))
    def target_variance(self, mini, maxi):
                    # var = np.zeros((self._N,1))
        distance = maxi - mini
        batch_size = 5
        begining_index = mini
        index = mini + distance / 5
        for i in range(batch_size):
            print "--------------------------------------------------------"
            sum_i = 0
            total_sum = 0
            number_of_points = 0
            for n in range(self._N):
                if (begining_index < self.X[n][0]) and ( self.X[n][0] < index):
                    total_sum += self.Y[n][0]
                    number_of_points += 1
                    print self.Y[n][0]

            if number_of_points == 0:
                break
            mean_total_sum = total_sum / number_of_points
            variance = 0
            number_of_points = 0
            for n in range(self._N):
                if (begining_index < self.X[n][0]) and ( self.X[n][0] < index):
                    variance += (mean_total_sum - self.Y[n][0]) ** 2
                    number_of_points += 1

            variance = variance / number_of_points
            self.variances.append(variance)
            begining_index = index
            index = index*2


    def get_variance(self, pattern):

        for index in range(len(self.variances)):
            if pattern < self.variances[index]:
                return self.variances[index]

        return 0

    def _approximate_estep(self, pattern):
        '''Do the approximate E-step.'''
        # set pattern
        # print "pattern"
        # print pattern
        self._x = np.array(pattern).T

        # h1 is a deterministic layer
        self._h1 = self._f(np.dot(self._W1, self._x) + self._Bh1)

        # get hidden layer sizes
        _, h2, h3, h4 = self._H

        # initialize other layers
        self._h2 = np.empty((h2, self._M))
        self._h3 = np.empty((h3, self._M))
        self._h4 = np.empty((h4, self._M))
        self._y  = np.empty((self._D, self._M))

        # initialize importance weights
        self._iweights = np.empty((self._D, self._M))

        h2_determ = self._f(np.dot(self._W2, self._h1) + self._Bh2)
        # for layer 2, compute the Bernoulli distribution
        conditional_prior_layer_2 = np.random.binomial(1, h2_determ)
        
        # var = np.zeros((self._M,1))
        # importance sampling
        for m in range(self._M):
            # h2 is a hybrid layer
            # var[m] = (np.array(target).T - self.sample(self._y[:,m])) ** 2
            self._h2[:,m] = np.copy(h2_determ)
            # sample a vector of Nh Bernoulli random variables
            self._h2[:self._Nh,m] = np.random.binomial(1, self._h2[:self._Nh,m])

            # h3 is a hybrid layer
            self._h3[:,m] = self._f(np.dot(self._W3, self._h2[:,m]) + self._Bh3)
            # sample a vector of Nh Bernoulli random variables
            self._h3[:self._Nh,m] = np.random.binomial(1, self._h3[:self._Nh,m])

            # h4 is a deterministic layer
            self._h4[:,m] = self._f(np.dot(self._W4, self._h3[:,m]) + self._Bh4)

            # compute output
            self._y[:,m] = self._f(np.dot(self._W5, self._h4[:,m]) + self._By)
            # print "Index = ", self.X.index(pattern)
            self._iweights[:,m] = np.random.normal(self._y[:,m], self.get_variance(pattern))

        # compute importance weight w for each importance sample using Eq. 7

        self._iweights /= np.mean(self._iweights, axis=1)[:,None]

    def _approximate_mstep(self, target, pattern):
        '''Do the approximate M-step.'''
        # initialize model updates
        dW1 = np.zeros_like(self._W1)
        dW2 = np.zeros_like(self._W2)
        dW3 = np.zeros_like(self._W3)
        dW4 = np.zeros_like(self._W4)
        dW5 = np.zeros_like(self._W5)
        
        dBh1 = np.zeros_like(self._Bh1)
        dBh2 = np.zeros_like(self._Bh2)
        dBh3 = np.zeros_like(self._Bh3)
        dBh4 = np.zeros_like(self._Bh4)
        dBy  = np.zeros_like(self._By)


        # print "self._y[:,m]"
        # print self._y[:,]
        # var = np.zeros((self._M,1))

        for m in range(self._M):
            # Errors split by weights
            # Back prop!
            # var[m] = (np.array(target).T - self.sample(self._y[:,m])) ** 2
            y_err  = np.random.normal(np.array(target).T, self.get_variance(pattern)) - self._y[:,m]
            # y_err  = np.array(target).T - self._y[:,m]  # ??? Use N(,)
            h4_err = np.dot(self._W5.T, y_err)
            h3_err = np.dot(self._W4.T, h4_err)
            h2_err = np.dot(self._W3.T, h3_err)
            h1_err = np.dot(self._W2.T, h2_err)

            # get importance weight w
            w = self._iweights[:,m]
            # Delta Function
            dy  = y_err   * self._f_prime(self._y[:,m])
            dh4 = h4_err  * self._f_prime(self._h4[:,m])
            dh3 = h3_err  * self._f_prime(self._h3[:,m])
            dh2 = h2_err  * self._f_prime(self._h2[:,m])
            dh1 = (h1_err * self._f_prime(self._h1))[:,None]
            
            # importance weights * np.dot(delta, weight matrix transpose)
            dW5 += w * np.dot(dy,  self._h4[:,m][:,None].T)  # ??? Use N(,)
            # dW5 += w * np.random.normal(np.dot(dy,  self._h4[:,m][:,None].T), self._var)  # ??? Use N(,)
            dW4 += w * np.dot(dh4, self._h3[:,m].T)
            dW3 += w * np.dot(dh3, self._h2[:,m].T)
            dW2 += w * np.dot(dh2, self._h1.T)
            dW1 += w * np.dot(dh1, self._x[:,None].T)
            
            # importance wieghts * delta * 1
            dBy  += w * dy
            dBh4 += w * dh4
            dBh3 += w * dh3
            dBh2 += w * dh2
            dBh1 += w * h1_err * self._f_prime(self._h1)


        # perform gradient ascent on Q
        self._W5 += self._lr / self._M * dW5
        self._W4 += self._lr / self._M * dW4
        self._W3 += self._lr / self._M * dW3
        self._W2 += self._lr / self._M * dW2
        self._W1 += self._lr / self._M * dW1
        
        self._By  += self._lr / self._M * dBy
        self._Bh4 += self._lr / self._M * dBh4
        self._Bh3 += self._lr / self._M * dBh3
        self._Bh2 += self._lr / self._M * dBh2
        self._Bh1 += self._lr / self._M * dBh1

    def train(self, epochs):
        '''Use the EM algorithm for learning SFNNs.'''
        for e in range(epochs):
            #self._lr = 0.3 / (1.001 + e)
            log.info('Running epoch %d', e)
            indices = list(range(self._N))
            random.shuffle(indices)
            for i in indices:
                self._approximate_estep(self.X[i])
                self._approximate_mstep(self.Y[i],self.X[i])
            # XXX update \sigma^2
            var = np.zeros((self._N,1))
            for i in range(self._N):
                var[i] = (np.array(self.Y[i]).T - self.sample(self.X[i])) ** 2
            self._var = var.sum() / (self._N - 2)
            # log.debug('sigma^2 = %f', self._var)

    def sample(self, pattern):
        '''Draw an exact sample.'''
        x  = np.array(pattern).T
        h1 = self._f(np.dot(self._W1, x) + self._Bh1)
        h2 = self._f(np.dot(self._W2, h1) + self._Bh2)
        h2[:self._Nh] = np.random.binomial(1, h2[:self._Nh])
        h3 = self._f(np.dot(self._W3, h2) + self._Bh3)
        h3[:self._Nh] = np.random.binomial(1, h3[:self._Nh])
        h4 = self._f(np.dot(self._W4, h3) + self._Bh4)
        y  = self._f(np.dot(self._W5, h4) + self._By)
        # print "np.random.normal(y, self._var)"
        # print np.random.normal(y, self._var)
        return np.random.normal(y, self.get_variance(pattern)) #y


if __name__ == '__main__':
    # command-line interface
    parser = argparse.ArgumentParser(description='Stochastic Feedforward Net')
    parser.add_argument('-e', type=int, help='Epochs', required=True)
    parser.add_argument('-X', default='../data/X.csv', help='/path/to/X.csv')
    parser.add_argument('-Y', default='../data/Y.csv', help='/path/to/Y.csv')
    args = parser.parse_args()

    # create logger
    logging.config.fileConfig('logging.conf')
    log = logging.getLogger(__name__)

    # instantiate network

    sfnn = SFNN(X=args.X, Y=args.Y)

    # train network
    sfnn.train(epochs=args.e)

    # plot Dataset A and exact samples
    patterns = np.linspace(0, 1, num=1000)
    samples = [sfnn.sample(p).item(0) for p in patterns]
    plt.plot(sfnn.X, sfnn.Y, 'b*', patterns, samples, 'r+')
    plt.ylim((-0.2, 1.2))
    plt.show()
