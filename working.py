import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sse(x, y):
    return np.sum((x - y) ** 2) / 2


class SFNN:
    def __init__(self, n):
        #passed in as 10
        self.n = n
        self.fitted = False

    def hidden_net(self, x):
        self._hidden_net = np.dot(x, self.h)
        return self._hidden_net

    def hidden_output(self, x):
        self._hidden_output = sigmoid(self.hidden_net(x))
        self._hidden_output = np.hstack((self._hidden_output,
                                         np.ones((self._hidden_output.shape[0], 1), dtype=self._hidden_output.dtype)))
        return self._hidden_output


    def net(self, x):
        self._net = np.dot(self.hidden_output(x), self.w)
        return self._net

    def output(self, x):
        net = self.net(x)
        self._output = sigmoid(net)
        return self._output



    def projected_predict(self, x):
        if not self.fitted:
            raise Exception("You should fit it first.")

        #NN output?
        return self.output(x)



    def predict(self, x):
        y = self.projected_predict(self.map_x(x))
        return self.min_max(y, 1, 0, self.max_y, self.min_y)




    @staticmethod
    def min_max(x, d_max, d_min, target_max, target_min):
        return (x - d_min) / (d_max - d_min) * (target_max - target_min) + target_min

    def map_x(self, x):
        ret = self.min_max(x, self.max_x, self.min_x, 1, 0)
        dim_extended = np.hstack((ret, np.ones((ret.shape[0], 1), dtype=ret.dtype)))
        return dim_extended

    def map_y(self, y):
        return self.min_max(y, self.max_y, self.min_y, 1, 0)

    def set_mappings(self, x, y):
        self.max_x = np.max(x, axis=0)
        self.min_x = np.min(x, axis=0)
        self.max_y = np.max(y, axis=0)
        self.min_y = np.min(y, axis=0)







    def learn(self, instance, target, mu, momentum):

        dw = np.empty_like(self.w)
        for i in range(target.shape[1]):
            for j in range(self.n + 1):
                v = (target[0, i] - self._output[0, i]) * sigmoid_d(self._net[0, i]) * self._hidden_output[0, j]
                dw[j, i] = v

        dh = np.empty_like(self.h)
        for i in range(self.n):
            for j in range(instance.shape[1]):
                acc = 0
                for k in range(target.shape[1]):
                    acc += (target[0, k] - self._output[0, k]) * sigmoid_d(self._net[0, k]) * self.w[i, k]
                acc *= sigmoid_d(self._hidden_net[0, i]) * instance[0, j]
                dh[j, i] = acc


        dw = mu * dw + momentum * self.last_dw
        dh = mu * dh + momentum * self.last_dh
        self.w += dw
        self.h += dh
        self.last_dw = dw
        self.last_dh = dh

    def fit(self, x, y, mu=0.1, momentum=0, epochs=1000, error=0):
        # Convert the data to a proper range
        self.set_mappings(x, y)

        x = self.map_x(x)
        y = self.map_y(y)

        # Create Weight matrix

        self.h = np.random.random((x.shape[1], self.n))
        self.w = np.random.random((self.n + 1, y.shape[1]))

        self.last_dw = np.zeros_like(self.w)
        self.last_dh = np.zeros_like(self.h)
        self.fitted = True


        for i in range(epochs):
            epoch_error = 0
            for instance, target in zip(x, y):

                instance = instance.reshape(1, -1)
                target = target.reshape(1, -1)


                output = self.projected_predict(instance)
                error = sse(output, target)
                epoch_error += error



                self.learn(instance, target, mu, momentum)

            if np.isclose(epoch_error, error):
                break


            print('EPOCH FINISHED', epoch_error)


# Xor
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

nn = SFNN(10)

nn.fit(x, y, mu=0.5, momentum=0.3)

print(nn.predict(x))

"""# Linear reg

x = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([[3], [5], [7], [9], [11], [13]])

nn.fit(x, y, mu=0.8)

print(nn.predict(x))"""
