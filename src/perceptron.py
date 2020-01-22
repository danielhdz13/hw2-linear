import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    Data can be transformed before being put into a linear discriminator. If the data
    is not linearly separable, it can be transformed to a space where the data
    is linearly separable, allowing the perceptron algorithm to work on it. This
    function should implement such a transformation on a specific dataset (NOT 
    in general).

    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features after being transformed by the function
    """
    x = np.array([feature[0] for feature in features])
    y = np.array([(feature[1]**2 + feature[0]**2)**.5 for feature in features])
    transformed_features = np.array([[x[index],y[index]] for index in range(len(x))])
    return transformed_features

class Perceptron():
    def __init__(self, max_iterations=200):
        """
        This implements a linear perceptron for classification. A single
        layer perceptron is an algorithm for supervised learning of a binary
        classifier. The idea is to draw a linear line in the space that separates
        the points in the space into two partitions. Points on one side of the 
        line are one class and points on the other side are the other class.
       
        begin initialize weights
            while not converged or not exceeded max_iterations
                for each example in features
                    if example is misclassified using weights
                    then weights = weights + example * label_for_example
            return weights
        end
        
        Note that label_for_example is either -1 or 1.

        Use only numpy to implement this algorithm. 

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged.

        """
        self.max_iterations = max_iterations
        self.weights = None

    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets, which
        are classes (-1 or 1). This function should terminate either after
        convergence (dividing line does not change between interations) or after
        max_iterations (defaults to 200) iterations are done. Here is pseudocode for 
        the perceptron learning algorithm:

        Args:
            features (np.ndarray): 2D array containing inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (saves model and training data internally)
        """
        new_features = np.asarray([np.append(1,feature) for feature in features])
        self.weights = np.array([0,0,0])
        iterate_count = 0
        while iterate_count < self.max_iterations:
            converge = True
            for index in range(len(new_features)):
                if (self.weights.transpose() @ new_features[index]) * targets[index] <= 1:
                    converge = False
                    break
            if converge:
                return
            for index in range(len(new_features)):
                if (self.weights.transpose() @ new_features[index]) * targets[index] <= 1:
                    self.weights = self.weights + np.array([elem * targets[index] * .1 for elem in new_features[index]])
            iterate_count += 1

    def predict(self, features):
        """
        Given features, a 2D numpy array, use the trained model to predict target 
        classes. Call this after calling fit.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features.
        """
        predictions = []
        for index in range(len(features)):
            label = self.weights.transpose() @ np.append([1],features[index])
            predictions.append(-1 if label < 0 else 1)
        solution = np.asarray(predictions)
        return solution

    def visualize(self, features, targets):
        """
        This function should produce a single plot containing a scatter plot of the
        features and the targets, and the perceptron fit by the model should be
        graphed on top of the points.

        DO NOT USE plt.show() IN THIS FUNCTION.

        Args:
            features (np.ndarray): 2D array containing real-valued inputs.
            targets (np.ndarray): 1D array containing binary targets.
        Returns:
            None (plots to the active figure)
        """
        pass
