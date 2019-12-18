
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class BayesianLinearRegression():
    def __init__(self, d, name=None):
        self.w_loc = tf.Variable(tf.random.normal([d, 1]), name='w_loc')
        self.w_std = tf.Variable(tf.random.normal([d, 1]), name='w_std')
        self.b_loc = tf.Variable(tf.random.normal([1]), name='b_loc')
        self.b_std = tf.Variable(tf.random.normal([1]), name='b_std')
        self.s_alpha = tf.Variable(tf.exp(tf.random.normal([1])), name='s_alpha')
        self.s_beta = tf.Variable(tf.exp(tf.random.normal([1])), name='s_beta')
    
    @property
    def weight(self):
        """Variational posterior for the weight"""
        return tfd.Normal(self.w_loc, tf.exp(self.w_std))
    
    @property
    def bias(self):
        """Variational posterior for the bias"""
        return tfd.Normal(self.b_loc, tf.exp(self.b_std))

    @property
    def std(self):
        """Variational posterior for the noise standard deviation"""
        return tfd.InverseGamma(tf.exp(self.s_alpha), tf.exp(self.s_beta))
    
    @property
    def losses(self):
        """Sum of KL divergences between posteriors and priors"""
        prior = tfd.Normal(0, 1)
        return (tf.reduce_sum(tfd.kl_divergence(self.weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(self.bias, prior)))
    
    def predict(self, x, sampling=False):
        """Predict p(y|x), based on current parameters"""
        sample = lambda x: x.sample() if sampling else x.mean()
        loc = tf.matmul(x, sample(self.weight)) + sample(self.bias)
        std = tf.sqrt(sample(self.std))
        return tfd.Normal(loc, std)
    
    def fit(self, xtrain, ytrain, num_steps=100, optimizer=None,
            learn_rate=0.05):
        """Fit the model's parameters"""
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=learn_rate)
            
        loss_fn = lambda: (self.losses - 
                           tf.reduce_sum(self.predict(xtrain).log_prob(ytrain)))
        
        losses = tfp.math.minimize(loss_fn, 
                                   num_steps=num_steps, 
                                   optimizer=optimizer)
        return losses
