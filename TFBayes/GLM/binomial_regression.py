
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

class BinomRegression():
    """
    d: number of theta categories
    K: number of components
    """
    def __init__(self, K, d=3, name=None):
        self.K = K
        self.d = d
        self.w_size = tf.Variable(tf.random.uniform([K], 0.3), name='w_size')
        self.beta_size = tf.Variable(tf.random.uniform([d, 2], 0.3), name='beta_size')
    
    @property
    def weight(self):
        """Variational posterior for the weight"""
        return tfd.Dirichlet(tf.math.exp(self.w_size))

    @property
    def ASR(self):
        """Variational posterior for the binomial rate"""
        return tfd.Beta(tf.math.exp(self.beta_size[:,0]), 
                        tf.math.exp(self.beta_size[:,1]))
    
    @property
    def losses(self):
        """Sum of KL divergences between posteriors and priors"""
        w_prior = tfd.Dirichlet(tf.ones([self.K]))
        theta_prior = tfd.Beta([0.1, 3, 9.9], [9.9, 3, 0.1])
        
        return (tf.reduce_sum(tfd.kl_divergence(self.weight, w_prior)) +
                tf.reduce_sum(tfd.kl_divergence(self.ASR, theta_prior)))
    
    def logLik(self, A, D, GT, sampling=False):
        """binomial coefficient needs to be added"""
        sample = lambda x: x.sample() if sampling else x.mean()
        _theta = tf.tensordot(tf.tensordot(GT, sample(self.ASR), axes=[[2], [0]]),
                              sample(self.weight), axes=[[1], [0]])
        
        _logLik = A * tf.math.log(_theta) + (D - A) * tf.math.log(1 - _theta)
        return _logLik
    
    def fit(self, A, D, GT, num_steps=100, optimizer=None,
            learn_rate=0.05):
        """Fit the model's parameters"""
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=learn_rate)
            
        loss_fn = lambda: (self.losses - tf.reduce_sum(self.logLik(A, D, GT)))
        
        losses = tfp.math.minimize(loss_fn, 
                                   num_steps=num_steps, 
                                   optimizer=optimizer)
        return losses
