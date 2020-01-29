
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

class GaussianMixture():
    """A Bayesian Gaussian mixture model.
    Assumes Gaussians' variances in each dimension are independent.
    
    Parameters
    ----------
    Nc : int > 0
        Number of mixture components.
    Nd : int > 0
        Number of dimensions.
    Ns : int > 0
        Number of data points.
    """
    def __init__(self, Nc, Nd, Ns=0):
        # Initialize
        self.Nc = Nc
        self.Nd = Nd
        self.Ns = Ns
        
        # Variational distribution variables for means
        self.locs = tf.Variable(tf.random.normal((Nc, Nd)))
        self.scales = tf.Variable(tf.pow(tf.random.gamma((Nc, Nd), 5, 5), -0.5))
        
        # Variational distribution variables for standard deviations
        self.alpha = tf.Variable(tf.random.uniform((Nc, Nd), 4., 6.))
        self.beta = tf.Variable(tf.random.uniform((Nc, Nd), 4., 6.))
        
        # Variational distribution variables for assignment logit
        self.gamma = tf.Variable(tf.random.uniform((Ns, Nc), -2, 2))
        
        self.set_prior()
        
    def set_prior(self, mu_prior=None, sigma_prior=None, ident_prior=None):
        """Set prior ditributions
        """
        # Prior distributions for the means
        if mu_prior is None:
            self.mu_prior = tfd.Normal(tf.zeros((self.Nc, self.Nd)), 
                                       tf.ones((self.Nc, self.Nd)))
        else:
            self.mu_prior = self.mu_prior

        # Prior distributions for the standard deviations
        if sigma_prior is None:
            self.sigma_prior = tfd.Gamma(2 * tf.ones((self.Nc, self.Nd)), 
                                         2 * tf.ones((self.Nc, self.Nd)))
        else:
            self.sigma_prior = sigma_prior
        
        # Prior distributions for sample assignment
        if ident_prior is None:
            self.ident_prior = tfd.Multinomial(total_count=1,
                            probs=tf.ones((self.Ns, self.Nc))/self.Nc)
        else:
            self.ident_prior = ident_prior
    
    @property
    def mu(self):
        """Variational posterior for distribution mean"""
        return tfd.Normal(self.locs, self.scales)
    
    @property
    def sigma(self):
        """Variational posterior for distribution variance"""
        return tfd.Gamma(self.alpha, self.beta)
        # return tfd.Gamma(tf.math.exp(self.alpha), tf.math.exp(self.beta))
    
    @property
    def ident(self):
        return tfd.Multinomial(total_count=1, 
                               probs=tf.math.softmax(self.gamma))

    
    @property
    def KLsum(self):
        """
        Sum of KL divergences between posteriors and priors
        The KL divergence for multinomial distribution is defined manually
        """
        kl_mu    = tf.reduce_sum(tfd.kl_divergence(self.mu,    self.mu_prior))
        kl_sigma = tf.reduce_sum(tfd.kl_divergence(self.sigma, self.sigma_prior))
        kl_ident = tf.reduce_sum(self.ident.mean() * 
                                 tf.math.log(self.ident.mean() / 
                                             self.ident_prior.mean())) # axis=0
        
        return kl_mu + kl_sigma + kl_ident
    
        
    def logLik(self, x, sampling=False, n_sample=10, use_ident=True):
        """Compute log likelihood given a batch of data.
        
        Parameters
        ----------
        x : tf.Tensor, (n_sample, n_dimention)
            A batch of data
        sampling : bool
            Whether to sample from the variational posterior
            distributions (if True, the default), or just use the
            mean of the variational distributions (if False).
        n_sample : int
            The number of samples to generate
        use_ident : bool
            Setting True for fitting the model and False for testing logLik
            
        Returns
        -------
        log_likelihoods : tf.Tensor
            Log likelihood for each sample
        """
        #TODO: sampling doesn't converge well in the example data set

        Nb, Nd = x.shape
        x = tf.reshape(x, (1, Nb, 1, Nd)) # (n_sample, Ns, Nc, Nd)

        # Sample from the variational distributions
        if sampling:
            _mu = self.mu.sample((n_sample, 1))
            _sigma = tf.pow(self.sigma.sample((n_sample, 1)), -0.5)
        else:
            _mu = tf.reshape(self.mu.mean(), (1, 1, self.Nc, self.Nd))
            _sigma = tf.pow(tf.reshape(self.sigma.mean(), 
                                       (1, 1, self.Nc, self.Nd)), -0.5)
        
        # Calculate the probability density
        _model = tfd.Normal(_mu, _sigma)
        _log_lik_mix = _model.log_prob(x)
        
        if use_ident:
            _ident = tf.reshape(self.ident.mean(), (1, self.Ns, self.Nc, 1))
            _log_lik_mix = _log_lik_mix * _ident
            log_likelihoods = tf.reduce_sum(_log_lik_mix, axis=[0, 2, 3])
        else:
            _fract = tf.reshape(tf.reduce_mean(self.ident.mean(), axis=0),
                                (1, 1, self.Nc, 1))
            _log_lik_mix = _log_lik_mix + tf.math.log(_fract)
            log_likelihoods = tf.reduce_mean(tf.math.reduce_logsumexp(
                tf.reduce_sum(_log_lik_mix, axis=3), axis=2), axis=0)
        
        return log_likelihoods
    
    def fit(self, x, num_steps=200, 
            optimizer=None, learn_rate=0.05, **kwargs):
        """Fit the model's parameters"""
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=learn_rate)
            
        loss_fn = lambda: (self.KLsum - 
            tf.reduce_sum(self.logLik(x, **kwargs)))
        
        losses = tfp.math.minimize(loss_fn, 
                                   num_steps=num_steps, 
                                   optimizer=optimizer)
        return losses
