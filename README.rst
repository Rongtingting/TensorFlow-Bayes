Tutorial and learning for automated Variational Bayes
=====================================================

In the repository, we implemeted a few common Bayesian models with TensorFlow 
and TensorFlow Probability, most with variational inference. We also aim to 
provide detailed examples_ on these implemented models.

.. _examples: https://github.com/huangyh09/TensorFlow-Bayes/blob/master/examples


Settings
--------

We recommend using conda_ to set a separate environment for installing all 
necessary dependent packages, which you could use the following commond lines to
create an environment `TFProb` for the dependent packages:

.. code-block:: bash
    
   conda create -n TFProb python=3.7 scipy numpy matplotlib scikit-learn tensorflow=2.0.0 
    
Note, tensorflow-probability v0.8.0 is not available on conda yet, you could 
install it from PyPI:

.. code-block:: bash

   conda activate TFProb
   pip install tensorflow-probability==0.8.0

**Update**: tensorflow-probability v0.8.0 is not available on conda
.. code-block:: bash
    conda install -c anaconda tensorflow-probability
    
https://anaconda.org/anaconda/tensorflow-probability

Also, you could add the newly created environment `TFProb` into `IPython kernel
<https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments>`_.
(make sure you have `ipykernel`).

.. code-block:: bash

   conda activate TFProb
   conda install ipykernel
   python -m ipykernel install --user --name TFProb --display-name "TFProb"

To activate the created environment, use ``conda activate TFProb``

.. _conda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


Notes
-----

* MacBook fails to work with both tf.tensordot and np.tensordot (or np.dot),
  while Linux works fine. See the reported `issue 
  <https://github.com/tensorflow/tensorflow/issues/34553>`_.

* Many of the functions should be implemeted with PyTorch too, which is under 
  testing.
