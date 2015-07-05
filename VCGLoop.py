################################################################################
# Code for managing and training a Variational Collaborative Generative Loop.  #
################################################################################

# basic python
import numpy as np
import numpy.random as npr
from collections import OrderedDict

# theano business
import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams as RandStream
from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandStream

# phil's sweetness
from NetLayers import HiddenLayer, DiscLayer, softplus_actfun, \
                      apply_mask
from LogPDFs import log_prob_bernoulli, log_prob_gaussian2, gaussian_kld
from HelperFuncs import get_param_updates
from OneStageModel import OneStageModel

#############################
# SOME HANDY LOSS FUNCTIONS #
#############################

def logreg_loss(Y, class_sign):
    """
    Simple binomial deviance (i.e. logistic regression) loss.

    This assumes that all predictions in Y have the same target class, which
    is indicated by class_sign, which should be in {-1, +1}. Note: this does
    not "normalize" for the number of predictions in Y.
    """
    loss = T.sum(softplus_actfun(-class_sign * Y))
    return loss

def lsq_loss(Yh, Yt=0.0):
    """
    Least-squares loss for predictions in Yh, given target Yt.
    """
    loss = T.sum((Yh - Yt)**2.0)
    return loss

def hinge_loss(Yh, Yt=0.0):
    """
    Unilateral hinge loss for Yh, given target Yt.
    """
    residual = Yt - Yh
    loss = T.sum((residual * (residual > 0.0)))
    return loss

def hinge_sq_loss(Yh, Yt=0.0):
    """
    Unilateral squared-hinge loss for Yh, given target Yt.
    """
    residual = Yt - Yh
    loss = T.sum((residual * (residual > 0.0))**2.0)
    return loss

class VCGLoop(object):
    """
    Controller for training a self-looping VAE using guidance provided by a
    classifier. The classifier tries to discriminate between samples generated
    by the looped VAE while the VAE minimizes a variational generative model
    objective and also shifts mass away from regions where the classifier can
    discern that the generated data is denser than the training data.

    This class can also train "policies" for reconstructing partially masked
    inputs. A reconstruction policy can readily be trained to share the same
    parameters as a policy for generating transitions while self-looping.

    The generator must be an instance of the InfNet class implemented in
    "InfNet.py". The discriminator must be an instance of the PeaNet class,
    as implemented in "PeaNet.py". The inferencer must be an instance of the
    InfNet class implemented in "InfNet.py".

    Parameters:
        rng: numpy.random.RandomState (for reproducibility)
        Xd: symbolic var for providing points for starting the Markov Chain
        Xc: symbolic var for providing points for starting the Markov Chain
        Xm: symbolic var for providing masks to mix Xd with Xc
        Xt: symbolic var for providing samples from the target distribution
        i_net: The InfNet instance that will serve as the inferencer
        g_net: The InfNet instance that will serve as the generator
        d_net: The PeaNet instance that will serve as the discriminator
        chain_len: number of steps to unroll the VAE Markov Chain
        data_dim: dimension of the generated data
        prior_dim: dimension of the model prior
        params: a dict of parameters for controlling various costs
            x_type: can be "bernoulli" or "gaussian"
            xt_transform: optional transform for gaussian means
            logvar_bound: optional bound on gaussian output logvar
            cost_decay: rate of decay for VAE costs in unrolled chain
            chain_type: can be 'walkout' or 'walkback'
            lam_l2d: regularization on squared discriminator output
    """
    def __init__(self, rng=None, Xd=None, Xc=None, Xm=None, Xt=None, \
                 i_net=None, g_net=None, d_net=None, chain_len=None, \
                 data_dim=None, prior_dim=None, params=None):
        # Do some stuff!
        self.rng = RandStream(rng.randint(100000))
        self.data_dim = data_dim
        self.prior_dim = prior_dim
        self.prior_mean = 0.0
        self.prior_logvar = 0.0
        if params is None:
            self.params = {}
        else:
            self.params = params
        if 'cost_decay' in self.params:
            self.cost_decay = self.params['cost_decay']
        else:
            self.cost_decay = 0.1
        if 'chain_type' in self.params:
            assert((self.params['chain_type'] == 'walkback') or \
                (self.params['chain_type'] == 'walkout'))
            self.chain_type = self.params['chain_type']
        else:
            self.chain_type = 'walkout'
        if 'xt_transform' in self.params:
            assert((self.params['xt_transform'] == 'sigmoid') or \
                    (self.params['xt_transform'] == 'none'))
            if self.params['xt_transform'] == 'sigmoid':
                self.xt_transform = lambda x: T.nnet.sigmoid(x)
            else:
                self.xt_transform = lambda x: x
        else:
            self.xt_transform = lambda x: T.nnet.sigmoid(x)
        if 'logvar_bound' in self.params:
            self.logvar_bound = self.params['logvar_bound']
        else:
            self.logvar_bound = 10
        #
        # x_type: this tells if we're using bernoulli or gaussian model for
        #         the observations
        #
        self.x_type = self.params['x_type']
        assert((self.x_type == 'bernoulli') or (self.x_type == 'gaussian'))

        # symbolic var for inputting samples for initializing the VAE chain
        self.Xd = Xd
        # symbolic var for masking subsets of the state variables
        self.Xm = Xm
        # symbolic var for controlling subsets of the state variables
        self.Xc = Xc
        # symbolic var for inputting samples from the target distribution
        self.Xt = Xt
        # integer number of times to cycle the VAE loop
        self.chain_len = chain_len

        # symbolic matrix of indices for data inputs
        self.It = T.arange(self.Xt.shape[0])
        # symbolic matrix of indices for noise/generated inputs
        self.Id = T.arange(self.chain_len * self.Xd.shape[0]) + self.Xt.shape[0]

        # get a clone of the desired VAE, for easy access
        self.OSM = OneStageModel(rng=rng, Xd=self.Xd, Xc=self.Xc, Xm=self.Xm, \
                p_x_given_z=g_net, q_z_given_x=i_net, x_dim=self.data_dim, \
                z_dim=self.prior_dim, params=self.params)
        self.IN = self.OSM.q_z_given_x
        self.GN = self.OSM.p_x_given_z
        self.transform_x_to_z = self.OSM.transform_x_to_z
        self.transform_z_to_x = self.OSM.transform_z_to_x
        self.bounded_logvar = self.OSM.bounded_logvar
        # self-loop some clones of the main VAE into a chain.
        # ** All VAEs in the chain share the same Xc and Xm, which are the
        #    symbolic inputs for providing the observed portion of the input
        #    and a mask indicating which part of the input is "observed".
        #    These inputs are used for training "reconstruction" policies.
        self.IN_chain = []
        self.GN_chain = []
        self.Xg_chain = []
        _Xd = self.Xd
        print("Unrolling chain...")
        for i in range(self.chain_len):
            # create a VAE infer/generate pair with _Xd as input and with
            # masking variables shared by all VAEs in this chain
            _IN = self.IN.shared_param_clone(rng=rng, \
                    Xd=apply_mask(Xd=_Xd, Xc=self.Xc, Xm=self.Xm), \
                    build_funcs=False)
            _GN = self.GN.shared_param_clone(rng=rng, Xd=_IN.output, \
                    build_funcs=False)
            _Xd = self.xt_transform(_GN.output_mean)
            self.IN_chain.append(_IN)
            self.GN_chain.append(_GN)
            self.Xg_chain.append(_Xd)
            print("    step {}...".format(i))

        # make a clone of the desired discriminator network, which will try
        # to discriminate between samples from the training data and samples
        # generated by the self-looped VAE chain.
        self.DN = d_net.shared_param_clone(rng=rng, \
                Xd=T.vertical_stack(self.Xt, *self.Xg_chain))

        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        # init shared var for weighting nll of data given posterior sample
        self.lam_chain_nll = theano.shared(value=zero_ary, name='vcg_lam_chain_nll')
        self.set_lam_chain_nll(lam_chain_nll=1.0)
        # init shared var for weighting posterior KL-div from prior
        self.lam_chain_kld = theano.shared(value=zero_ary, name='vcg_lam_chain_kld')
        self.set_lam_chain_kld(lam_chain_kld=1.0)
        # init shared var for controlling l2 regularization on params
        self.lam_l2w = theano.shared(value=zero_ary, name='vcg_lam_l2w')
        self.set_lam_l2w(lam_l2w=1e-4)
        # shared var learning rates for all networks
        self.lr_dn = theano.shared(value=zero_ary, name='vcg_lr_dn')
        self.lr_gn = theano.shared(value=zero_ary, name='vcg_lr_gn')
        self.lr_in = theano.shared(value=zero_ary, name='vcg_lr_in')
        # shared var momentum parameters for all networks
        self.mom_1 = theano.shared(value=zero_ary, name='vcg_mom_1')
        self.mom_2 = theano.shared(value=zero_ary, name='vcg_mom_2')
        self.it_count = theano.shared(value=zero_ary, name='vcg_it_count')
        # shared var weights for adversarial classification objective
        self.dw_dn = theano.shared(value=zero_ary, name='vcg_dw_dn')
        self.dw_gn = theano.shared(value=zero_ary, name='vcg_dw_gn')
        # init parameters for controlling learning dynamics
        self.set_all_sgd_params()
        
        self.set_disc_weights()  # init adversarial cost weights for GN/DN
        # set a shared var for regularizing the output of the discriminator
        self.lam_l2d = theano.shared(value=(zero_ary + params['lam_l2d']), \
                name='vcg_lam_l2d')

        # Grab the full set of "optimizable" parameters from the generator
        # and discriminator networks that we'll be working with. We need to
        # ignore parameters in the final layers of the proto-networks in the
        # discriminator network (a generalized pseudo-ensemble). We ignore them
        # because the VCGair requires that they be "bypassed" in favor of some
        # binary classification layers that will be managed by this VCGair.
        self.dn_params = []
        for pn in self.DN.proto_nets:
            for pnl in pn[0:-1]:
                self.dn_params.extend(pnl.params)
        self.in_params = [p for p in self.IN.mlp_params]
        self.in_params.append(self.OSM.output_logvar)
        self.gn_params = [p for p in self.GN.mlp_params]
        self.joint_params = self.in_params + self.gn_params + self.dn_params

        # Now construct a binary discriminator layer for each proto-net in the
        # discriminator network. And, add their params to optimization list.
        self._construct_disc_layers(rng)
        self.disc_reg_cost = self.lam_l2d[0] * \
                T.sum([dl.act_l2_sum for dl in self.disc_layers])

        # Construct costs for the generator and discriminator networks based 
        # on adversarial binary classification
        self.disc_cost_dn, self.disc_cost_gn = self._construct_disc_costs()

        # first, build the cost to be optimized by the discriminator network,
        # in general this will be treated somewhat indepedently of the
        # optimization of the generator and inferencer networks.
        self.dn_cost = self.disc_cost_dn + self.DN.act_reg_cost + \
                self.disc_reg_cost

        # construct costs relevant to the optimization of the generator and
        # discriminator networks
        self.chain_nll_cost = self.lam_chain_nll[0] * \
                self._construct_chain_nll_cost(cost_decay=self.cost_decay)
        self.chain_kld_cost = self.lam_chain_kld[0] * \
                self._construct_chain_kld_cost(cost_decay=self.cost_decay)
        self.other_reg_cost = self._construct_other_reg_cost()
        self.osm_cost = self.disc_cost_gn + self.chain_nll_cost + \
                self.chain_kld_cost + self.other_reg_cost
        # compute total cost on the discriminator and VB generator/inferencer
        self.joint_cost = self.dn_cost + self.osm_cost

        # Get the gradient of the joint cost for all optimizable parameters
        self.joint_grads = OrderedDict()
        print("Computing VCGLoop DN cost gradients...")
        grad_list = T.grad(self.dn_cost, self.dn_params, disconnected_inputs='warn')
        for i, p in enumerate(self.dn_params):
            self.joint_grads[p] = grad_list[i]
        print("Computing VCGLoop IN cost gradients...")
        grad_list = T.grad(self.osm_cost, self.in_params, disconnected_inputs='warn')
        for i, p in enumerate(self.in_params):
            self.joint_grads[p] = grad_list[i]
        print("Computing VCGLoop GN cost gradients...")
        grad_list = T.grad(self.osm_cost, self.gn_params, disconnected_inputs='warn')
        for i, p in enumerate(self.gn_params):
            self.joint_grads[p] = grad_list[i]

        # construct the updates for the discriminator, generator and 
        # inferencer networks. all networks share the same first/second
        # moment momentum and iteration count. the networks each have their
        # own learning rates, which lets you turn their learning on/off.
        self.dn_updates = get_param_updates(params=self.dn_params, \
                grads=self.joint_grads, alpha=self.lr_dn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8, max_grad_norm=10.0)
        self.gn_updates = get_param_updates(params=self.gn_params, \
                grads=self.joint_grads, alpha=self.lr_gn, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8, max_grad_norm=10.0)
        self.in_updates = get_param_updates(params=self.in_params, \
                grads=self.joint_grads, alpha=self.lr_in, \
                beta1=self.mom_1, beta2=self.mom_2, it_count=self.it_count, \
                mom2_init=1e-3, smoothing=1e-8, max_grad_norm=10.0)

        # bag up all the updates required for training
        self.joint_updates = OrderedDict()
        for k in self.dn_updates:
            self.joint_updates[k] = self.dn_updates[k]
        for k in self.gn_updates:
            self.joint_updates[k] = self.gn_updates[k]
        for k in self.in_updates:
            self.joint_updates[k] = self.in_updates[k]
        # construct an update for tracking the mean KL divergence of
        # approximate posteriors for this chain
        new_kld_mean = (0.98 * self.IN.kld_mean) + ((0.02 / self.chain_len) * \
            sum([T.mean(I_N.kld_cost) for I_N in self.IN_chain]))
        self.joint_updates[self.IN.kld_mean] = T.cast(new_kld_mean, 'floatX')

        # construct the function for training on training data
        print("Compiling VCGLoop theano functions....")
        self.train_joint = self._construct_train_joint()
        return

    def set_dn_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the discriminator network.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_dn.set_value(new_lr.astype(theano.config.floatX))
        return

    def set_in_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the inferencer network.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        return

    def set_gn_sgd_params(self, learn_rate=0.01):
        """
        Set learning rate for the generator network.
        """
        zero_ary = np.zeros((1,))
        new_lr = zero_ary + learn_rate
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        return

    def set_all_sgd_params(self, learn_rate=0.01, mom_1=0.9, mom_2=0.999):
        """
        Set learning rate and momentum parameter for all updates.
        """
        zero_ary = np.zeros((1,))
        # set learning rates to the same value
        new_lr = zero_ary + learn_rate
        self.lr_dn.set_value(new_lr.astype(theano.config.floatX))
        self.lr_gn.set_value(new_lr.astype(theano.config.floatX))
        self.lr_in.set_value(new_lr.astype(theano.config.floatX))
        # set the first/second moment momentum parameters
        new_mom_1 = zero_ary + mom_1
        new_mom_2 = zero_ary + mom_2
        self.mom_1.set_value(new_mom_1.astype(theano.config.floatX))
        self.mom_2.set_value(new_mom_2.astype(theano.config.floatX))
        return

    def set_disc_weights(self, dweight_gn=1.0, dweight_dn=1.0):
        """
        Set weights for the adversarial classification cost.
        """
        zero_ary = np.zeros((1,)).astype(theano.config.floatX)
        new_dw_dn = zero_ary + dweight_dn
        self.dw_dn.set_value(new_dw_dn)
        new_dw_gn = zero_ary + dweight_gn
        self.dw_gn.set_value(new_dw_gn)
        return

    def set_lam_chain_nll(self, lam_chain_nll=1.0):
        """
        Set weight for controlling the influence of the data likelihood.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_chain_nll
        self.lam_chain_nll.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_chain_kld(self, lam_chain_kld=1.0):
        """
        Set the strength of regularization on KL-divergence for continuous
        posterior variables. When set to 1.0, this reproduces the standard
        role of KL(posterior || prior) in variational learning.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_chain_kld
        self.lam_chain_kld.set_value(new_lam.astype(theano.config.floatX))
        return

    def set_lam_l2w(self, lam_l2w=1e-3):
        """
        Set the relative strength of l2 regularization on network params.
        """
        zero_ary = np.zeros((1,))
        new_lam = zero_ary + lam_l2w
        self.lam_l2w.set_value(new_lam.astype(theano.config.floatX))
        return

    def _construct_disc_layers(self, rng):
        """
        Construct binary discrimination layers for each spawn-net in the
        underlying discrimnator pseudo-ensemble. All spawn-nets spawned from
        the same proto-net will use the same disc-layer parameters.
        """
        self.disc_layers = []
        self.disc_outputs = []
        dn_init_scale = self.DN.init_scale
        for sn in self.DN.spawn_nets:
            # construct a "binary discriminator" layer to sit on top of each
            # spawn net in the discriminator pseudo-ensemble
            sn_fl = sn[-1]
            init_scale = dn_init_scale * (1. / np.sqrt(sn_fl.in_dim))
            self.disc_layers.append(DiscLayer(rng=rng, \
                    input=sn_fl.noisy_input, in_dim=sn_fl.in_dim, \
                    W_scale=dn_init_scale))
            # capture the (linear) output of the DiscLayer, for possible reuse
            self.disc_outputs.append(self.disc_layers[-1].linear_output)
            # get the params of this DiscLayer, for convenient optimization
            self.dn_params.extend(self.disc_layers[-1].params)
        return

    def _construct_disc_costs(self):
        """
        Construct the generator and discriminator adversarial costs.
        """
        gn_costs = []
        dn_costs = []
        for dl_output in self.disc_outputs:
            data_preds = dl_output.take(self.It, axis=0)
            noise_preds = dl_output.take(self.Id, axis=0)
            # compute the cost with respect to which we will be optimizing
            # the parameters of the discriminator network
            data_size = T.cast(self.It.size, 'floatX')
            noise_size = T.cast(self.Id.size, 'floatX')
            dnl_dn_cost = (logreg_loss(data_preds, 1.0) / data_size) + \
                          (logreg_loss(noise_preds, -1.0) / noise_size)
            # compute the cost with respect to which we will be optimizing
            # the parameters of the generative model
            dnl_gn_cost = (hinge_loss(noise_preds, 0.0) + hinge_sq_loss(noise_preds, 0.0)) / (2.0 * noise_size)
            dn_costs.append(dnl_dn_cost)
            gn_costs.append(dnl_gn_cost)
        dn_cost = self.dw_dn[0] * T.sum(dn_costs)
        gn_cost = self.dw_gn[0] * T.sum(gn_costs)
        return [dn_cost, gn_cost]

    def _log_prob_wrapper(self, x_true, x_apprx):
        """
        Wrap log-prob with switching for bernoulli/gaussian output types.
        """
        if self.x_type == 'bernoulli':
            ll_cost = log_prob_bernoulli(x_true, x_apprx)
        else:
            ll_cost = log_prob_gaussian2(x_true, x_apprx, \
                    log_vars=self.bounded_logvar)
        nll_cost = -ll_cost
        return nll_cost

    def _construct_chain_nll_cost(self, cost_decay=0.1):
        """
        Construct the negative log-likelihood part of cost to minimize.

        This is for operation in "free chain" mode, where a seed point is used
        to initialize a long(ish) running markov chain.
        """
        assert((cost_decay >= 0.0) and (cost_decay <= 1.0))
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        nll_costs = []
        step_weight = 1.0
        step_weights = []
        step_decay = cost_decay
        for i in range(self.chain_len):
            if self.chain_type == 'walkback':
                # train with walkback roll-outs -- reconstruct initial point
                IN_i = self.IN_chain[0]
            else:
                # train with walkout roll-outs -- reconstruct previous point
                IN_i = self.IN_chain[i]
            x_true = IN_i.Xd
            x_apprx = self.Xg_chain[i]
            c = T.mean(self._log_prob_wrapper(x_true, x_apprx))
            nll_costs.append(step_weight * c)
            step_weights.append(step_weight)
            step_weight = step_weight * step_decay
        nll_cost = sum(nll_costs) / sum(step_weights)
        return nll_cost

    def _construct_chain_kld_cost(self, cost_decay=0.1):
        """
        Construct the posterior KL-d from prior part of cost to minimize.

        This is for operation in "free chain" mode, where a seed point is used
        to initialize a long(ish) running markov chain.
        """
        assert((cost_decay >= 0.0) and (cost_decay <= 1.0))
        obs_count = T.cast(self.Xd.shape[0], 'floatX')
        kld_mean = self.IN.kld_mean[0]
        kld_costs = []
        step_weight = 1.0
        step_weights = []
        step_decay = cost_decay
        for i in range(self.chain_len):
            IN_i = self.IN_chain[i]
            # basic variational term on KL divergence between post and prior
            kld_i = gaussian_kld(IN_i.output_mean, IN_i.output_logvar, \
                    self.prior_mean, self.prior_logvar)
            kld_i_costs = T.sum(kld_i, axis=1)
            # sum and reweight the KLd cost for this step in the chain
            c = T.mean(kld_i_costs)
            kld_costs.append(step_weight * c)
            step_weights.append(step_weight)
            step_weight = step_weight * step_decay
        kld_cost = sum(kld_costs) / sum(step_weights)
        return kld_cost

    def _construct_other_reg_cost(self):
        """
        Construct the cost for low-level basic regularization. E.g. for
        applying l2 regularization to the network parameters.
        """
        gp_cost = sum([T.sum(par**2.0) for par in self.gn_params])
        ip_cost = sum([T.sum(par**2.0) for par in self.in_params])
        other_reg_cost = self.lam_l2w[0] * (gp_cost + ip_cost)
        return other_reg_cost

    def _construct_train_joint(self):
        """
        Construct theano function to train generator and discriminator jointly.
        """
        # symbolic vars for passing input to training function
        xd = T.matrix()
        xc = T.matrix()
        xm = T.matrix()
        xt = T.matrix()
        batch_reps = T.lscalar()
        # collect outputs to return to caller
        outputs = [self.joint_cost, self.chain_nll_cost, self.chain_kld_cost, \
                self.chain_nll_cost, self.chain_kld_cost, self.disc_cost_gn, \
                self.disc_cost_dn, self.other_reg_cost]
        func = theano.function(inputs=[ xd, xc, xm, xt, batch_reps ], \
                outputs=outputs, updates=self.joint_updates, \
                givens={ self.Xd: xd.repeat(batch_reps, axis=0), \
                         self.Xc: xc.repeat(batch_reps, axis=0), \
                         self.Xm: xm.repeat(batch_reps, axis=0), \
                         self.Xt: xt })
        return func

    def sample_from_chain(self, X_d, X_c=None, X_m=None, loop_iters=5, \
            sigma_scale=None):
        """
        Sample for several rounds through the I<->G loop, initialized with the
        the "data variable" samples in X_d.
        """
        result = self.OSM.sample_from_chain(X_d, X_c=X_c, X_m=X_m, \
                loop_iters=loop_iters, sigma_scale=sigma_scale)
        return result

    def sample_from_prior(self, samp_count):
        """
        Draw independent samples from the model's prior, using the gaussian
        continuous prior of the underlying GenNet.
        """
        Xs = self.OSM.sample_from_prior(samp_count)
        return Xs


if __name__=="__main__":
    print("Hello world!")





##############
# EYE BUFFER #
##############
