import random
import numpy as np
import tensorflow as tf

class PolicyGradientREINFORCE(object):

  def __init__(self, session,
                     optimizer,
                     policy_network,
                     state_dim,
                     num_actions,
                     rnn_hidden_dim,
                     init_exp=0.5,         # initial exploration prob
                     final_exp=0.0,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=0.99, # discount future rewards
                     reg_param=0.001,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_writer=None,
                     summary_every=100,
                     is_new_response=False):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer

    # model components
    self.policy_network = policy_network
    self.rnn_hidden_dim = rnn_hidden_dim

    # training parameters
    self.state_dim       = state_dim
    self.num_actions     = num_actions
    self.discount_factor = discount_factor
    self.max_gradient    = max_gradient
    self.reg_param       = reg_param

    # exploration parameters
    self.exploration  = init_exp
    self.init_exp     = init_exp
    self.final_exp    = final_exp
    self.anneal_steps = anneal_steps

    # counters
    self.train_iteration = 0
    self.sl_train_iteration = 0
    self.is_new_response = is_new_response

    # rollout buffer
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []
    self.rnn_state_buffer = []
    self.bit_vec_buffer = []
    self.user_utt_buffer = []

    # sl rollout buffer
    self.sl_real_action_buffer = []

    # record reward history for normalization
    self.all_rewards = []
    self.max_reward_length = 1000000

    # create and initialize variables
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      print("add_graph")
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = summary_every

  def resetModel(self):
    self.cleanUp()
    self.train_iteration = 0
    self.sl_train_iteration = 0
    self.exploration     = self.init_exp
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

  def create_variables(self):

    with tf.name_scope("model_inputs"):
      # raw state representation
      self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
      self.initial_rnn_state = tf.contrib.rnn.LSTMStateTuple(tf.placeholder(tf.float32, [1, self.rnn_hidden_dim]),
                                                          tf.placeholder(tf.float32, [1, self.rnn_hidden_dim]))
      self.bit_vec = tf.placeholder(tf.float32, shape=(self.num_actions), name="bit_vec_placeholder")
      self.user_utt = tf.placeholder(tf.int32, shape=(None,), name="user_utt_token_placeholder")

    # rollout action based on current policy
    with tf.name_scope("predict_actions"):
      # initialize policy network
      with tf.variable_scope("policy_network"):
        self.policy_outputs, self.prev_rnn_state = self.policy_network(self.states, self.initial_rnn_state, self.bit_vec,
                                                                       self.user_utt)

      # predict actions from policy network
      self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
      # Note 1: tf.multinomial is not good enough to use yet
      # so we don't use self.predicted_actions for now
      self.predicted_actions = tf.multinomial(self.action_scores, 1)

    # regularization loss
    policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

    # compute loss and gradients
    with tf.name_scope("compute_pg_gradients"):
      # gradients for selecting action from policy network
      self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
      self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

      with tf.variable_scope("policy_network", reuse=True):
        self.logprobs, self.prev_rnn_state = self.policy_network(self.states, self.initial_rnn_state, self.bit_vec,
                                                                   self.user_utt)

      # compute policy loss and regularization loss
      self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs, labels=self.taken_actions)
      self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
      self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
      self.loss               = self.pg_loss + self.reg_param * self.reg_loss


      # compute gradients
      self.gradients = self.optimizer.compute_gradients(self.loss)

      # compute policy gradients
      for i, (grad, var) in enumerate(self.gradients):
        if grad is not None:
          self.gradients[i] = (grad * self.discounted_rewards, var)

      for grad, var in self.gradients:
        tf.summary.histogram(var.name, var)
        if grad is not None:
          tf.summary.histogram(var.name + '/gradients', grad)

      # emit summaries
      tf.summary.scalar("policy_loss", self.pg_loss)
      tf.summary.scalar("reg_loss", self.reg_loss)
      tf.summary.scalar("total_loss", self.loss)

    # training update
    with tf.name_scope("train_policy_network"):
      # apply gradients to update policy network
      self.train_op = self.optimizer.apply_gradients(self.gradients)

    with tf.name_scope("SL_parts"):
      if not self.is_new_response:
        self.sl_actions = tf.placeholder(tf.int32, (None,), name="supervised_actions")
      else:
        self.sl_actions = tf.placeholder(tf.float32, (None, self.num_actions), name="supervised_actions_probs")
      with tf.name_scope("compute_SL_gradients"):
        if not self.is_new_response:
          self.sl_cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs,
                                                                                      labels=self.sl_actions)
        else:
          self.sl_cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logprobs,
                                                                               labels=self.sl_actions)
        tf.summary.scalar("sl_cross_entropy_loss", tf.reduce_sum(self.sl_cross_entropy_loss))
        self.sl_train_op = self.optimizer.minimize(self.sl_cross_entropy_loss)

    self.summarize = tf.summary.merge_all()
    self.no_op = tf.no_op()
    self.saver = tf.train.Saver()

  def sampleAction(self, states, initial_rnn_states, bit_vector, user_utt):
    # TODO: use this code piece when tf.multinomial gets better
    # sample action from current policy
    # actions = self.session.run(self.predicted_actions, {self.states: states})[0]
    # return actions[0]

    # temporary workaround
    def softmax(y):
      """ simple helper function here that takes unnormalized logprobs """
      # TODO, what if y is all zeros
      # print(y)
      zero_id = []
      nonzero_id = []
      nonzero_values = []
      for i, one_y in enumerate(y):
        if one_y != 0:
          nonzero_id += [i]
          nonzero_values += [one_y]
        else:
         zero_id += [i]
      # print y
      if len(nonzero_values) > 0:
        maxy = np.amax(nonzero_values)
        e = np.exp(nonzero_values - maxy)
        softmax_values = e / np.sum(e)
        returned = [0] * len(y)
        for i, id in enumerate(nonzero_id):
          returned[id] = softmax_values[i]
        return np.array(returned)
      else:
        return np.array([1/len(y)]*len(y))

    # epsilon-greedy exploration strategy
    if random.random() < self.exploration:
      print("exploration")
      available_act = np.array(range(self.num_actions))[np.array(bit_vector)==1]
      random_action = np.random.choice(available_act, size=1)[0]
      random_probs = np.array([0.05]*10)
      random_probs[random_action] = 0.55
      return random_action, initial_rnn_states, random_probs
    else:
      action_scores, prev_rnn_state_eval = self.session.run([self.action_scores, self.prev_rnn_state],
                                                           {self.states: states,
                                                            self.initial_rnn_state: initial_rnn_states,
                                                            self.bit_vec: bit_vector,
                                                            self.user_utt: user_utt})# [0]
      action_scores = action_scores[0]
      # print("in pg_reinforce, action_scores: {}".format(action_scores))
      action_probs  = softmax(action_scores) - 1e-5
      action = np.argmax(action_probs)
      return action, prev_rnn_state_eval, action_probs

  def updateModel(self, SL=False):

    N = len(self.state_buffer)
    r = 0 # use discounted reward to approximate Q value

    # if not SL:
      # compute discounted future rewards
    discounted_rewards = np.zeros(N)
    for t in reversed(range(N)):
      # future discounted reward from now on
      r = self.reward_buffer[t] + self.discount_factor * r
      discounted_rewards[t] = r
    # reduce gradient variance by normalization
    self.all_rewards += discounted_rewards.tolist()
    self.all_rewards = self.all_rewards[:self.max_reward_length]
    discounted_rewards -= np.mean(self.all_rewards)
    if np.std(self.all_rewards) > 0:
        discounted_rewards /= np.std(self.all_rewards)


    # whether to calculate summaries
    calculate_summaries = True# self.summary_writer is not None and self.train_iteration % self.summary_every == 0
    # print(discounted_rewards)
    # update policy network with the rollout in batches
    for t in range(N-1):

      # prepare inputs
      states  = self.state_buffer[t][np.newaxis, :]
      # if not SL:
      actions = np.array([self.action_buffer[t]])
      # if not SL:
      rewards = np.array([discounted_rewards[t]])
      rnn_states = self.rnn_state_buffer[t]
      bit_vecs = self.bit_vec_buffer[t]
      user_utts = self.user_utt_buffer[t]
      #if SL:
      sl_actions = np.array([self.sl_real_action_buffer[t]])

      if not SL:
        # evaluate gradients
        grad_evals = [grad for grad, var in self.gradients]

        # perform one update of training
        _, summary_str = self.session.run([
          self.train_op,
          self.summarize if calculate_summaries else self.no_op
        ], {
          self.states:             states,
          self.sl_actions:         sl_actions,
          self.taken_actions:      actions,
          self.discounted_rewards: rewards,
          self.initial_rnn_state:  rnn_states,
          self.bit_vec: bit_vecs,
          self.user_utt: user_utts
        })
      else:
        _, summary_str = self.session.run([
          self.sl_train_op,
          self.summarize if calculate_summaries else self.no_op
        ], {
          self.states:             states,
          self.sl_actions:      sl_actions,
          self.taken_actions: actions,
          self.discounted_rewards: rewards,
          self.initial_rnn_state:  rnn_states,
          self.bit_vec: bit_vecs,
          self.user_utt: user_utts
        })


      # emit summaries
      if calculate_summaries and not SL:
        self.summary_writer.add_summary(summary_str, self.train_iteration)

      if calculate_summaries and SL:
        self.summary_writer.add_summary(summary_str, self.sl_train_iteration)

    if not SL:
      self.annealExploration()
      self.train_iteration += 1
    else:
      self.sl_train_iteration += 1

    # clean up
    self.cleanUp()

  def annealExploration(self, stategy='linear'):
    ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
    self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

  def storeRollout(self, state, action, reward, rnn_states, bit_vecs, user_utt, real_action):
    self.action_buffer.append(action)
    self.reward_buffer.append(reward)
    self.state_buffer.append(state)
    self.rnn_state_buffer.append(rnn_states)
    self.bit_vec_buffer.append(bit_vecs)
    self.user_utt_buffer.append(user_utt)
    #if real_action is not None:
    self.sl_real_action_buffer.append(real_action)

  def cleanUp(self):
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []
    self.rnn_state_buffer = []
    self.bit_vec_buffer = []
    self.user_utt_buffer = []
    self.sl_real_action_buffer = []
