from __future__ import print_function, division, absolute_import
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from simulator import dialog_config
from config import Config
from rl.utils.replay_memory import Memory
import pdb

config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyGradientREINFORCE(object):

    def __init__(self,# session,
                     optimizer,
                     policy_network,
                     state_dim,
                     num_actions,
                     device=device,
                     discrete_act=True,
                     batch_size=64,
                     init_exp=0.5,         # initial exploration prob
                     final_exp=0.0,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=0.99, # discount future rewards
                     reg_param=0.001,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_every=100,
                     with_bit=True,
                     verbose=True,
                     replay=False):

        self.optimizer      = optimizer
        self.device         = device
        self.verbose = verbose
        if discrete_act:
            self.criterion = nn.NLLLoss()
        self.z_loss = nn.NLLLoss(ignore_index=0)

        # model components
        self.policy_network = policy_network

        # training parameters
        self.discrete_act = discrete_act
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

        # rollout buffer
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []
        self.bit_vec_buffer = []
        # sl rollout buffer
        self.sl_real_action_buffer = []
        # record reward history for normalization
        self.all_rewards = []
        self.max_reward_length = 1000000

        # replay
        self.with_bit = with_bit
        self.memory = Memory()
        self.batch_size = batch_size
        self.replay = replay

    def resetModel(self):
        self.cleanUp()
        self.train_iteration = 0
        # self.sl_train_iteration = 0
        self.exploration = self.init_exp

    def sampleAction(self, state, rl_test=False, bit_vecs=None, available_act=None, available_act_p=None):

        explored_act = None
        # epsilon-greedy exploration strategy
        if random.random() < self.exploration and not rl_test:
            if self.verbose:
                print("exploration")

            if bit_vecs is not None:
                available_act = [idx for idx, bit_vecs in enumerate(bit_vecs) if bit_vecs == 1]
            if available_act is None:
                available_act_p = [1/self.num_actions] * self.num_actions
                available_act = range(self.num_actions)
            else:
                available_act_p = [1/len(available_act)] * len(available_act)
                assert available_act_p
            selected_act = np.random.choice(available_act, replace=False, p=available_act_p)

            explored_act = torch.tensor([[selected_act]],
                                device=self.device, dtype=torch.long)
            # return torch.tensor([[selected_act]],
            #                     device=self.device, dtype=torch.long), turn_num, prev_z, turn_states

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            ######################################################################################################
            # output = self.policy_network(torch.tensor(state, dtype=torch.float32, device=self.device))# .max(1)[1].view(1, 1)
            usr_utt, turn_num, prev_m, prev_z, turn_states, z_input, np_state = state
            # import pdb
            # pdb.set_trace()
            result_from_policy_net = self.policy_network.rl_interactive_single_turn(np_state=np_state,
                                                                                    usr_utt=usr_utt,
                                                                                    turn_num=turn_num,
                                                                                    prev_m=prev_m,
                                                                                    prev_z=prev_z,
                                                                                    turn_states=turn_states,
                                                                                    true_z_input=z_input,
                                                                                    rl_test=rl_test)
            if self.discrete_act:
                turn_num, m_idx, prev_z, turn_states, pz_proba, converted_z_input = result_from_policy_net
            else:
                turn_num, m_idx, prev_z, turn_states, pz_proba, converted_z_input, mt_proba = result_from_policy_net
            # print(output)
            if self.discrete_act:
                output = F.softmax(m_idx, dim=1)
                if bit_vecs is None:
                    if explored_act is None:
                        ###############################################################
                        return output.max(1)[1].view(1, 1), turn_num, prev_z, turn_states
                    else:
                        return explored_act, turn_num, prev_z, turn_states
                else:
                    bit_vecs = torch.tensor(bit_vecs, dtype=torch.float32, device=self.device, requires_grad=False)
                    print("bit_vecs in sampleAction", bit_vecs)
                    output_after_bit = bit_vecs * output
                    print("output_after_bit in sampleAction", output_after_bit)

                    if explored_act is None:
                        return output_after_bit.max(1)[1].view(1, 1), turn_num, prev_z, turn_states
                    else:
                        return explored_act, turn_num, prev_z, turn_states

            else:
                return m_idx, turn_num, prev_z, turn_states


    def updateModel(self, mode=dialog_config.RL_TRAINING):

        if self.replay:
            print('in updateModel, memory length', len(self.memory))
            self.memory.push(self.state_buffer, self.action_buffer, self.bit_vec_buffer, self.reward_buffer)
            if len(self.memory) < self.batch_size:
                # clean up
                self.train_iteration += 1
                self.cleanUp()
                return
                # pdb.set_trace()
            else:
                if len(self.memory) % config.batch_size != 0:
                    self.train_iteration += 1
                    self.cleanUp()
                    return
                batch = self.memory.sample(self.batch_size)
                loss = self.generateBatchLoss(batch, mode)

        else:

            N = len(self.state_buffer)
            print(N)
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
            if len(self.all_rewards) > 100:
                discounted_rewards -= np.mean(self.all_rewards)
            if np.std(self.all_rewards) > 0:
                discounted_rewards /= np.std(self.all_rewards)


            # print(discounted_rewards)
            # update policy network with the rollout in batches
            cross_entropy = []
            rewards = []
            for t in range(N):
                # prepare inputs
                # states_t  = torch.tensor(self.state_buffer[t][np.newaxis, :], dtype=torch.float32, device=self.device)
                states_t = self.state_buffer[t]
                actions_t = torch.tensor([self.action_buffer[t]], dtype=torch.long, device=self.device)
                rewards_t = torch.tensor(discounted_rewards[t], dtype=torch.float32, device=self.device)
                # bitvecs_t = torch.tensor(self.bit_vec_buffer[t], dtype=torch.float32, device=self.device)

                usr_utt, turn_num, prev_m, prev_z, turn_states, z_input, np_state = states_t

                result_from_policy_net = self.policy_network.rl_interactive_single_turn(usr_utt=usr_utt,
                                                                                                      turn_num=turn_num,
                                                                                                      prev_m=prev_m,
                                                                                                      prev_z=prev_z,
                                                                                                      turn_states=turn_states,
                                                                                                      true_z_input=z_input,
                                                                                        np_state=np_state,
                                                                                                      rl_test=False)
                if self.discrete_act:
                    turn_num, m_idx, prev_z, turn_states, pz_proba, converted_z_input = result_from_policy_net
                else:
                    turn_num, m_idx, prev_z, turn_states, pz_proba, converted_z_input, mt_proba = result_from_policy_net

                if self.discrete_act:
                    print("m_idx", m_idx)
                    output_of_policy_net_t = F.log_softmax(m_idx, dim=1)
                #print(output_of_policy_net_t)
                #print(actions_t)
                print("**"*20)
                print("in update!!!!!!!!!!!!!!!")
                print("**"*20)
                print("time ", t)
                print("user utt", usr_utt)
                print("action ", actions_t)
                print("output from policy net", output_of_policy_net_t)
                print("pz_proba", pz_proba)
                pz_proba = torch.log(pz_proba)
                pz_proba = pz_proba[:, :, :config.vocab_size].contiguous()
                print("log pz_proba", pz_proba)
                print("converted_z_input", converted_z_input)
                loss_t_z = self.z_loss(pz_proba.view(-1, pz_proba.size(2)), converted_z_input.view(-1))
                if self.discrete_act:
                    loss_a = self.criterion(output_of_policy_net_t, actions_t)
                else:
                    raise NotImplementedError
                print('loss_z', loss_t_z)
                print('loss_a', loss_a)
                loss_t = loss_a + loss_t_z
                cross_entropy.append(loss_t.unsqueeze(0))
                rewards.append(rewards_t)

            #print(cross_entropy)
            cross_entropy = torch.cat(cross_entropy)
            rewards_tensor = [Variable(torch.zeros(1, device=self.device).fill_(r)) for r in rewards]
            rewards = torch.cat(rewards_tensor)

            if mode == dialog_config.RL_WARM_START:
                loss = cross_entropy.squeeze().sum()
                # print("loss in warm_start", loss)
            elif mode == dialog_config.RL_TRAINING:
                if cross_entropy.shape[0] == 1:
                    loss = cross_entropy.dot(rewards)
                    loss = loss
                else:
                    loss = cross_entropy.squeeze().dot(rewards.squeeze())
                # print("loss in rl_training", loss)
            else:
                raise ValueError("mode not correct {}".format(mode))


        ################# l2 loss ##################################
        reg_loss = []
        for par in self.policy_network.m.parameters():
            reg_loss.append(par.pow(2).sum().view(1, 1))

        reg_loss = torch.cat(reg_loss).sum()
        ###################################################

        loss = loss + self.reg_param * reg_loss

        # Optimize the model
        if loss.item() == 0:
            pass
            # pdb.set_trace()
        print("#" * 30)
        print("loss", loss)
        print("#" * 30)

        # for param in self.policy_network.named_parameters():
        #     if param[1].grad is None:
        #         pdb.set_trace()
        #         param[].grad.data.clamp_(-1, 1)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.m.parameters(), self.max_gradient)
        self.optimizer.step()


        self.annealExploration()
        self.train_iteration += 1

        # clean up
        self.cleanUp()

    def generateBatchLoss(self, batch, mode):
        losses = []
        state_buffer_batch, action_buffer_batch, bit_vec_buffer_batch, reward_buffer_batch = batch

        for i in range(self.batch_size):
            state_buffer, action_buffer, bit_vec_buffer, reward_buffer = state_buffer_batch[i], action_buffer_batch[i], \
                                                                         bit_vec_buffer_batch[i], reward_buffer_batch[i]

            N = len(state_buffer)
            # print(N)
            r = 0 # use discounted reward to approximate Q value

            # if not SL:
              # compute discounted future rewards
            discounted_rewards = np.zeros(N)
            for t in reversed(range(N)):
                # future discounted reward from now on
                if t >= len(reward_buffer):
                    pdb.set_trace()
                r = reward_buffer[t] + self.discount_factor * r
                discounted_rewards[t] = r
            # reduce gradient variance by normalization
            self.all_rewards += discounted_rewards.tolist()
            self.all_rewards = self.all_rewards[:self.max_reward_length]
            # if len(self.all_rewards) > 100:
            #     discounted_rewards -= np.mean(self.all_rewards)
            #     if np.std(self.all_rewards) > 0:
            #         discounted_rewards /= np.std(self.all_rewards)


            # print(discounted_rewards)
            # update policy network with the rollout in batches
            cross_entropy = []
            rewards = []
            for t in range(N):
                # prepare inputs
                # states_t  = torch.tensor(self.state_buffer[t][np.newaxis, :], dtype=torch.float32, device=self.device)
                states_t = state_buffer[t]
                actions_t = torch.tensor([action_buffer[t]], dtype=torch.long, device=self.device)
                rewards_t = torch.tensor(discounted_rewards[t], dtype=torch.float32, device=self.device)
                if self.with_bit:
                    bitvecs_t = torch.tensor(bit_vec_buffer[t], dtype=torch.float32, device=self.device,
                                             requires_grad=False)

                usr_utt, turn_num, prev_m, prev_z, turn_states, z_input, np_state = states_t
                import pdb
                # pdb.set_trace()
                result_from_policy_net = self.policy_network.rl_interactive_single_turn(usr_utt=usr_utt,
                                                                                        turn_num=turn_num,
                                                                                        prev_m=prev_m,
                                                                                        prev_z=prev_z,
                                                                                        turn_states=turn_states,
                                                                                        np_state=np_state,
                                                                                        true_z_input=z_input,
                                                                                        rl_test=False)
                if self.discrete_act:
                    turn_num, m_idx, prev_z, turn_states, pz_proba, converted_z_input = result_from_policy_net
                else:
                    turn_num, m_idx, prev_z, turn_states, pz_proba, converted_z_input, mt_proba = result_from_policy_net

                if self.discrete_act:
                    # print("m_idx", m_idx)
                    output_of_policy_net_t = m_idx
                    if self.with_bit:
                        output_of_policy_net_t = bitvecs_t * output_of_policy_net_t
                    output_of_policy_net_t = F.log_softmax(output_of_policy_net_t, dim=1)

                #print(output_of_policy_net_t)
                #print(actions_t)
                # print("**"*20)
                # print("in update!!!!!!!!!!!!!!!")
                # print("**"*20)
                # print("time ", t)
                # print("user utt", usr_utt)
                # print("action ", actions_t)
                # print("output from policy net", output_of_policy_net_t)
                # print("pz_proba", pz_proba)
                # pz_proba = torch.log(pz_proba)
                # pz_proba = pz_proba[:, :, :config.vocab_size].contiguous()
                # print("log pz_proba", pz_proba)
                # print("converted_z_input", converted_z_input)
                loss_t_z = self.z_loss(pz_proba.view(-1, pz_proba.size(2)), converted_z_input.view(-1))
                if self.discrete_act:
                    loss_a = self.criterion(output_of_policy_net_t, actions_t)
                else:
                    raise NotImplementedError
                # print('loss_z', loss_t_z)
                # print('loss_a', loss_a)
                loss_t = loss_a # + loss_t_z
                cross_entropy.append(loss_t.unsqueeze(0))
                rewards.append(rewards_t)

            #print(cross_entropy)
            cross_entropy = torch.cat(cross_entropy)
            rewards_tensor = [Variable(torch.zeros(1, device=self.device).fill_(r)) for r in rewards]
            rewards = torch.cat(rewards_tensor)

            if mode == dialog_config.RL_WARM_START:
                loss = cross_entropy.squeeze().sum()
                # print("loss in warm_start", loss)
            elif mode == dialog_config.RL_TRAINING:
                if cross_entropy.shape[0] == 1:
                    loss = cross_entropy.dot(rewards)
                    loss = loss
                else:
                    loss = cross_entropy.squeeze().dot(rewards.squeeze())
                # print("loss in rl_training", loss)
            else:
                raise ValueError("mode not correct {}".format(mode))
            losses.append(loss.view(1, 1))

        losses = torch.cat(losses).sum()

        return losses


    def annealExploration(self, stategy='linear'):
        ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
        self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

    def storeRollout(self, state, action, reward, bit_vecs=None, real_action=None):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)
        #self.rnn_state_buffer.append(rnn_states)
        self.bit_vec_buffer.append(bit_vecs)
        #self.user_utt_buffer.append(user_utt)

    def cleanUp(self):
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []
        #self.rnn_state_buffer = []
        self.bit_vec_buffer = []
        #self.user_utt_buffer = []
        self.sl_real_action_buffer = []
