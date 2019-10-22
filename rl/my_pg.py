from __future__ import print_function, division, absolute_import
from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from simulator import dialog_config
from rl.utils.replay_memory import Memory
import pdb
#from config import Config
#config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyGradientREINFORCE(object):

    def __init__(self,# session,
                     optimizer,
                     policy_network,
                     state_dim,
                     num_actions,
                     config,
                     device=device,
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
        self.criterion = nn.NLLLoss()
        self.config = config

        # model components
        self.policy_network = policy_network

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

        # epsilon-greedy exploration strategy
        if random.random() < self.exploration and not rl_test:
            if self.verbose:
                print("exploration")

            if bit_vecs is not None:
                available_act = [idx for idx, bit_vecs in enumerate(bit_vecs) if bit_vecs == 1]
                # print("available_Act in sampleAction", available_act)
            if available_act is None:
                available_act_p = [1/self.num_actions] * self.num_actions
                available_act = range(self.num_actions)
            else:
                available_act_p = [1/len(available_act)] * len(available_act)
                assert available_act_p
            print("available_Act in sampleAction", available_act)
            selected_act = np.random.choice(available_act, replace=False, p=available_act_p)

            return torch.tensor([[selected_act]],
                                device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                ######################################################################################################
                output = self.policy_network(state)# .max(1)[1].view(1, 1)
                # print(output)

                # output = F.softmax(output, dim=1)
                if bit_vecs is None:
                    output = F.softmax(output, dim=1)
                    if not self.config.use_multinomial:

                        ###############################################################
                        if len(set(output.tolist()[0])) == 1:
                            return output.multinomial(1).view(1, 1)
                        else:
                            # pdb.set_trace()
                            return output.max(1)[1].view(1, 1)
                    else:
                        return output.multinomial(1).view(1, 1)
                else:
                    if not self.config.use_multinomial:
                        output = F.softmax(output, dim=1)

                        # valid_logits = [output_after_softmax[i] for i, bit in enumerate(bitvecs_t) if bit == 1]
                        # valid_logits = F.softmax(torch.tensor(valid_logits, dtype=torch.float32))
                        # valid_probs = torch.tensor([valid_logits[i] if bit == 1 else 0.0 for i, bit in enumerate(bitvecs_t)],
                        #                            dtype=torch.float32)
                        #
                        # output_of_policy_net_t = torch.log(valid_probs)



                        valid_probs = [output[0][i] for i, b in enumerate(bit_vecs) if b == 1]
                        bit_vecs = torch.tensor(bit_vecs, dtype=torch.float32, device=self.device, requires_grad=False)
                        print("bit_vecs in sampleAction", bit_vecs)
                        output_after_bit = bit_vecs * output
                        print("output_after_bit in sampleAction", output_after_bit)

                        ###############################################################
                        if len(set(valid_probs)) == 1:
                            return output_after_bit.multinomial(1).view(1, 1)
                        else:
                            return output_after_bit.max(1)[1].view(1, 1)
                    else:
                        output_after_softmax = F.softmax(output, dim=1)

                        valid_logits = [(i, output_after_softmax[0][i]) for i, bit in enumerate(bit_vecs) if bit == 1]
                        valid_logits_tensor = F.softmax(torch.tensor([v[1] for v in valid_logits], dtype=torch.float32),
                                                        dim=0)
                        valid_probs = [0] * output_after_softmax.shape[1]
                        for i, v in enumerate(valid_logits):
                            valid_probs[v[0]] = valid_logits_tensor[i]
                        valid_probs = torch.tensor(valid_probs,
                                                   dtype=torch.float32)

                        return valid_probs.multinomial(1).view(1, 1)

    def updateModel(self, mode=dialog_config.RL_TRAINING):

        if self.replay:
            print('in updateModel, memory length', len(self.memory))
            self.memory.push(self.state_buffer, self.action_buffer, self.bit_vec_buffer, self.reward_buffer)
            if len(self.memory) < self.batch_size:
                # clean up
                self.train_iteration += 1
                self.cleanUp()
                # pdb.set_trace()
                return
                # pdb.set_trace()
            else:
                if len(self.memory) % self.config.update_every != 0:
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
                states_t  = self.state_buffer[t]#[np.newaxis, :]# , dtype=torch.float32, device=self.device)
                # states_t = torch.Tensor(states_t)
                actions_t = torch.tensor([self.action_buffer[t]], dtype=torch.long, device=self.device)
                rewards_t = torch.tensor(discounted_rewards[t], dtype=torch.float32, device=self.device)
                if self.with_bit:
                    bitvecs_t = torch.tensor(self.bit_vec_buffer[t], dtype=torch.float32,
                                             device=self.device,
                                             requires_grad=False)

                output_of_policy_net_t = self.policy_network(states_t)
                # pdb.set_trace()
                if self.with_bit:
                    output_of_policy_net_t = bitvecs_t * output_of_policy_net_t
                #print(output_of_policy_net_t)
                #print(actions_t)
                loss_t = self.criterion(output_of_policy_net_t, actions_t)
                cross_entropy.append(loss_t.unsqueeze(0))
                rewards.append(rewards_t)


            cross_entropy = torch.cat(cross_entropy)
            rewards_tensor = [Variable(torch.zeros(1, device=self.device).fill_(r)) for r in rewards]
            rewards = torch.cat(rewards_tensor)
            print("nll", cross_entropy)
            print("rewards", rewards)


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
        # pdb.set_trace()
        ################# l2 loss ##################################
        reg_loss = []
        for par in self.policy_network.parameters():
            reg_loss.append(par.pow(2).sum().view(1, 1))

        reg_loss = torch.cat(reg_loss).sum()
        ###################################################

        # pdb.set_trace()
        loss = loss + self.reg_param * reg_loss
        loss = loss.mean()

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
        for name, param in self.policy_network.named_parameters():
            if param.grad is None:
                print('no grad', name)
                # pdb.set_trace()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_gradient)
        self.optimizer.step()

        # if not self.replay:
        #     # reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
        #     reg_loss = []
        #     for par in self.policy_network.parameters():
        #         reg_loss.append(par.pow(2).sum().view(1, 1))
        #
        #     # pdb.set_trace()
        #     reg_loss = torch.cat(reg_loss).sum()
        #
        #     loss = loss + self.reg_param * reg_loss
        #
        #     # Optimize the model
        #     if loss.item() == 0:
        #         pass
        #         #pdb.set_trace()
        #     print("#"*30)
        #     print("loss", loss)
        #     print("#"*30)
        #
        #     # for param in self.policy_network.named_parameters():
        #     #     if param[1].grad is None:
        #     #         pdb.set_trace()
        #     #         param[].grad.data.clamp_(-1, 1)
        #
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_gradient)
        #     self.optimizer.step()
        #
        # else:
        #     if len(self.memory) <= self.batch_size:
        #         self.memory.append(loss.view(1, 1))
        #             #pdb.set_trace()
        #         print("#"*30)
        #         print("loss", loss)
        #         print("#"*30)
        #
        #
        #     else:
        #
        #         loss = self.sample(self.memory, self.batch_size)
        #         loss = list(loss)
        #         loss = torch.cat(loss).sum()
        #         # reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
        #         reg_loss = []
        #         for par in self.policy_network.parameters():
        #             reg_loss.append(par.pow(2).sum().view(1, 1))
        #
        #         # pdb.set_trace()
        #         reg_loss = torch.cat(reg_loss).sum()
        #
        #         loss = loss + self.reg_param * reg_loss
        #
        #         # Optimize the model
        #         if loss.item() == 0:
        #             pass
        #             #pdb.set_trace()
        #         print("#"*30)
        #         print("loss", loss)
        #         print("#"*30)
        #
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm(self.policy_network.parameters(), 40)
        #         # for param in self.policy_network.parameters():
        #         #     param.grad.data.clamp_(-1, 1)
        #         self.optimizer.step()

        self.annealExploration()
        self.train_iteration += 1

        # clean up
        self.cleanUp()

    def generateBatchLoss(self, batch, mode):
        losses = []
        state_buffer_batch, action_buffer_batch, bit_vec_buffer_batch, reward_buffer_batch = batch
        for i in range(self.batch_size):
            state_buffer, action_buffer, bit_vec_buffer, reward_buffer = state_buffer_batch[i], action_buffer_batch[i],\
                                                                         bit_vec_buffer_batch[i], reward_buffer_batch[i]
            # pdb.set_trace()
            N = len(state_buffer)
            # print(N)
            r = 0 # use discounted reward to approximate Q value

            # if not SL:
              # compute discounted future rewards
            discounted_rewards = np.zeros(N)
            for t in reversed(range(N)):
                # future discounted reward from now on
                r = reward_buffer[t] + self.discount_factor * r
                discounted_rewards[t] = r
            # reduce gradient variance by normalization
            self.all_rewards += discounted_rewards.tolist()
            self.all_rewards = self.all_rewards[:self.max_reward_length]
            # if len(self.all_rewards) > 1000:
            #     discounted_rewards -= np.mean(self.all_rewards)
            #     if np.std(self.all_rewards) > 0:
            #         discounted_rewards /= np.std(self.all_rewards)


            # print(discounted_rewards)
            # update policy network with the rollout in batches
            cross_entropy = []
            rewards = []
            for t in range(N):
                # prepare inputs
                states_t  = state_buffer[t]# torch.tensor(state_buffer[t][np.newaxis, :], dtype=torch.float32, device=self.device)
                # states_t = torch.Tensor(states_t)
                actions_t = torch.tensor([action_buffer[t]], dtype=torch.long, device=self.device)
                rewards_t = torch.tensor(discounted_rewards[t], dtype=torch.float32, device=self.device)
                if self.with_bit:
                    bitvecs_t = bit_vec_buffer[t]#torch.tensor(bit_vec_buffer[t], dtype=torch.float32, device=self.device,
                                             #requires_grad=False)

                output_of_policy_net_t = self.policy_network(states_t)
                if self.config.bit_not_used_in_update:
                    output_of_policy_net_t = F.log_softmax(output_of_policy_net_t, dim=1)
                else:
                    if self.with_bit:
                        # pdb.set_trace()
                        if self.config.use_multinomial:
                            output_after_softmax = F.softmax(output_of_policy_net_t, dim=1)

                            valid_logits = [(i, output_after_softmax[0][i]) for i, bit in enumerate(bitvecs_t) if bit == 1]
                            valid_logits_tensor = F.softmax(torch.tensor([v[1] for v in valid_logits], dtype=torch.float32), dim=0)
                            valid_probs = [0]*output_after_softmax.shape[1]
                            for i, v in enumerate(valid_logits):
                                valid_probs[v[0]] = valid_logits_tensor[i]
                            valid_probs = torch.tensor(valid_probs,
                                                       dtype=torch.float32)

                            output_of_policy_net_t = torch.log(valid_probs).unsqueeze(0)
                        else:
                            bitvecs_t = torch.tensor(bit_vec_buffer[t], dtype=torch.float32, device=self.device)
                            output_of_policy_net_t = bitvecs_t * output_of_policy_net_t
                            output_of_policy_net_t = F.log_softmax(output_of_policy_net_t)


                    # pdb.set_trace()
                    else:
                        output_of_policy_net_t = F.log_softmax(output_of_policy_net_t, dim=1)

                #print("output_of_policy_net_t in generateBatchLoss", output_of_policy_net_t)
                #print(actions_t)

                loss_t = self.criterion(output_of_policy_net_t, actions_t)
                cross_entropy.append(loss_t.unsqueeze(0))
                rewards.append(rewards_t)


            # cross_entropy_tensor([Variable(torch.zeros(1, device=self.device).fill_(c)) for c in cross_entropy])
            cross_entropy = torch.cat(cross_entropy)
            # cross_entropy = Variable(cross_entropy)
            # pdb.set_trace()
            #print("cross_entropy require_grad", cross_entropy.requires_grad)
            rewards_tensor = [Variable(torch.zeros(1, device=self.device).fill_(r)) for r in rewards]
            rewards = torch.cat(rewards_tensor)
            # print("nll", cross_entropy)
            # print("rewards", rewards)


            if mode == dialog_config.RL_WARM_START:
                loss = cross_entropy.squeeze().mean()
                # print("loss in warm_start", loss)
            elif mode == dialog_config.RL_TRAINING:
                if cross_entropy.shape[0] == 1:
                    loss = (cross_entropy * rewards).mean()# cross_entropy.dot(rewards)
                    # loss = loss
                else:
                    loss = (cross_entropy.squeeze() * rewards.squeeze()).mean()
                # print("loss in rl_training", loss)
            else:
                raise ValueError("mode not correct {}".format(mode))
            losses.append(loss.view(1, 1))

        losses = torch.cat(losses)# .mean()
        # pdb.set_trace()

        return losses


    def sample(self, examples, n=1, p=None):
        if p is None:
            # uniform
            p = [1 / len(examples)] * len(examples)

        if n == 1:
            return np.random.choice(examples, p=p)
        else:
            return list(np.random.choice(examples, n, replace=False, p=p))

    def annealExploration(self, strategy='linear'):
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
