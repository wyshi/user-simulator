from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
# Author: Tiancheng Zhao
# Date: 9/13/17

import numpy as np
from simulator.agent.core import SystemAct, UserAct, BaseUsrSlot
from simulator.agent import core
import json
import copy


def sample_from(action):
    """

    :param action: action.act = UserAct.INFORM_TYPE
                   action.parameters = {'<type>': 'chinese',
                                        '<pricerange>': 'expensive'}
    :return:
    """


    sampled_template = 'I would like a <pricerange> <type>  restaurant.'
    sampled_sent = sampled_template.replace('<pricerange>', action.parameters['<pricerange>'])

    return sampled_sent

class AbstractNlg(object):
    """
    Abstract class of NLG
    """

    def __init__(self, domain, complexity):
        self.domain = domain
        self.complexity = complexity

    def generate_sent(self, actions, **kwargs):
        """
        Map a list of actions to a string.

        :param actions: a list of actions
        :return: uttearnces in string
        """
        raise NotImplementedError("Generate sent is required for NLG")

    def sample(self, examples):
        return np.random.choice(examples)


class SysCommonNlg(object):
    """
    template or sampling
    """
    templates = {
        SystemAct.ASK_TYPE: ["Do you have any preference on the [informable_slot]?"],
                             # "How about the [informable_slot]?",
                             # "What's the [informable_slot] preference?"],
        SystemAct.PRESENT_RESULT: ["[name] is a [food] restaurant on the [area] side of town. It's in the [pricerange] price range. Is there anything else I can help you with?"],
        SystemAct.NOMATCH_RESULT: ["I am sorry, but there are no restaurants matching your request. Is there anything else I can help you with?"],
        SystemAct.NO_OTHER: ["I'm sorry, there is no other restaurant matching your request. "],
        SystemAct.PROVIDE_INFO: ["[name] is located at [address] . Its phone number is [phone] and the postcode is [postcode] . is there anything else i can help you with ?"],

        # task 2. reservation
        SystemAct.ASK_RESERVATION_INFO: ["Could you please tell me [reservation_slot] you would like for the reservation?"],
        SystemAct.BOOKING_SUCCESS: ["I have booked the reservation for you. And you reference number is [reference]."],
        SystemAct.BOOKING_FAIL: ["I am sorry, but the restaurant is fully booked at that time. Is there anything else I can help you with?"],
        #REFERENCE_NUM = "reference_num"
        SystemAct.GOODBYE: ["Thank you for using the system. Have a good day and goodbye!"]}


class UsrCommonNlg(object):
    """
    template or sampling
    """
    templates = {

    UserAct.INFORM_TYPE: ["I am looking for a restaurant, "],
    UserAct.INFORM_TYPE_CHANGE: ['how about '],
    UserAct.ASK_INFO: ["What is the following info of the restaurant, "],
    UserAct.MAKE_RESERVATION: ["can you help me to make a reservation for [people] people at [time] on [day]?"],
    UserAct.MAKE_RESERVATION_CHANGE_TIME: ["how about [time] on [day]?"],
    UserAct.ANYTHING_ELSE: ["Are there any other options?"],
    UserAct.GOODBYE: ["Thank you for helping me! Goodbye"]

    }



class SysNlg(AbstractNlg):
    """
    NLG class to generate utterances for the system side.
    """
    def __init__(self, domain, complexity):
        # super
        super().__init__(domain=domain, complexity=complexity)
        self.domain = domain
        self.complexity = complexity

    def generate_sent(self, actions, domain=None, templates=SysCommonNlg.templates, generator=None, context=None):
        """
         Map a list of system actions to a string.

        :param actions: a list of actions
        :param templates: a common NLG template that uses the default one if not given
        :return: utterances in string
        """
        str_actions = []
        lexicalized_actions = []

        if not isinstance(actions, list):
            actions = [actions]

        if generator is None:
            actions_with_parameters = [SystemAct.ASK_TYPE, SystemAct.PRESENT_RESULT, SystemAct.PROVIDE_INFO,
                                       SystemAct.BOOKING_SUCCESS, SystemAct.ASK_RESERVATION_INFO]
            actions_without_parameters = [SystemAct.NOMATCH_RESULT, SystemAct.NO_OTHER,
                                          SystemAct.BOOKING_FAIL,
                                          SystemAct.GOODBYE]

            for a in actions:
                a_copy = copy.deepcopy(a)
                if a.act in actions_without_parameters:
                    if domain:
                        str_actions.append(domain.greet)
                    else:
                        str_actions.append(self.sample(templates[a.act]))
                elif a.act == SystemAct.ASK_TYPE:
                    args = [p for p in a.parameters]
                    if len(args) == 1:
                        args = args[0]
                    elif len(args) == 2:
                        args = args[0] + " and " + args[1]
                    elif len(args) == 3:
                        args = args[0] + ", " + args[1] + " and " + args[2]
                    # args = ", ".join(args)
                    str_actions.append(self.sample(templates[a.act]).replace("[informable_slot]", args))
                elif a.act == SystemAct.PRESENT_RESULT:
                    sent = self.sample(templates[a.act])
                    if 'name' in a.parameters:
                        sent = sent.replace("[name]", a.parameters['name'])
                    if 'food' in a.parameters:
                        sent = sent.replace("[food]", a.parameters['food'])
                    if 'area' in a.parameters:
                        sent = sent.replace("[area]", a.parameters['area'])
                    if 'pricerange' in a.parameters:
                        sent = sent.replace("[pricerange]", a.parameters['pricerange'])
                    str_actions.append(sent)
                elif a.act == SystemAct.PROVIDE_INFO:
                    sent = self.sample(templates[a.act])
                    if 'name' in a.parameters:
                        sent = sent.replace("[name]", a.parameters['name'])
                    if 'address' in a.parameters:
                        sent = sent.replace("[address]", a.parameters['address'])
                    if 'phone' in a.parameters:
                        sent = sent.replace("[phone]", a.parameters['phone'])
                    if 'postcode' in a.parameters:
                        sent = sent.replace("[postcode]", a.parameters['postcode'])
                    str_actions.append(sent)

                elif a.act == SystemAct.BOOKING_SUCCESS:
                    sent = self.sample(templates[a.act])
                    if 'reference' in a.parameters:
                        sent = sent.replace("[reference]", a.parameters['reference'])
                    str_actions.append(sent)
                elif a.act == SystemAct.ASK_RESERVATION_INFO:
                    sent = self.sample(templates[a.act])
                    args = [p for p in a.parameters]
                    if len(args) == 3:
                        args = 'how many people and what day and time'
                    elif len(args) == 2:
                        if 'people' in args and 'day' in args:
                            args = 'how many people and what day'
                        elif 'people' in args and 'time' in args:
                            args = 'how many people and what time'
                        elif 'time' in args and 'day' in args:
                            args = 'what day and time'
                    elif len(args) == 1:
                        if 'people' in args:
                            args = 'how many people'
                        else:
                            args = 'what ' + args[0]
                    else:
                        print(args)
                    sent = sent.replace("[reservation_slot]", args)
                    str_actions.append(sent)
                else:
                    raise ValueError("Unknown dialog act %s" % a.act)

                lexicalized_actions.append(a_copy)

            return " ".join(str_actions), lexicalized_actions
        else:
            raise NotImplementedError
            return " ".join(str_actions), lexicalized_actions


class UserNlg(AbstractNlg):
    """
    NLG class to generate utterances for the user side.
    """
    def __init__(self, domain=None, complexity=None):
        # super
        super().__init__(domain=domain, complexity=complexity)
        self.domain = domain
        self.complexity = complexity
        self.default_templates = UsrCommonNlg.templates
        import json
        # with open("dataset/full_data/er_act_to_label.json", "r") as fh:
        #     self.SysAct_to_label = json.load(fh)
#
        # with open("dataset/full_data/ee_act_to_label.json", "r") as fh:
        #     self.UsrAct_to_label = json.load(fh)

    def generate_sent(self, actions, templates=UsrCommonNlg.templates, generator=None, context=None):
        """
         Map a list of user actions to a string.

        :param actions: a list of actions
        :return: uttearnces in string
        """
        str_actions = []
        lexicalized_actions = []

        if not isinstance(actions, list):
            actions = [actions]

        if generator is None:
            actions_without_parameters = [UserAct.ANYTHING_ELSE, UserAct.GOODBYE]
            actions_with_parameters = [UserAct.INFORM_TYPE,
                                       UserAct.INFORM_TYPE_CHANGE,
                                       UserAct.ASK_INFO,
                                       UserAct.MAKE_RESERVATION,
                                       UserAct.MAKE_RESERVATION_CHANGE_TIME ]

            for a in actions:
                a_copy = copy.deepcopy(a)
                if a.act in actions_without_parameters:
                    str_actions.append(self.sample(templates[a.act]))

                elif a.act == UserAct.INFORM_TYPE:
                    args = [p + " " + v if p!='pricerange' else 'price range' + " " + v for p, v in a.parameters.items()]
                    sent = self.sample(templates[a.act]) + ", ".join(args)
                    str_actions.append(sent)

                elif a.act == UserAct.INFORM_TYPE_CHANGE:
                    args = [p + " " + v if p!='pricerange' else 'price range' + " " + v  for p, v in a.parameters.items()]
                    sent = self.sample(templates[a.act]) + ", ".join(args)
                    str_actions.append(sent)

                elif a.act == UserAct.ASK_INFO:
                    args = [p for p in a.parameters]
                    sent = self.sample(templates[a.act]) + " ".join(args) + "?"
                    str_actions.append(sent)

                elif a.act in [UserAct.MAKE_RESERVATION, UserAct.MAKE_RESERVATION_CHANGE_TIME]:
                    sent = self.sample(templates[a.act])
                    if 'people' in a.parameters:
                        sent = sent.replace("[people]", a.parameters['people'])
                    if 'day' in a.parameters:
                        sent = sent.replace("[day]", a.parameters['day'])
                    if 'time' in a.parameters:
                        sent = sent.replace("[time]", a.parameters['time'])
                    str_actions.append(sent)

                else:
                    raise ValueError("Unknown dialog act %s" % a.act)

                lexicalized_actions.append(a_copy)

            return " ".join(str_actions), lexicalized_actions

        else:
            actions_without_parameters = [UserAct.ANYTHING_ELSE, UserAct.GOODBYE]
            actions_with_parameters = [UserAct.INFORM_TYPE,
                                       UserAct.INFORM_TYPE_CHANGE,
                                       UserAct.ASK_INFO,
                                       UserAct.MAKE_RESERVATION,
                                       UserAct.MAKE_RESERVATION_CHANGE_TIME ]

            for a in actions:
                a_copy = copy.deepcopy(a)
                if a.act in actions_without_parameters:
                    str_actions.append(self.sample(templates[a.act][('',)]))

                elif a.act == UserAct.INFORM_TYPE:
                    # args = [p + " " + v for p, v in a.parameters.items()]
                    sorted_ps = tuple(sorted(["value_"+p if p != 'name' else 'restaurant_name' for p in a.parameters]))
                    sent = self.sample(templates[a.act][sorted_ps])
                    for p, v in a.parameters.items():
                        sent = sent.replace("value_"+p, v)
                    str_actions.append(sent)

                elif a.act == UserAct.INFORM_TYPE_CHANGE:
                    sorted_ps = tuple(sorted(["value_"+p if p != 'name' else 'restaurant_name' for p in a.parameters]))
                    if sorted_ps not in templates[a.act]:
                        args = [p + " " + v for p, v in a.parameters.items()]
                        sent = self.sample(self.default_templates[a.act]) + ", ".join(args)
                    else:
                        sent = self.sample(templates[a.act][sorted_ps])
                        for p, v in a.parameters.items():
                            sent = sent.replace("value_"+p, v)
                    str_actions.append(sent)

                elif a.act == UserAct.ASK_INFO:
                    args = [p for p in a.parameters]
                    sent = "what's the following information? " + " ".join(args) + "?"

                    # sorted_ps = tuple(sorted([p for p in a.parameters]))
                    # sent = "what's the following information? "#self.sample(templates[a.act][sorted_ps])
                    # for p, v in a.parameters.items():
                    #     sent.replace("value_"+p, v)
                    str_actions.append(sent)

                elif a.act in [UserAct.MAKE_RESERVATION]:
                    sorted_ps = tuple(sorted(["value_"+p  if p != 'people' else 'value_count' for p in a.parameters]))
                    sent = self.sample(templates[a.act][sorted_ps])
                    # print(a.parameters)
                    for p, v in a.parameters.items():
                        # print("v: ", v)
                        if p == 'people':
                            sent = sent.replace("value_count", v)
                        else:
                            sent = sent.replace("value_"+p, v)
                    str_actions.append(sent)

                elif a.act == UserAct.MAKE_RESERVATION_CHANGE_TIME:
                    sorted_ps = tuple(sorted(["value_"+p  if p != 'people' else 'value_count' for p in a.parameters]))
                    if sorted_ps not in templates[a.act]:
                        sent = self.sample(self.default_templates[a.act])
                        if 'people' in a.parameters:
                            sent = sent.replace("[people]", a.parameters['people'])
                        if 'day' in a.parameters:
                            sent = sent.replace("[day]", a.parameters['day'])
                        if 'time' in a.parameters:
                            sent = sent.replace("[time]", a.parameters['time'])
                    else:
                        sent = self.sample(templates[a.act][sorted_ps])
                        for p, v in a.parameters.items():
                            if p == 'people':
                                sent = sent.replace("value_count", v)
                            else:
                                sent = sent.replace("value_"+p, v)
                    str_actions.append(sent)

                else:
                    raise ValueError("Unknown dialog act %s" % a.act)

                lexicalized_actions.append(a_copy)

            return " ".join(str_actions), lexicalized_actions


    def add_hesitation(self, sents, actions):
        pass

    def add_self_restart(self, sents, actions):
        pass

