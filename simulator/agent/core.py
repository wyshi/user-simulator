# -*- coding: utf-8 -*-
# author: Tiancheng Zhao

import logging
import copy


class Agent(object):
    """
    Abstract class for Agent (user or system)
    """

    def __init__(self, domain, complexity):
        self.domain = domain
        self.complexity = complexity

    def step(self, *args, **kwargs):
        """
        Given the new inputs, generate the next response

        :return: reward, terminal, response
        """
        raise NotImplementedError("Implement step function is required")


class Action(dict):
    """
    A generic class that corresponds to a discourse unit. An action is made of an Act and a list of parameters.

    :ivar act: dialog act String
    :ivar parameters: [{slot -> usr_constrain}, {sys_slot -> value}] for INFORM, and [(type, value)...] for other acts.

    """

    def __init__(self, act, parameters=None):
        self.act = act
        if parameters is None:
            self.parameters = {}
        elif type(parameters) is not dict:
            self.parameters = {parameters: ""}
        else:
            self.parameters = parameters
        super(Action, self).__init__(act=self.act, parameters=self.parameters)

    def add_parameter(self, type, value):
        self.parameters[type] = value

    def dump_string(self):
        str_paras = []
        for p in self.parameters:
            if type(p) is not str:
                str_paras.append(str(p))
            else:
                str_paras.append(p)
        str_paras = "-".join(str_paras)
        return "%s:%s" % (self.act, str_paras)


class State(object):
    """
    The base class for a dialog state

    :ivar history: a list of turns
    :cvar USR: user name
    :cvar SYS: system name
    :cvar LISTEN: the agent is waiting for other's input
    :cvar SPEAK: the agent is generating it's output
    :cvar EXT: the agent leaves the session
    """

    USR = "usr"
    SYS = "sys"

    LISTEN = "listen"
    SPEAK = "speak"
    EXIT = "exit"

    def __init__(self):
        self.history = []

    def yield_floor(self, *args, **kwargs):
        """
        Base function that decides if the agent should yield the conversation floor
        """
        raise NotImplementedError("Yield is required")

    def is_terminal(self, *args, **kwargs):
        """
        Base function decides if the agent is left
        """
        raise NotImplementedError("is_terminal is required")

    def last_actions(self, target_speaker):
        """
        Search in the dialog hisotry given a speaker.

        :param target_speaker: the target speaker
        :return: the last turn produced by the given speaker. None if not found.
        """
        for spk, utt in self.history[::-1]:
            if spk == target_speaker:
                return utt
        return None

    def update_history(self, speaker, actions):
        """
        Append the new turn into the history

        :param speaker: SYS or USR
        :param actions: a list of Action
        """
        # make a deep copy of actions
        self.history.append((speaker, copy.deepcopy(actions)))


class SystemAct(object):
    """
    :cvar IMPLICIT_CONFIRM: you said XX
    :cvar EXPLICIT_CONFIRM: do you mean XX
    :cvar INFORM: I think XX is a good fit
    :cvar REQUEST: which location?
    :cvar GREET: hello
    :cvar GOODBYE: goodbye
    :cvar CLARIFY: I think you want either A or B. Which one is right?
    :cvar ASK_REPHRASE: can you please say it in another way?
    :cvar ASK_REPEAT: what did you say?
    """
    """
    GREET = "greet"
    INTRODUCE_CHARITY = "introduce_charity"
    SUGGEST_DONATION = "suggest_donation" # P1 yes
    PROVIDE_FACT = "provide_fact" # L1 yes
    DONATION_IMPACT = "donation_impact" # L3 yes
    # PERSONAL_STORY = "personal_story"
    EMOTION_APPEAL = "emotion_appeal" # E2 yes
    TASK_INFO = "task_info" # TF yes
    INDICATE_SELF_DONATION = "indicate_self_donation" # The persuade often ask whether persuader will donate. In order to persuade the persuadee to donate, most of persuaders will show their approval
    THANK_DONATION = "thank_donation"
    END_CONV = "end_conv" # cannot appear very soon

    OTHER = "other"
    """
    # task 1. restaurant recommendation
    ASK_TYPE = "ask_type"
    PRESENT_RESULT = "present_result"
    NOMATCH_RESULT = "nomatch_result"
    NO_OTHER = "no_other"
    PROVIDE_INFO = "provide_info"

    # task 2. reservation
    ASK_RESERVATION_INFO = "ask_reservation_info"
    BOOKING_SUCCESS = "booking_success"
    BOOKING_FAIL = "booking_fail"
    #REFERENCE_NUM = "reference_num"
    GOODBYE = "goodbye"


    # 'Restaurant-Inform',
    # 'Restaurant-Recommend',
    # 'Restaurant-Select',
    # 'Restaurant-Request',
    # 'Restaurant-NoOffer',
#
    # 'Booking-Book',
    # 'Booking-Request',
    # 'Booking-Inform',
    # 'Booking-NoBook',
#
    # 'general-welcome',
    # 'general-greet',
    # 'general-bye',
    # 'general-reqmore',


class UserAct(object):
    """
    :cvar INFORM_TYPE: yes
    :cvar ASK_INFO: no
    :cvar MAKE_RESERVATION: Is it going to rain?
    :cvar ANYTHING_ELSE: I like Chinese food.
    :cvar GOODBYE: find me a place to eat.
    """
    INFORM_TYPE = "inform_type"
    INFORM_TYPE_CHANGE = "inform_type_change"
    ASK_INFO = "ask_info"
    MAKE_RESERVATION = "make_reservation"
    MAKE_RESERVATION_CHANGE_TIME = "make_reservation_change_time"
    ANYTHING_ELSE = "anything_else"
    GOODBYE = "goodbye"

class ConvAct(object):
    START_CONV = "start_conv"
    END_CONV = "end_conv"


class BaseSysSlot(object):
    """
    :cvar DEFAULT: the db entry
    :cvar PURPOSE: what's the purpose of the system
    """

    PURPOSE = "#purpose"
    DEFAULT = "#default"


class BaseUsrSlot(object):
    """
    :cvar NEED: what user want
    :cvar HAPPY: if user is satisfied about system's results
    :cvar AGAIN: the user rephrase the same sentence.
    :cvar SELF_CORRECT: the user correct itself.
    """
    NEED = "#need"
    HAPPY = "#happy"
    AGAIN = "#again"
    SELF_CORRECT = "#self_correct"
