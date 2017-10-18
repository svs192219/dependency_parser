# models.py

from utils import *
from adagrad_trainer import *
from treedata import *
import numpy as np
import random


# Greedy parsing model. This model treats shift/reduce decisions as a multiclass classification problem.
class GreedyModel(object):
    def __init__(self, feature_indexer, feature_weights):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):
        state = initial_parser_state(len(sentence))
        while not state.is_finished():
            label_indexer = get_label_indexer()
            prob = np.zeros((len(label_indexer)))
            for label_idx in range(0, len(label_indexer)):
                prob[label_idx] = score_indexed_features(extract_features(self.feature_indexer, sentence, state, label_indexer.get_object(label_idx), False), self.feature_weights)
            decision = label_indexer.get_object(np.argmax(prob))
            if state.stack_len() < 2:
                state = state.shift()
            elif decision == "L" and state.stack_two_back() != -1:
                state = state.left_arc()
            elif decision == "R" and not(state.stack_two_back() == -1 and state.buffer_len() != 0):
                state = state.right_arc()
            elif state.buffer_len() > 0:
                state = state.shift()
            else:
                state = state.right_arc()
        return ParsedSentence(sentence.tokens, state.get_dep_objs(len(sentence)))

class EagerGreedyModel(object):
    def __init__(self, feature_indexer, feature_weights):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):
        state = initial_parser_state(len(sentence))
        while not state.is_eager_finished():
            label_indexer = get_eager_label_indexer()
            prob = np.zeros((len(label_indexer)))
            for label_idx in range(0, len(label_indexer)):
                prob[label_idx] = score_indexed_features(extract_features(self.feature_indexer, sentence, state, label_indexer.get_object(label_idx), False), self.feature_weights)
            decision = label_indexer.get_object(np.argmax(prob))
            legal_transitions = legal(state)
            # print('State : ', state)
            # print('Legal : ', legal_transitions)
            # print('Decision : ', decision)
            if decision in legal_transitions:
                # print('Taking action : ', decision)
                state = state.take_eager_action(decision)
            else:
                # print('Taking action : ', legal_transitions[0])
                state = state.take_eager_action(legal_transitions[0])
        return ParsedSentence(sentence.tokens, state.get_eager_dep_objs(len(sentence)))


def legal(state):
    legal_transitions = ["LA", "RA", "RE", "SH"]

    if state.buffer_len() == 0:
        return []

    stack_top = state.stack_head()
    if stack_top == -1:
        if "RE" in legal_transitions: legal_transitions.remove("RE")
        if "LA" in legal_transitions: legal_transitions.remove("LA")

    if stack_top != -1:
        if stack_top not in state.deps.keys():
            if "RE" in legal_transitions: legal_transitions.remove("RE")
        else:
            if "LA" in legal_transitions: legal_transitions.remove("LA")
    return legal_transitions



# Beam-search-based global parsing model. Shift/reduce decisions are still modeled with local features, but scores are
# accumulated over the whole sequence of decisions to give a "global" decision.
class BeamedModel(object):
    def __init__(self, feature_indexer, feature_weights, beam_size=1):
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.beam_size = beam_size
        # TODO: Modify or add arguments as necessary

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):
        curr_beam = Beam(self.beam_size)
        start_state = initial_parser_state(len(sentence))
        curr_beam.add(start_state, 0)
        for _ in range(0, 2*len(sentence)):
            curr_beam = compute_successorBeam(sentence, curr_beam, self.feature_indexer, self.feature_weights)
        final_state = curr_beam.head()
        return ParsedSentence(sentence.tokens, final_state.get_dep_objs(len(sentence)))


def compute_successorBeam(sentence, curr_beam, feature_indexer, feature_weights):
    next_beam = Beam(curr_beam.size)
    for state, score in curr_beam.get_elts_and_scores():
        label_indexer = get_label_indexer()
        for label_idx in range(0, len(label_indexer)):
            action = label_indexer.get_object(label_idx)
            if is_action_legal(state, action):
                next_score = score + score_indexed_features(extract_features(feature_indexer, sentence, state, action, False), feature_weights)
                next_beam.add(state.take_action(action), next_score)
    return next_beam

def is_action_legal(state, action):
    if action == "S":
        return state.buffer_len() > 0
    elif action == "L":
        return state.stack_len() >= 2 and state.stack_two_back() != -1
    elif action == "R":
        return state.stack_len() >= 2 and not(state.stack_two_back() == -1 and state.buffer_len() != 0)
    return False

# Stores state of a shift-reduce parser, namely the stack, buffer, and the set of dependencies that have
# already been assigned. Supports various accessors as well as the ability to create new ParserStates
# from left_arc, right_arc, and shift.
class ParserState(object):
    # stack and buffer are lists of indices
    # The stack is a list with the top of the stack being the end
    # The buffer is a list with the first item being the front of the buffer (next word)
    # deps is a dictionary mapping *child* indices to *parent* indices
    # (this is the one-to-many map; parent-to-child doesn't work in map-like data structures
    # without having the values be lists)
    def __init__(self, stack, buffer, deps):
        self.stack = stack
        self.buffer = buffer
        self.deps = deps

    def __repr__(self):
        return repr(self.stack) + " " + repr(self.buffer) + " " + repr(self.deps)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.stack == other.stack and self.buffer == other.buffer and self.deps == other.deps
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def stack_len(self):
        return len(self.stack)

    def buffer_len(self):
        return len(self.buffer)

    def is_legal(self):
        return self.stack[0] == -1

    def is_finished(self):
        return len(self.buffer) == 0 and len(self.stack) == 1

    def is_eager_finished(self):
        return len(self.buffer) == 0

    def buffer_head(self):
        return self.get_buffer_word_idx(0)

    # Returns the buffer word at the given index
    def get_buffer_word_idx(self, index):
        if index >= len(self.buffer):
            raise Exception("Can't take the " + repr(index) + " word from the buffer of length " + repr(len(self.buffer)) + ": " + repr(self))
        return self.buffer[index]

    # Returns True if idx has all of its children attached already, False otherwise
    def is_complete(self, idx, parsed_sentence):
        _is_complete = True
        for child in xrange(0, len(parsed_sentence)):
            if parsed_sentence.get_parent_idx(child) == idx and (child not in self.deps.keys() or self.deps[child] != idx):
                _is_complete = False
        return _is_complete

    def stack_head(self):
        if len(self.stack) < 1:
            raise Exception("Can't go one back in the stack if there are no elements: " + repr(self))
        return self.stack[-1]

    def stack_two_back(self):
        if len(self.stack) < 2:
            raise Exception("Can't go two back in the stack if there aren't two elements: " + repr(self))
        return self.stack[-2]

    # Returns a new ParserState that is the result of taking the given action.
    # action is a string, either "L", "R", or "S"
    def take_action(self, action):
        if action == "L":
            return self.left_arc()
        elif action == "R":
            return self.right_arc()
        elif action == "S":
            return self.shift()
        else:
            raise Exception("No implementation for action " + action)

    def take_eager_action(self, action):
        if action == "LA":
            return self.eager_left()
        elif action == "RA":
            return self.eager_right()
        elif action == "RE":
            return self.eager_reduce()
        else:
            return self.eager_shift()

    # Returns a new ParserState that is the result of applying left arc to the current state. May crash if the
    # preconditions for left arc aren't met.
    def left_arc(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_two_back(): self.stack_head()})
        new_stack = list(self.stack[0:-2])
        new_stack.append(self.stack_head())
        return ParserState(new_stack, self.buffer, new_deps)

    # Returns a new ParserState that is the result of applying right arc to the current state. May crash if the
    # preconditions for right arc aren't met.
    def right_arc(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_head(): self.stack_two_back()})
        new_stack = list(self.stack[0:-1])
        return ParserState(new_stack, self.buffer, new_deps)

    # Returns a new ParserState that is the result of applying shift to the current state. May crash if the
    # preconditions for right arc aren't met.
    def shift(self):
        new_stack = list(self.stack)
        new_stack.append(self.buffer_head())
        return ParserState(new_stack, self.buffer[1:], self.deps)

    def eager_left(self):
        new_deps = dict(self.deps)
        new_deps.update({self.stack_head(): self.buffer_head()})
        new_stack = list(self.stack[0:-1])
        return ParserState(new_stack, self.buffer, new_deps)

    def eager_right(self):
        new_deps = dict(self.deps)
        new_deps.update({self.buffer_head(): self.stack_head()})
        new_stack = list(self.stack)
        new_stack.append(self.buffer_head())
        return ParserState(new_stack, self.buffer[1:], new_deps)

    def eager_reduce(self):
        new_deps = dict(self.deps)
        return ParserState(self.stack[0:-1], self.buffer, new_deps)

    def eager_shift(self):
        new_stack = list(self.stack)
        new_stack.append(self.buffer_head())
        return ParserState(new_stack, self.buffer[1:], self.deps)


    # Return the Dependency objects corresponding to the dependencies added so far to this ParserState
    def get_dep_objs(self, sent_len):
        dep_objs = []
        for i in xrange(0, sent_len):
            dep_objs.append(Dependency(self.deps[i], "?"))
        return dep_objs

    def get_eager_dep_objs(self, sent_len):
        dep_objs = []
        for i in xrange(0, sent_len):
            if i not in self.deps:
                dep_objs.append(Dependency(-1, "?"))
            else:
                dep_objs.append(Dependency(self.deps[i], "?"))
        return dep_objs


# Returns an initial ParserState for a sentence of the given length. Note that because the stack and buffer
# are maintained as indices, knowing the words isn't necessary.
def initial_parser_state(sent_len):
    return ParserState([-1], range(0, sent_len), {})


# Returns an indexer for the three actions so you can iterate over them easily.
def get_label_indexer():
    label_indexer = Indexer()
    label_indexer.get_index("S")
    label_indexer.get_index("L")
    label_indexer.get_index("R")
    return label_indexer

def get_eager_label_indexer():
    label_indexer = Indexer()
    label_indexer.get_index("LA")
    label_indexer.get_index("RA")
    label_indexer.get_index("RE")
    label_indexer.get_index("SH")
    return label_indexer    


# Returns a GreedyModel trained over the given treebank.
def train_greedy_model(parsed_sentences):
    # feature_cache[sentence_idx][idx_gold_sequence][decision_idx] -> features
    feature_cache = []
    feature_indexer = Indexer()
    label_indexer = get_label_indexer()
    decision_sequences = []
    for parsed_sentence in parsed_sentences:
        (decisions, states) = get_decision_sequence(parsed_sentence)
        decision_sequences.append((decisions, states))
        cache = [[] for i in range(0, len(decisions))]
        for seq_idx in range(0, len(decisions)):
            for label_idx in range(0, len(label_indexer)):
                cache[seq_idx].append(extract_features(feature_indexer, parsed_sentence, states[seq_idx], label_indexer.get_object(label_idx), True))
        feature_cache.append(cache)

    # training
    feature_weights = np.random.rand((len(feature_indexer)))
    model = GreedyModel(feature_indexer, feature_weights)
    epochs = 15
    lr = 0.01
    lamb = 0.1
    for epoch in range(0, epochs):
        print("Epoch : %d" % (epoch+1))
        # gradient = Counter() 
        for sentence_idx in range(0, len(parsed_sentences)):
            for seq_idx in range(0, len(decision_sequences[sentence_idx][0])):

                gold_label_idx = label_indexer.get_index(decision_sequences[sentence_idx][0][seq_idx])
                max_idx = -1
                slack = -1
                for label_idx in range(0, len(label_indexer)):
                    tmp = score_indexed_features(feature_cache[sentence_idx][seq_idx][label_idx], model.feature_weights)
                    if label_idx != gold_label_idx:
                        tmp += 1
                    if tmp > slack:
                        max_idx = label_idx
                        slack = tmp

                gradient = Counter()
                gradient.increment_all(feature_cache[sentence_idx][seq_idx][max_idx], 1.0)
                gradient.increment_all(feature_cache[sentence_idx][seq_idx][gold_label_idx], -1.0)

                for weight_idx in gradient.keys():
                    model.feature_weights[weight_idx] -= (lr * gradient.get_count(weight_idx)) 
                gradient = Counter()

    return model

def zero_cost_left(state, parsed_sentence):
    if state.stack_len() == 0 or state.buffer_len() == 0:
        return False
    stack_top = state.stack_head()
    
    for elem in state.buffer[1:]:
        if parsed_sentence.get_parent_idx(elem) == stack_top:
            return False
        if stack_top != -1 and parsed_sentence.get_parent_idx(stack_top) == elem:
            return False

    return True

def zero_cost_right(state, parsed_sentence):
    if state.stack_len() == 0 or state.buffer_len() == 0:
        return False
    buffer_head = state.buffer_head()

    for elem in state.stack[0:-1]:
        if elem != -1 and  parsed_sentence.get_parent_idx(elem) == buffer_head:
            return False
        if parsed_sentence.get_parent_idx(buffer_head) == elem:
            return False

    for elem in state.buffer[1:]:
        if parsed_sentence.get_parent_idx(buffer_head) == elem:
            return False

    return True

def zero_cost_reduce(state, parsed_sentence):
    if state.buffer_len() < 1:
        return False
    stack_top = state.stack_head()
    if stack_top == -1:
        return False
    for elem in state.buffer:
        if parsed_sentence.get_parent_idx(elem) == stack_top:
            return False
    return True

def zero_cost_shift(state, parsed_sentence):
    if state.buffer_len() < 1:
        return False
    buffer_head = state.buffer_head()
    for elem in state.stack:
        if elem != -1 and parsed_sentence.get_parent_idx(elem) == buffer_head:
            return False
        if parsed_sentence.get_parent_idx(buffer_head) == elem:
            return False
    return True

def zero_cost_transitions(state, parsed_sentence, legal_transitions):
    zero_cost = []
    if zero_cost_left(state, parsed_sentence) and "LA" in legal_transitions:
        zero_cost.append("LA")
    elif zero_cost_right(state, parsed_sentence) and "RA" in legal_transitions:
        zero_cost.append("RA")
    elif zero_cost_reduce(state, parsed_sentence) and "RE" in legal_transitions:
        zero_cost.append("RE")
    elif zero_cost_shift(state, parsed_sentence) and "SH" in legal_transitions:
        zero_cost.append("SH")
    return zero_cost

def choose_next_amb(epoch, pred_t, zero_cost_t):
    if pred_t in zero_cost_t:
        return pred_t
    else:
        return random.choice(zero_cost_t)

def choose_next_exp(epoch, pred_t, zero_cost_t):
    if epoch > 1 and np.random.rand() > 0.1:
        return pred_t
    else:
        return choose_next_amb(epoch, pred_t, zero_cost_t)

def train_eager_greedy_model(parsed_sentences, useDynamic=False):
    # feature_cache[sentence_idx][idx_gold_sequence][decision_idx] -> features
    feature_cache = []
    feature_indexer = Indexer()
    label_indexer = get_eager_label_indexer()
    decision_sequences = []
    for parsed_sentence in parsed_sentences:
        (decisions, states) = get_eager_decision_sequence(parsed_sentence)
        decision_sequences.append((decisions, states))
        cache = [[] for i in range(0, len(decisions))]
        for seq_idx in range(0, len(decisions)):
            for label_idx in range(0, len(label_indexer)):
                cache[seq_idx].append(extract_features(feature_indexer, parsed_sentence, states[seq_idx], label_indexer.get_object(label_idx), True))
        feature_cache.append(cache)

    # training
    feature_weights = np.zeros((len(feature_indexer)))
    avg_feature_weights = np.zeros((len(feature_indexer)))
    epochs = 10
    lr = 1
    step = epochs * len(parsed_sentences)
    total_iters = step

    for epoch in range(0, epochs):
        print("Epoch : %d" % (epoch+1))
        for sentence_idx in range(0, len(parsed_sentences)):
            if useDynamic:
                state = initial_parser_state(len(parsed_sentences[sentence_idx]))
                seq_idx = 0
                while not state.is_eager_finished():
                    legal_t = legal(state)
                    legal_indices = []
                    for t in legal_t:
                        legal_indices.append(label_indexer.index_of(t))
                    zero_cost_t = zero_cost_transitions(state, parsed_sentences[sentence_idx], legal_t)
                    zero_cost_indices = []
                    for t in zero_cost_t:
                        zero_cost_indices.append(label_indexer.index_of(t))
                    pred_scores = []
                    for label_idx in range(0, len(label_indexer)):
                        pred_scores.append(score_indexed_features(feature_cache[sentence_idx][seq_idx][label_idx], feature_weights))

                    pred_t_idx = max(legal_indices, key=lambda p:pred_scores[p])
                    if len(zero_cost_t) == 0:
                        state = state.take_eager_action(label_indexer.get_object(pred_t_idx))
                        continue
                    zero_t_idx = max(zero_cost_indices, key=lambda p:pred_scores[p])

                    if pred_t_idx != zero_t_idx:
                        gradient = Counter()
                        gradient.increment_all(feature_cache[sentence_idx][seq_idx][pred_t_idx], 1.0)
                        gradient.increment_all(feature_cache[sentence_idx][seq_idx][zero_t_idx], -1.0)

                        for weight_idx in gradient.keys():
                            feature_weights[weight_idx] -= (lr * gradient.get_count(weight_idx))
                            avg_feature_weights[weight_idx] -= (step * lr * gradient.get_count(weight_idx))/total_iters
                        gradient = Counter()

                    action_idx = choose_next_amb(epoch, pred_t_idx, zero_cost_indices)
                    state = state.take_eager_action(label_indexer.get_object(action_idx))

            else:
                for seq_idx in range(0, len(decision_sequences[sentence_idx][0])):
                    gold_label_idx = label_indexer.get_index(decision_sequences[sentence_idx][0][seq_idx])
                    max_idx = -1
                    slack = -1
                    for label_idx in range(0, len(label_indexer)):
                        tmp = score_indexed_features(feature_cache[sentence_idx][seq_idx][label_idx], feature_weights)
                        if tmp > slack:
                            max_idx = label_idx
                            slack = tmp

                    gradient = Counter()
                    gradient.increment_all(feature_cache[sentence_idx][seq_idx][max_idx], 1.0)
                    gradient.increment_all(feature_cache[sentence_idx][seq_idx][gold_label_idx], -1.0)

                    for weight_idx in gradient.keys():
                        feature_weights[weight_idx] -= (lr * gradient.get_count(weight_idx))
                        avg_feature_weights[weight_idx] -= (step * lr * gradient.get_count(weight_idx))/total_iters
                    gradient = Counter()
            step -= 1

    model = EagerGreedyModel(feature_indexer, feature_weights)
    return model

class EagerDynamicModel(object):
    def __init__(self, classifier):
        self.classifier = classifier
        self.train_tick = 0
        self.train_last_tick = defaultdict(lambda: 0)
        self.train_totals = defaultdict(lambda: 0)

    def update(self, pred, truth, features):
        def update_feature_label(label, f, value):
            w = 0
            try:
                w = self.classifier.weights[f][label]
            except KeyError:
                if f not in self.classifier.weights:
                    self.classifier.weights[f] = defaultdict(lambda: 0)

            delta = self.train_tick - self.train_last_tick[(f, label)]
            self.train_totals[(f, label)] += delta * w
            self.classifier.weights[f][label] += value
            self.train_last_tick[(f, label)] = self.train_tick

        self.train_tick += 1
        for f, _ in features.items():
            update_feature_label(truth, f, 1.0)
            update_feature_label(pred, f, -1.0)


    def avg_weights(self):
        for f in self.classifier.weights:
            for l in self.classifier.weights[f]:
                total = self.train_totals[(f, l)]
                delta = self.train_tick - self.train_last_tick[(f, l)]
                total += delta * self.classifier.weights[f][l]
                avg = round(total / float(self.train_tick))
                if avg:
                    self.classifier.weights[f][l] = avg

    # Given a ParsedSentence, returns a new ParsedSentence with predicted dependency information.
    # The new ParsedSentence should have the same tokens as the original and new dependencies constituting
    # the predicted parse.
    def parse(self, sentence):
        state = initial_parser_state(len(sentence))
        while not state.is_eager_finished():
            label_indexer = get_eager_label_indexer()
            feats = extract_feature_dict(sentence, state)
            pred_scores = self.classifier.score(feats)
            legal_transitions = legal(state)
            legal_indices = []
            for t in legal_transitions:
                legal_indices.append(label_indexer.index_of(t))
            decision_idx = max(legal_indices, key=lambda p:pred_scores[p])
            decision = label_indexer.get_object(decision_idx)
            state = state.take_eager_action(decision)
        return ParsedSentence(sentence.tokens, state.get_eager_dep_objs(len(sentence)))


class Classifier(object):
    def __init__(self, weights, labels):
        self.weights = weights
        self.labels = labels

    def score(self, features):
        scores = [0, 0, 0, 0]
        for feat, value in features.items():
            if value == 0:
                continue
            if feat not in self.weights:
                continue

            for label, weight in self.weights[feat].items():
                scores[label] += weight * value
        return scores


def train_eager_dynamic_model(parsed_sentences):
    label_indexer = get_eager_label_indexer()
    epochs = 10
    model = EagerDynamicModel(Classifier({}, [0, 1, 2, 3]))

    for epoch in range(0, epochs):
        print("Epoch : %d" % (epoch+1))
        for parsed_sentence in parsed_sentences:
            state = initial_parser_state(len(parsed_sentence))

            while not state.is_eager_finished():
                legal_t = legal(state)
                legal_indices = []
                for t in legal_t:
                    legal_indices.append(label_indexer.index_of(t))
                zero_cost_t = zero_cost_transitions(state, parsed_sentence, legal_t)
                zero_cost_indices = []
                for t in zero_cost_t:
                    zero_cost_indices.append(label_indexer.index_of(t))

                features = extract_feature_dict(parsed_sentence, state)
                pred_scores = model.classifier.score(features)

                pred_t_idx = max(legal_indices, key=lambda p:pred_scores[p])
                if len(zero_cost_t) == 0:
                    state = state.take_eager_action(label_indexer.get_object(pred_t_idx))
                    continue
                zero_t_idx = max(zero_cost_indices, key=lambda p:pred_scores[p])

                if pred_t_idx != zero_t_idx:
                    model.update(pred_t_idx, zero_t_idx, features)

                action_idx = choose_next_amb(epoch, pred_t_idx, zero_cost_indices)
                state = state.take_eager_action(label_indexer.get_object(action_idx))

    model.avg_weights()
    return model


# Returns a BeamedModel trained over the given treebank.
def train_beamed_model(parsed_sentences):
    feature_cache = []
    feature_indexer = Indexer()
    label_indexer = get_label_indexer()
    decision_sequences = []
    for parsed_sentence in parsed_sentences:
        (decisions, states) = get_decision_sequence(parsed_sentence)
        decision_sequences.append((decisions, states))
        cache = [[] for i in range(0, len(decisions))]
        for seq_idx in range(0, len(decisions)):
            for label_idx in range(0, len(label_indexer)):
                cache[seq_idx].append(extract_features(feature_indexer, parsed_sentence, states[seq_idx], label_indexer.get_object(label_idx), True))
        feature_cache.append(cache)

    # training
    feature_weights = np.zeros((len(feature_indexer)))
    avg_feature_weights = np.zeros((len(feature_indexer)))
    epochs = 1
    lr = 1
    lamb = 0.1
    beam_size = 5
    step = epochs * len(parsed_sentences)
    total_iters = step
    
    for epoch in range(0, epochs):
        print("Epoch : %d" % (epoch+1))
        for sentence_idx in range(0, len(parsed_sentences)):
            sentence = parsed_sentences[sentence_idx]
            start_state = initial_parser_state(len(sentence))
            curr_beam = Beam(beam_size)
            curr_beam.add((start_state, []), 0)
            gradient = Counter()
            gold_feats = []
            max_feats = []
            early = False
            for idx in range(0, len(decision_sequences[sentence_idx][0])):
                next_beam = Beam(curr_beam.size)
                gold_state = decision_sequences[sentence_idx][1][idx]
                gold_action = decision_sequences[sentence_idx][0][idx]
                for (state, feats), score in curr_beam.get_elts_and_scores():
                    label_indexer = get_label_indexer()
                    for label_idx in range(0, len(label_indexer)):
                        action = label_indexer.get_object(label_idx)
                        if is_action_legal(state, action):
                            tmp = extract_features(feature_indexer, sentence, state, action, False)
                            next_score = score + score_indexed_features(tmp, feature_weights)
                            new_feats = list(feats)
                            new_feats.append(tmp)
                            next_beam.add((state.take_action(action), new_feats), next_score)
                gold_label_idx = label_indexer.get_index(gold_action)
                gold_feats.append(feature_cache[sentence_idx][idx][gold_label_idx])

                # Early update
                present = False
                for state, _ in next_beam.get_elts():
                    if decision_sequences[sentence_idx][1][idx+1] == state:
                        present = True
                        break
                if not present:
                    early = True
                    curr_beam = next_beam
                    break
                curr_beam = next_beam

            max_feats = list(curr_beam.head()[1])

            # apply gradient
            gradient = Counter()
            for feats in max_feats:
                gradient.increment_all(feats, 1.0)
            for feats in gold_feats:
                gradient.increment_all(feats, -1.0)

            for weight_idx in gradient.keys():
                feature_weights[weight_idx] -= (lr * gradient.get_count(weight_idx))
                avg_feature_weights[weight_idx] -= (step * lr * gradient.get_count(weight_idx))/total_iters
            gradient = Counter()
            step -= 1

    model = BeamedModel(feature_indexer, feature_weights, beam_size=beam_size)
    return model


# Extract features for the given decision in the given parser state. Features look at the top of the
# stack and the start of the buffer. Note that this isn't in any way a complete feature set -- play around with
# more of your own!
def extract_features(feat_indexer, sentence, parser_state, decision, add_to_indexer):
    feats = []
    sos_tok = Token("<s>", "<S>", "<S>")
    root_tok = Token("<root>", "<ROOT>", "<ROOT>")
    eos_tok = Token("</s>", "</S>", "</S>")
    if parser_state.stack_len() >= 1:
        head_idx = parser_state.stack_head()
        stack_head_tok = sentence.tokens[head_idx] if head_idx != -1 else root_tok
        if parser_state.stack_len() >= 2:
            two_back_idx = parser_state.stack_two_back()
            stack_two_back_tok = sentence.tokens[two_back_idx] if two_back_idx != -1 else root_tok
        else:
            stack_two_back_tok = sos_tok
    else:
        stack_head_tok = sos_tok
        stack_two_back_tok = sos_tok
    buffer_first_tok = sentence.tokens[parser_state.get_buffer_word_idx(0)] if parser_state.buffer_len() >= 1 else eos_tok
    buffer_second_tok = sentence.tokens[parser_state.get_buffer_word_idx(1)] if parser_state.buffer_len() >= 2 else eos_tok
    # Shortcut for adding features
    def add_feat(feat):
        maybe_add_feature(feats, feat_indexer, add_to_indexer, feat)
    add_feat(decision + ":S0Word=" + stack_head_tok.word)
    add_feat(decision + ":S0Pos=" + stack_head_tok.pos)
    add_feat(decision + ":S0CPos=" + stack_head_tok.cpos)
    add_feat(decision + ":S1Word=" + stack_two_back_tok.word)
    add_feat(decision + ":S1Pos=" + stack_two_back_tok.pos)
    add_feat(decision + ":S1CPos=" + stack_two_back_tok.cpos)
    add_feat(decision + ":B0Word=" + buffer_first_tok.word)
    add_feat(decision + ":B0Pos=" + buffer_first_tok.pos)
    add_feat(decision + ":B0CPos=" + buffer_first_tok.cpos)
    add_feat(decision + ":B1Word=" + buffer_second_tok.word)
    add_feat(decision + ":B1Pos=" + buffer_second_tok.pos)
    add_feat(decision + ":B1CPos=" + buffer_second_tok.cpos)
    add_feat(decision + ":S1S0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos)
    add_feat(decision + ":S0B0Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S1B0Pos=" + stack_two_back_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B1Pos=" + stack_head_tok.pos + "&" + buffer_second_tok.pos)
    add_feat(decision + ":B0B1Pos=" + buffer_first_tok.pos + "&" + buffer_second_tok.pos)
    add_feat(decision + ":S0B0WordPos=" + stack_head_tok.word + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B0PosWord=" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S1S0WordPos=" + stack_two_back_tok.word + "&" + stack_head_tok.pos)
    add_feat(decision + ":S1S0PosWord=" + stack_two_back_tok.pos + "&" + stack_head_tok.word)
    add_feat(decision + ":S1S0B0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos + "&" + buffer_first_tok.pos)
    add_feat(decision + ":S0B0B1Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos + "&" + buffer_second_tok.pos)
    return feats

def extract_feature_dict(sentence, parser_state):
    feats = {}
    sos_tok = Token("<s>", "<S>", "<S>")
    root_tok = Token("<root>", "<ROOT>", "<ROOT>")
    eos_tok = Token("</s>", "</S>", "</S>")
    if parser_state.stack_len() >= 1:
        head_idx = parser_state.stack_head()
        stack_head_tok = sentence.tokens[head_idx] if head_idx != -1 else root_tok
        if parser_state.stack_len() >= 2:
            two_back_idx = parser_state.stack_two_back()
            stack_two_back_tok = sentence.tokens[two_back_idx] if two_back_idx != -1 else root_tok
        else:
            stack_two_back_tok = sos_tok
    else:
        stack_head_tok = sos_tok
        stack_two_back_tok = sos_tok
    buffer_first_tok = sentence.tokens[parser_state.get_buffer_word_idx(0)] if parser_state.buffer_len() >= 1 else eos_tok
    buffer_second_tok = sentence.tokens[parser_state.get_buffer_word_idx(1)] if parser_state.buffer_len() >= 2 else eos_tok

    feats[":S0Word=" + stack_head_tok.word] = 1
    feats[":S0Pos=" + stack_head_tok.pos] = 1
    feats[":S0CPos=" + stack_head_tok.cpos] = 1
    feats[":S1Word=" + stack_two_back_tok.word] = 1
    feats[":S1Pos=" + stack_two_back_tok.pos] = 1
    feats[":S1CPos=" + stack_two_back_tok.cpos] = 1
    feats[":B0Word=" + buffer_first_tok.word] = 1
    feats[":B0Pos=" + buffer_first_tok.pos] = 1
    feats[":B0CPos=" + buffer_first_tok.cpos] = 1
    feats[":B1Word=" + buffer_second_tok.word] = 1
    feats[":B1Pos=" + buffer_second_tok.pos] = 1
    feats[":B1CPos=" + buffer_second_tok.cpos] = 1
    feats[":S1S0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos] = 1
    feats[":S0B0Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos] = 1
    feats[":S1B0Pos=" + stack_two_back_tok.pos + "&" + buffer_first_tok.pos] = 1
    feats[":S0B1Pos=" + stack_head_tok.pos + "&" + buffer_second_tok.pos] = 1
    feats[":B0B1Pos=" + buffer_first_tok.pos + "&" + buffer_second_tok.pos] = 1
    feats[":S0B0WordPos=" + stack_head_tok.word + "&" + buffer_first_tok.pos] = 1
    feats[":S0B0PosWord=" + stack_head_tok.pos + "&" + buffer_first_tok.pos] = 1
    feats[":S1S0WordPos=" + stack_two_back_tok.word + "&" + stack_head_tok.pos] = 1
    feats[":S1S0PosWord=" + stack_two_back_tok.pos + "&" + stack_head_tok.word] = 1
    feats[":S1S0B0Pos=" + stack_two_back_tok.pos + "&" + stack_head_tok.pos + "&" + buffer_first_tok.pos] = 1
    feats[":S0B0B1Pos=" + stack_head_tok.pos + "&" + buffer_first_tok.pos + "&" + buffer_second_tok.pos] = 1
    return feats


# Computes the sequence of decisions and ParserStates for a gold-standard sentence using the arc-standard
# transition framework. We use the minimum stack-depth heuristic, namely that
# Invariant: states[0] is the initial state. Applying decisions[i] to states[i] yields states[i+1].
def get_decision_sequence(parsed_sentence):
    decisions = []
    states = []
    state = initial_parser_state(len(parsed_sentence))
    while not state.is_finished():
        if not state.is_legal():
            raise Exception(repr(decisions) + " " + repr(state))
        # Look at whether left-arc or right-arc would add correct arcs
        if len(state.stack) < 2:
            result = "S"
        else:
            # Stack and buffer must both contain at least one thing
            one_back = state.stack_head()
            two_back = state.stack_two_back()
            # -1 is the ROOT symbol, so this forbids attaching the ROOT as a child of anything
            # (passing -1 as an index around causes crazy things to happen so we check explicitly)
            if two_back != -1 and parsed_sentence.get_parent_idx(two_back) == one_back and state.is_complete(two_back, parsed_sentence):
                result = "L"
            # The first condition should never be true, but doesn't hurt to check
            elif one_back != -1 and parsed_sentence.get_parent_idx(one_back) == two_back and state.is_complete(one_back, parsed_sentence):
                result = "R"
            elif len(state.buffer) > 0:
                result = "S"
            else:
                result = "R" # something went wrong, buffer is empty, just do right arcs to finish the tree
        decisions.append(result)
        states.append(state)
        if result == "L":
            state = state.left_arc()
        elif result == "R":
            state = state.right_arc()
        else:
            state = state.shift()
    states.append(state)
    return (decisions, states)


def get_eager_decision_sequence(parsed_sentence):
    decisions = []
    states = []
    state = initial_parser_state(len(parsed_sentence))
    while not state.is_eager_finished():
        if not state.is_legal():
            raise Exception(repr(decisions) + " " + repr(state))
        
        if state.buffer_len() == 0:
            result = "SH"
        else:
            stack_top = state.stack_head()
            buffer_head = state.buffer_head()
            result = ""

            # left arc
            if stack_top in parsed_sentence.heads[buffer_head]:
                result = "LA"
            elif buffer_head in parsed_sentence.heads[stack_top]:
                result = "RA"
            else:
                reducePossible = False
                for dep in parsed_sentence.heads[buffer_head]:
                    if dep < stack_top:
                        reducePossible = True
                if parsed_sentence.get_parent_idx(buffer_head) < stack_top:
                    reducePossible = True
                if reducePossible:
                    result = "RE"
                else:
                    result = "SH"

        decisions.append(result)
        states.append(state)
        if result == "LA":
            state = state.eager_left()
        elif result == "RA":
            state = state.eager_right()
        elif result == "RE":
            state = state.eager_reduce()
        else:
            state = state.eager_shift()
    states.append(state)
    return (decisions, states)

