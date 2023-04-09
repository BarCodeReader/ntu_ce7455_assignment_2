import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from .constants import *

teacher_forcing_ratio = 0.5


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device=device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device=device)
    return (input_tensor, target_tensor)


def auto_regressive_beam_search(
    decoder_model, start_tokens, beam_width, max_len, alpha
):
    end_token = -1
    num_states = len(decoder_model.init_states(1))
    batch_size = start_tokens.shape[0]

    # Initialize the beams
    beams = [
        {
            "tokens": start_tokens,
            "log_prob": 0.0,
            "states": decoder_model.init_states(batch_size),
        }
    ]

    for i in range(max_len):
        # Generate the next set of candidates for each beam
        candidates = []
        for beam in beams:
            # Get the current token and states for the beam
            token = beam["tokens"][:, -1:]
            states = beam["states"]

            # Predict the log probabilities for the next token and update the states
            logits, new_states = decoder_model(token, states)
            log_probs = F.log_softmax(logits.squeeze(1), dim=-1)
            new_states = (new_states[0].detach(), new_states[1].detach())

            # Add the new candidates to the list
            for j in range(beam_width):
                candidate = {
                    "tokens": torch.cat(
                        [
                            beam["tokens"],
                            torch.zeros((batch_size, 1), dtype=torch.long),
                        ],
                        dim=-1,
                    ),
                    "log_prob": beam["log_prob"] + log_probs[:, j],
                    "states": new_states,
                }
                candidates.append(candidate)

        # Keep the top k candidates based on their log probabilities
        candidates = sorted(candidates, key=lambda x: x["log_prob"], reverse=True)[
            :beam_width
        ]
        beams = []

        # Check if any of the beams have reached the end token
        for candidate in candidates:
            if candidate["tokens"][:, -1] == end_token:
                beams.append(candidate)
            else:
                beams.append(
                    {
                        "tokens": candidate["tokens"],
                        "log_prob": candidate["log_prob"],
                        "states": candidate["states"],
                    }
                )

        # If all the beams have reached the end token, stop searching
        if all([beam["tokens"][:, -1] == end_token for beam in beams]):
            break

    # Choose the beam with the highest log probability
    beam = max(beams, key=lambda x: x["log_prob"])
    tokens = beam["tokens"]

    # Apply length normalization
    if alpha != 1.0:
        length = torch.pow(torch.tensor(tokens.shape[1], dtype=torch.float32), alpha)
        log_probs = beam["log_prob"] / length
    else:
        log_probs = beam["log_prob"]

    return tokens, log_probs


def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    device,
    max_length=MAX_LENGTH,
    is_bidirectional=False
):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        if is_bidirectional:
            input = torch.cat(input_tensor[ei], input_tensor[input_length-1-ei])
        else:
            input = input_tensor[ei]
        encoder_output, encoder_hidden = encoder(input, encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden  # last token's hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(
    logger,
    encoder,
    decoder,
    train_pairs,
    epochs,
    tensorboard_log_dir,
    device,
    input_lang,
    output_lang,
    print_every=1000,
    plot_every=100,
    learning_rate=0.01,
):
    tensorboard_writter = SummaryWriter(tensorboard_log_dir)
    start = time.time()
    # plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    iter = 1
    n_iters = len(train_pairs) * epochs

    for epoch in range(epochs):
        logger.info("Epoch: %d/%d" % (epoch, epochs))
        for training_pair in train_pairs:
            training_pair = tensorsFromPair(
                input_lang, output_lang, training_pair, device=device
            )

            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(
                input_tensor,
                target_tensor,
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                criterion,
                device=device,
            )
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                logger.info(
                    "%s (%d %d%%) %.4f"
                    % (
                        timeSince(start, iter / n_iters),
                        iter,
                        iter / n_iters * 100,
                        print_loss_avg,
                    )
                )

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss_total = 0

                tensorboard_writter.add_scalar(
                    "loss_avg", plot_loss_avg, iter + (epoch + 1) * len(train_pairs)
                )

            iter += 1


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))
