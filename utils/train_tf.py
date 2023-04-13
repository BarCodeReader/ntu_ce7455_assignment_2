import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from .model import  generate_square_subsequent_mask
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


def train(
        input_tensor,
        target_tensor,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        criterion,
        device,
        max_length=MAX_LENGTH):

    #encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0
    src_mask = generate_square_subsequent_mask(MAX_LENGTH).to(device)
    if input_length != MAX_LENGTH:
        src_mask = src_mask[:input_length, :input_length]

    encoder_output = encoder(input_tensor, src_mask) # torch.Size([8, 1, 512])
    for ei in range(input_length):
        encoder_outputs[ei] = encoder_output[ei][0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = decoder.initHidden()

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
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
    learning_rate=0.01
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
                device
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
