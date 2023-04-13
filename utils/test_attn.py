import random

import numpy as np
import torch
from torchmetrics.text.rouge import ROUGEScore

from .constants import *
from .train import tensorFromSentence

rouge = ROUGEScore()


def evaluate(
    encoder,
    decoder,
    sentence,
    device,
    input_lang,
    output_lang,
    max_length=MAX_LENGTH
):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device=device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateRandomly(
    encoder, decoder, pairs, device, input_lang, output_lang, logger, n=10
):
    for _ in range(n):
        pair = random.choice(pairs)
        logger.info(f"> {pair[0]}")
        logger.info(f"= {pair[1]}")
        output_words = evaluate(
            encoder,
            decoder,
            pair[0],
            device,
            input_lang=input_lang,
            output_lang=output_lang
        )
        output_sentence = " ".join(output_words)
        logger.info(f"< {output_sentence}")


def test(encoder, decoder, testing_pairs, device, input_lang, output_lang, logger):
    input = []
    gt = []
    predict = []
    metric_score = {
        "rouge1_fmeasure": [],
        "rouge1_precision": [],
        "rouge1_recall": [],
        "rouge2_fmeasure": [],
        "rouge2_precision": [],
        "rouge2_recall": [],
    }
    from tqdm import tqdm

    for i in tqdm(range(len(testing_pairs))):
        pair = testing_pairs[i]
        output_words = evaluate(
            encoder,
            decoder,
            pair[0],
            device,
            input_lang=input_lang,
            output_lang=output_lang
        )
        output_sentence = " ".join(output_words)

        input.append(pair[0])
        gt.append(pair[1])
        predict.append(output_sentence)

        try:
            rs = rouge(output_sentence, pair[1])
        except:
            continue
        metric_score["rouge1_fmeasure"].append(rs["rouge1_fmeasure"])
        metric_score["rouge1_precision"].append(rs["rouge1_precision"])
        metric_score["rouge1_recall"].append(rs["rouge1_recall"])
        metric_score["rouge2_fmeasure"].append(rs["rouge2_fmeasure"])
        metric_score["rouge2_precision"].append(rs["rouge2_precision"])
        metric_score["rouge2_recall"].append(rs["rouge2_recall"])

    metric_score["rouge1_fmeasure"] = np.array(metric_score["rouge1_fmeasure"]).mean()
    metric_score["rouge1_precision"] = np.array(metric_score["rouge1_precision"]).mean()
    metric_score["rouge1_recall"] = np.array(metric_score["rouge1_recall"]).mean()
    metric_score["rouge2_fmeasure"] = np.array(metric_score["rouge2_fmeasure"]).mean()
    metric_score["rouge2_precision"] = np.array(metric_score["rouge2_precision"]).mean()
    metric_score["rouge2_recall"] = np.array(metric_score["rouge2_recall"]).mean()

    logger.info("=== Evaluation score - Rouge score ===")
    logger.info(f"Rouge1 fmeasure:\t{metric_score['rouge1_fmeasure']}")
    logger.info(f"Rouge1 precision:\t{metric_score['rouge1_precision']}")
    logger.info(f"Rouge1 recall:  \t{metric_score['rouge1_recall']}")
    logger.info(f"Rouge2 fmeasure:\t{metric_score['rouge2_fmeasure']}")
    logger.info(f"Rouge2 precision:\t{metric_score['rouge2_precision']}")
    logger.info(f"Rouge2 recall:  \t{metric_score['rouge2_recall']}")
    return input, gt, predict, metric_score
