import argparse

import torch
from sklearn.model_selection import train_test_split

from utils.constants import *
from utils.data import *
from utils.model import DecoderLSTM, EncoderLSTM
from utils.test import evaluateRandomly, test
from utils.train import trainIters
from utils.utils import register_logger


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="lstm")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--print_every", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tensorboard_dir", type=str, default="tb_logs/t2/gru")

    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    logger = register_logger("experiments_logs/task_2.log")

    with open("experiments_logs/task_2.log", "a+") as file:
        file.write("\n-------------------------------------------\n")

    logger.info(
        f"Starting the experiment for task 2 for {opt.model_type} with {opt.epochs} epochs,"
        + f" the hidden_size is {opt.hidden_size}, lr: {opt.lr}"
    )
    input_lang, output_lang, pairs = prepareData("eng", "fra", True)

    X = [i[0] for i in pairs]
    y = [i[1] for i in pairs]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    train_pairs = list(zip(X_train, y_train))
    test_pairs = list(zip(X_test, y_test))

    if opt.model_type.lower() == "lstm":
        is_bidirectional = False
        encoder1 = EncoderLSTM(input_lang.n_words, opt.hidden_size, device, is_bidirectional=is_bidirectional).to(device)
        decoder1 = DecoderLSTM(opt.hidden_size, output_lang.n_words, device, is_bidirectional=is_bidirectional).to(device)
    else:
        is_bidirectional = True
        encoder1 = EncoderLSTM(input_lang.n_words, opt.hidden_size, device, num_layers=1, is_bidirectional=is_bidirectional).to(device)
        decoder1 = DecoderLSTM(opt.hidden_size, output_lang.n_words, device, num_layers=1, is_bidirectional=is_bidirectional).to(device)

    logger.info("Training")
    trainIters(
        logger=logger,
        encoder=encoder1,
        decoder=decoder1,
        train_pairs=train_pairs,
        epochs=opt.epochs,
        tensorboard_log_dir=opt.tensorboard_dir,
        device=device,
        input_lang=input_lang,
        output_lang=output_lang,
        print_every=opt.print_every,
        learning_rate=opt.lr,
        is_bidirectional=is_bidirectional
    )

    torch.save(encoder1.state_dict(), f"models/task_2/encoder_{opt.model_type}.pt")
    torch.save(decoder1.state_dict(), f"models/task_2/decode_{opt.model_type}r.pt")
    logger.info("Testing")
    evaluateRandomly(
        encoder1,
        decoder1,
        pairs=test_pairs,
        device=device,
        input_lang=input_lang,
        output_lang=output_lang,
        logger=logger,
        is_bidirectional=is_bidirectional
    )
    print('---------------training pair eval result')
    input, gt, predict, score = test(
        encoder1,
        decoder1,
        train_pairs,
        device=device,
        input_lang=input_lang,
        output_lang=output_lang,
        logger=logger,
        is_bidirectional=is_bidirectional
    )
    print('---------------testing pair eval result')
    input, gt, predict, score = test(
        encoder1,
        decoder1,
        test_pairs,
        device=device,
        input_lang=input_lang,
        output_lang=output_lang,
        logger=logger,
        is_bidirectional=is_bidirectional
    )
