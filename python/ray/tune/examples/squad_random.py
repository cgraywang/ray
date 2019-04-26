from __future__ import print_function

import argparse
import random
import os
import logging
import time
import warnings
import collections

import numpy as np
import mxnet as mx
from mxnet import gluon, nd

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from bert_qa_model import BertForQALoss, BertForQA
from bert_qa_dataset import (SQuADTransform, preprocess_dataset)
from bert_qa_evaluate import get_F1_EM, predictions


log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')


parser = argparse.ArgumentParser(description='BERT QA example.'
                                 'We fine-tune the BERT model on SQuAD dataset.')

parser.add_argument('--only_predict',
                    action='store_true',
                    help='Whether to predict only.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.')

parser.add_argument('--bert_dataset',
                    type=str,
                    default='book_corpus_wiki_en_uncased',
                    help='BERT dataset name.'
                    'options are book_corpus_wiki_en_uncased and book_corpus_wiki_en_cased.')

parser.add_argument('--pretrained_bert_parameters',
                    type=str,
                    default=None,
                    help='Pre-trained bert model parameter file. default is None')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--output_dir',
                    type=str,
                    default='./output_dir',
                    help='The output directory where the model params will be written.'
                    ' default is ./output_dir')

parser.add_argument('--epochs',
                    type=int,
                    default=3,
                    help='number of epochs, default is 3')

parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='Batch size. Number of examples per gpu in a minibatch. default is 32')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--optimizer',
                    type=str,
                    default='bertadam',
                    help='optimization algorithm. default is bertadam(mxnet >= 1.5.0.)')

parser.add_argument('--accumulate',
                    type=int,
                    default=None,
                    help='The number of batches for '
                    'gradients accumulation to simulate large batch size. Default is None')

parser.add_argument('--lr',
                    type=float,
                    default=5e-5,
                    help='Initial learning rate. default is 5e-5')

parser.add_argument('--warmup_ratio',
                    type=float,
                    default=0.1,
                    help='ratio of warmup steps that linearly increase learning rate from '
                    '0 to target learning rate. default is 0.1')

parser.add_argument('--log_interval',
                    type=int,
                    default=50,
                    help='report interval. default is 50')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                    'Sequences longer than this will be truncated, and sequences shorter '
                    'than this will be padded. default is 384')

parser.add_argument('--doc_stride',
                    type=int,
                    default=128,
                    help='When splitting up a long document into chunks, how much stride to '
                    'take between chunks. default is 128')

parser.add_argument('--max_query_length',
                    type=int,
                    default=64,
                    help='The maximum number of tokens for the question. Questions longer than '
                    'this will be truncated to this length. default is 64')

parser.add_argument('--n_best_size',
                    type=int,
                    default=20,
                    help='The total number of n-best predictions to generate in the '
                    'nbest_predictions.json output file. default is 20')

parser.add_argument('--max_answer_length',
                    type=int,
                    default=30,
                    help='The maximum length of an answer that can be generated. This is needed '
                    'because the start and end predictions are not conditioned on one another.'
                    ' default is 30')

parser.add_argument('--version_2',
                    action='store_true',
                    help='SQuAD examples whether contain some that do not have an answer.')

parser.add_argument('--null_score_diff_threshold',
                    type=float,
                    default=0.0,
                    help='If null_score - best_non_null is greater than the threshold predict null.'
                    'Typical values are between -1.0 and -5.0. default is 0.0')

parser.add_argument('--gpu',
                    action='store_true',
                    help='whether to use gpu for finetuning')
parser.add_argument('--expname', type=str, default='autoindoor')
parser.add_argument(
    '--reuse_actors', action="store_true", help="reuse actor")
parser.add_argument('--checkpoint_freq', default=20, type=int,
                    help='checkpoint_freq')
parser.add_argument(
    '--checkpoint_at_end', action="store_true", help="checkpoint_at_end")
parser.add_argument('--max_failures', default=20, type=int,
                    help='max_failures')
parser.add_argument(
    '--queue_trials', action="store_true", help="queue_trials")
parser.add_argument(
    '--with_server', action="store_true", help="with_server")
parser.add_argument(
    '--num_samples',
    type=int,
    default=50,
    metavar='N',
    help='number of samples')
parser.add_argument('--scheduler', type=str, default='fifo')
parser.add_argument('--stop_accuracy', default=0.94, type=float,
                    help='stop_accuracy')
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of preprocessing workers')
args = parser.parse_args()


def train_indoor(args, config, reporter):
    vars(args).update(config)
    np.random.seed(args.seed)
    random.seed(args.seed)
    mx.random.seed(args.seed)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fh = logging.FileHandler(os.path.join(
        args.output_dir, 'finetune_squad.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    log.addHandler(console)
    log.addHandler(fh)

    log.info(args)

    model_name = args.bert_model
    dataset_name = args.bert_dataset
    only_predict = args.only_predict
    model_parameters = args.model_parameters
    pretrained_bert_parameters = args.pretrained_bert_parameters
    lower = args.uncased

    epochs = args.epochs
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    lr = args.lr
    ctx = mx.cpu() if not args.gpu else mx.gpu()

    accumulate = args.accumulate
    log_interval = args.log_interval * accumulate if accumulate else args.log_interval
    if accumulate:
        log.info('Using gradient accumulation. Effective batch size = {}'.
                 format(accumulate * batch_size))

    optimizer = args.optimizer
    warmup_ratio = args.warmup_ratio

    version_2 = args.version_2
    null_score_diff_threshold = args.null_score_diff_threshold

    max_seq_length = args.max_seq_length
    doc_stride = args.doc_stride
    max_query_length = args.max_query_length
    n_best_size = args.n_best_size
    max_answer_length = args.max_answer_length

    if max_seq_length <= max_query_length + 3:
        raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                         '(%d) + 3' % (max_seq_length, max_query_length))

    bert, vocab = nlp.model.get_model(
        name=model_name,
        dataset_name=dataset_name,
        pretrained=not model_parameters and not pretrained_bert_parameters,
        ctx=ctx,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False)

    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
        nlp.data.batchify.Pad(axis=0, pad_val=vocab[vocab.padding_token]),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'),
        nlp.data.batchify.Stack('float32'))

    berttoken = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

    net = BertForQA(bert=bert)
    if pretrained_bert_parameters and not model_parameters:
        bert.load_parameters(pretrained_bert_parameters, ctx=ctx,
                             ignore_extra=True)
    if not model_parameters:
        net.span_classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    else:
        net.load_parameters(model_parameters, ctx=ctx)
    net.hybridize(static_alloc=True)

    loss_function = BertForQALoss()
    loss_function.hybridize(static_alloc=True)

    """Training function."""
    log.info('Loader Train data...')
    if version_2:
        train_data = SQuAD('train', version='2.0')
    else:
        train_data = SQuAD('train', version='1.1')
    log.info('Number of records in Train data:{}'.format(len(train_data)))

    train_data_transform, _ = preprocess_dataset(
        train_data, SQuADTransform(
            berttoken,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=True,
            is_training=True))
    log.info('The number of examples after preprocessing:{}'.format(
        len(train_data_transform)))

    train_dataloader = mx.gluon.data.DataLoader(
        train_data_transform, batchify_fn=batchify_fn,
        batch_size=batch_size, num_workers=args.num_workers, shuffle=True)


    """Evaluate the model on validation dataset.
    """
    log.info('Loader dev data...')
    if version_2:
        dev_data = SQuAD('dev', version='2.0')
    else:
        dev_data = SQuAD('dev', version='1.1')
    log.info('Number of records in Train data:{}'.format(len(dev_data)))

    dev_dataset = dev_data.transform(
        SQuADTransform(
            berttoken,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=False,
            is_training=False)._transform)

    dev_data_transform, _ = preprocess_dataset(
        dev_data, SQuADTransform(
            berttoken,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=False,
            is_training=False))
    log.info('The number of examples after preprocessing:{}'.format(
        len(dev_data_transform)))

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn,
        num_workers=args.num_workers, batch_size=test_batch_size, shuffle=False, last_batch='keep')

    log.info('Start predict')

    _Result = collections.namedtuple(
        '_Result', ['example_id', 'start_logits', 'end_logits'])
    all_results = {}

    epoch_tic = time.time()
    total_num = 0

    log.info('Start Training')

    optimizer_params = {'learning_rate': lr}
    try:
        trainer = gluon.Trainer(net.collect_params(), optimizer,
                                optimizer_params, update_on_kvstore=False)
    except ValueError as e:
        print(e)
        warnings.warn('AdamW optimizer is not found. Please consider upgrading to '
                      'mxnet>=1.5.0. Now the original Adam optimizer is used instead.')
        trainer = gluon.Trainer(net.collect_params(), 'adam',
                                optimizer_params, update_on_kvstore=False)

    num_train_examples = len(train_data_transform)
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_steps = int(num_train_examples / step_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    def set_new_lr(step_num, batch_id):
        """set new learning rate"""
        # set grad to zero for gradient accumulation
        if accumulate:
            if batch_id % accumulate == 0:
                net.collect_params().zero_grad()
                step_num += 1
        else:
            step_num += 1
        # learning rate schedule
        # Notice that this learning rate scheduler is adapted from traditional linear learning
        # rate scheduler where step_num >= num_warmup_steps, new_lr = 1 - step_num/num_train_steps
        if step_num < num_warmup_steps:
            new_lr = lr * step_num / num_warmup_steps
        else:
            offset = (step_num - num_warmup_steps) * lr / \
                     (num_train_steps - num_warmup_steps)
            new_lr = lr - offset
        trainer.set_learning_rate(new_lr)
        return step_num

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in net.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    # Collect differentiable parameters
    params = [p for p in net.collect_params().values()
              if p.grad_req != 'null']
    # Set grad_req if gradient accumulation is required
    if accumulate:
        for p in params:
            p.grad_req = 'add'


    def train(epoch_id):
        for batch_id, data in enumerate(train_dataloader):
            # set new lr
            step_num = set_new_lr(step_num, batch_id)
            # forward and backward
            with mx.autograd.record():
                _, inputs, token_types, valid_length, start_label, end_label = data

                out = net(inputs.astype('float32').as_in_context(ctx),
                          token_types.astype('float32').as_in_context(ctx),
                          valid_length.astype('float32').as_in_context(ctx))

                ls = loss_function(out, [
                    start_label.astype('float32').as_in_context(ctx),
                    end_label.astype('float32').as_in_context(ctx)]).mean()

                if accumulate:
                    ls = ls / accumulate
            ls.backward()
            # update
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1)

    def test():
        test_loss = 0
        for data in dev_dataloader:
            example_ids, inputs, token_types, valid_length, _, _ = data
            out = net(inputs.astype('float32').as_in_context(ctx),
                      token_types.astype('float32').as_in_context(ctx),
                      valid_length.astype('float32').as_in_context(ctx))

            output = nd.split(out, axis=2, num_outputs=2)
            start_logits = output[0].reshape((0, -3)).asnumpy()
            end_logits = output[1].reshape((0, -3)).asnumpy()

            for example_id, start, end in zip(example_ids, start_logits, end_logits):
                example_id = example_id.asscalar()
                if example_id not in all_results:
                    all_results[example_id] = []
                all_results[example_id].append(
                    _Result(example_id, start.tolist(), end.tolist()))
        all_predictions, all_nbest_json, scores_diff_json = predictions(
            dev_dataset=dev_dataset,
            all_results=all_results,
            tokenizer=nlp.data.BERTBasicTokenizer(lower=lower),
            max_answer_length=max_answer_length,
            null_score_diff_threshold=null_score_diff_threshold,
            n_best_size=n_best_size,
            version_2=version_2)
        em_f1 = get_F1_EM(dev_data, all_predictions)
        reporter(mean_loss=test_loss, mean_accuracy=em_f1['f1'])

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()

if __name__ == "__main__":
    args = parser.parse_args()

    import ray
    from ray import tune
    from ray.tune.schedulers import AsyncHyperBandScheduler, FIFOScheduler, HyperBandScheduler

    ray.init()
    if args.scheduler == 'fifo':
        sched = FIFOScheduler()
    elif args.scheduler == 'asynchyperband':
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="neg_mean_loss",
            max_t=400,
            grace_period=60)
    elif args.scheduler == 'hyperband':
        sched = HyperBandScheduler(
            time_attr="training_iteration",
            reward_attr="neg_mean_loss",
            max_t=400)
    else:
        raise NotImplementedError
    tune.register_trainable(
        "TRAIN_FN",
        lambda config, reporter: train_indoor(args, config, reporter))
    tune.run(
        "TRAIN_FN",
        name=args.expname,
        verbose=2,
        scheduler=sched,
        reuse_actors=args.reuse_actors,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=args.checkpoint_at_end,
        max_failures=args.max_failures,
        queue_trials=args.queue_trials,
        with_server=args.with_server,
        **{
            "stop": {
                "mean_accuracy": args.stop_accuracy,
                "training_iteration": args.epochs
            },
            "resources_per_trial": {
                "cpu": int(args.num_workers),
                "gpu": 1
            },
            "num_samples": args.num_samples,
            "config": {
                "lr": tune.sample_from(
                    lambda spec: np.power(10.0, np.random.uniform(-7, -3))),#0.1 log uniform
                "epsilon": tune.sample_from(
                    lambda spec: np.power(10.0, np.random.uniform(-8, -4))),#0.9
            }
        })
