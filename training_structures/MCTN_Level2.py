"""Implements training pipeline for 2 Level MCTN."""
from eval_scripts.complexity import all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from fusions.MCTN import Seq2Seq, L2_MCTN
from utils.evaluation_metric import eval_mosei_senti_return
from unimodals.common_models import MLP
from tqdm import tqdm
from torch.nn import functional as F
from torch import nn
import torch
import sys
import os

from torch.serialization import save

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


feature_dim = 300
hidden_dim = 2

reg_encoder = nn.GRU(hidden_dim, 128).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
head = MLP(128, 64, 1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

criterion_t = nn.MSELoss()
criterion_c = nn.MSELoss()
criterion_r = nn.L1Loss()


def train(
        traindata, validdata,
        encoder0, decoder0, encoder1, decoder1,
        reg_encoder, head,
        criterion_t0=nn.MSELoss(), criterion_c=nn.MSELoss(),
        criterion_t1=nn.MSELoss(), criterion_r=nn.L1Loss(),
        max_seq_len=20,
        mu_t0=0.01, mu_c=0.01, mu_t1=0.01,
        dropout_p=0.1, early_stop=False, patience_num=15,
        lr=1e-4, weight_decay=0.01, op_type=torch.optim.AdamW,
        epoch=100, model_save='best_mctn.pt',
        testdata=None):
    """Train a 2-level MCTN Instance

    Args:
        traindata (torch.util.data.DataLoader): Training data loader
        validdata (torch.util.data.DataLoader): Test data loader
        encoder0 (nn.Module): Encoder for first Seq2Seq Module
        decoder0 (nn.Module): Decoder for first SeqSeq Module
        encoder1 (nn.Module): Encoder for second Seq2Seq Module
        decoder1 (nn.Module): Decoder for second Seq2Seq Module
        reg_encoder (nn.Module): Regularization encoder.
        head (nn.Module): Actual classifier.
        criterion_t0 (nn.Module, optional): Loss function for t0. Defaults to nn.MSELoss().
        criterion_c (nn.Module, optional): Loss function for c. Defaults to nn.MSELoss().
        criterion_t1 (nn.Module, optional): Loss function for t1. Defaults to nn.MSELoss().
        criterion_r (nn.Module, optional): Loss function for r. Defaults to nn.L1Loss().
        max_seq_len (int, optional): Maximum sequence length. Defaults to 20.
        mu_t0 (float, optional): mu_t0. Defaults to 0.01.
        mu_c (float, optional): mu_c. Defaults to 0.01.
        mu_t1 (float, optional): mu_t. Defaults to 0.01.
        dropout_p (float, optional): Dropout Probability. Defaults to 0.1.
        early_stop (bool, optional): Whether to apply early stopping or not. Defaults to False.
        patience_num (int, optional): Patience Number for early stopping. Defaults to 15.
        lr (float, optional): Learning rate. Defaults to 1e-4.
        weight_decay (float, optional): Weight decay coefficient. Defaults to 0.01.
        op_type (torch.optim.Optimizer, optional): Optimizer instance. Defaults to torch.optim.AdamW.
        epoch (int, optional): Number of epochs. Defaults to 100.
        model_save (str, optional): Path to save best model. Defaults to 'best_mctn.pt'.
        testdata (torch.utils.data.DataLoader, optional): Data Loader for test data. Defaults to None.
    """
    seq2seq0 = Seq2Seq(encoder0, decoder0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    seq2seq1 = Seq2Seq(encoder1, decoder1).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = L2_MCTN(seq2seq0, seq2seq1, reg_encoder, head, p=dropout_p).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    op = op_type(model.parameters(), lr=lr, weight_decay=weight_decay)

    patience = 0
    best_acc = 0
    best_mae = 10000

    for ep in range(epoch):
        model.train()
        print('start training ---------->>')

        sum_total_loss = 0
        sum_reg_loss = 0
        total_batch = 0
        for i, inputs in enumerate(traindata):
            src, trg0, trg1, labels, f_dim = _process_input_L2(
                inputs, max_seq_len)
            translation_loss_0 = 0
            cyclic_loss = 0
            translation_loss_1 = 0
            reg_loss = 0
            total_loss = 0

            op.zero_grad()

            out, reout, rereout, head_out = model(src, trg0, trg1)

            for j, o in enumerate(out):
                translation_loss_0 += criterion_t0(o, trg0[j])
            translation_loss_0 = translation_loss_0 / out.size(0)

            for j, o in enumerate(reout):
                cyclic_loss += criterion_c(o, src[j])
            cyclic_loss = cyclic_loss / reout.size(0)

            for j, o in enumerate(rereout):
                translation_loss_1 += criterion_t1(o, trg1[j])
            translation_loss_1 = translation_loss_1 / rereout.size(0)

            reg_loss = criterion_r(head_out, labels)

            total_loss = mu_t0 * translation_loss_0 + mu_c * \
                cyclic_loss + mu_t1 * translation_loss_1 + reg_loss

            sum_total_loss += total_loss
            sum_reg_loss += reg_loss
            total_batch += 1

            total_loss.backward()
            op.step()

        sum_total_loss /= total_batch
        sum_reg_loss /= total_batch

        print('Train Epoch {}, total loss: {}, regression loss: {}, embedding loss: {}'.format(ep, sum_total_loss,
                                                                                               sum_reg_loss,
                                                                                               sum_total_loss - sum_reg_loss))

        model.eval()
        print('Start Evaluating ---------->>')
        pred = []
        true = []
        with torch.no_grad():
            for i, inputs in enumerate(validdata):
                # process input
                src, trg0, trg1, labels, feature_dim = _process_input_L2(
                    inputs, max_seq_len)

                #  We only need the source text as input! No need for target!
                _, _, _, head_out = model(src)
                pred.append(head_out)
                true.append(labels)

            eval_results_include = eval_mosei_senti_return(
                torch.cat(pred, 0), torch.cat(true, 0), exclude_zero=False)
            eval_results_exclude = eval_mosei_senti_return(
                torch.cat(pred, 0), torch.cat(true, 0), exclude_zero=True)
            mae = eval_results_include[0]
            Acc1 = eval_results_include[-1]
            Acc2 = eval_results_exclude[-1]
            print('Eval Epoch: {}, MAE: {}, Acc1: {}, Acc2: {}'.format(
                ep, mae, Acc1, Acc2))

            if mae < best_mae:
                patience = 0
                best_acc = Acc2
                best_mae = mae
                print('<------------ Saving Best Model')
                print()
                torch.save(model, model_save)
            else:
                patience += 1
            if early_stop and patience > patience_num:
                break


def single_test(model, testdata, max_seq_len=20):
    """Get accuracy for a single model and dataloader.

    Args:
        model (nn.Module): MCTN2 Model
        testdata (torch.utils.data.DataLoader): Test Dataloader
        max_seq_len (int, optional): Maximum sequence length. Defaults to 20.

    Returns:
        _type_: _description_
    """
    model.eval()
    print('Start Testing ---------->>')
    pred = []
    true = []
    with torch.no_grad():
        for i, inputs in enumerate(testdata):
            # process input
            src, _, _, labels, _ = _process_input_L2(inputs, max_seq_len)

            #  We only need the source text as input! No need for target!
            _, _, _, head_out = model(src)
            pred.append(head_out)
            true.append(labels)

        eval_results_include = eval_mosei_senti_return(
            torch.cat(pred, 0), torch.cat(true, 0), exclude_zero=False)
        eval_results_exclude = eval_mosei_senti_return(
            torch.cat(pred, 0), torch.cat(true, 0), exclude_zero=True)
        mae = eval_results_include[0]
        Acc1 = eval_results_include[-1]
        Acc2 = eval_results_exclude[-1]
        print('Test: MAE: {}, Acc1: {}, Acc2: {}'.format(mae, Acc1, Acc2))
        return {'Acc:': Acc2}


def test(model, test_dataloaders_all, dataset, method_name='My method', is_packed=False, criterion=nn.CrossEntropyLoss(), task="classification", auprc=False, input_to_float=True, no_robust=True):
    """Test MCTN_Level2 Module on a set of test dataloaders.

    Args:
        model (nn.Module): MCTN2 Module
        test_dataloaders_all (list): List of dataloaders
        dataset (Dataset): Dataset Name
        method_name (str, optional): Name of method. Defaults to 'My method'.
        is_packed (bool, optional): (unused). Defaults to False.
        criterion (_type_, optional): (unused). Defaults to nn.CrossEntropyLoss().
        task (str, optional): (unused). Defaults to "classification".
        auprc (bool, optional): (unused). Defaults to False.
        input_to_float (bool, optional): (unused). Defaults to True.
        no_robust (bool, optional): Whether to not apply robustness transformations or not. Defaults to True.
    """
    if no_robust:
        def _testprocess():
            single_test(model, test_dataloaders_all)
        all_in_one_test(_testprocess, [model])
    else:
        def _testprocess():
            single_test(model, test_dataloaders_all[list(
                test_dataloaders_all.keys())[0]][0])
        all_in_one_test(_testprocess, [model])
        for noisy_modality, test_dataloaders in test_dataloaders_all.items():
            print("Testing on noisy data ({})...".format(noisy_modality))
            robustness_curve = dict()
            for test_dataloader in tqdm(test_dataloaders):
                single_test_result = single_test(model, test_dataloader)
                for k, v in single_test_result.items():
                    curve = robustness_curve.get(k, [])
                    curve.append(v)
                    robustness_curve[k] = curve
            for measure, robustness_result in robustness_curve.items():
                robustness_key = '{} {}'.format(dataset, noisy_modality)
                print("relative robustness ({}, {}): {}".format(noisy_modality, measure, str(
                    relative_robustness(robustness_result, robustness_key))))
                if len(robustness_curve) != 1:
                    robustness_key = '{} {}'.format(robustness_key, measure)
                print("effective robustness ({}, {}): {}".format(noisy_modality, measure, str(
                    effective_robustness(robustness_result, robustness_key))))
                fig_name = '{}-{}-{}-{}'.format(method_name,
                                                robustness_key, noisy_modality, measure)
                single_plot(robustness_result, robustness_key, xlabel='Noise level',
                            ylabel=measure, fig_name=fig_name, method=method_name)
                print("Plot saved as "+fig_name)


def _process_input_L2(inputs, max_seq=20):
    src = inputs[0][2][:, :max_seq, :]
    trg0 = inputs[0][0][:, :max_seq, :]
    trg1 = inputs[0][1][:, :max_seq, :]
    feature_dim = max(src.size(-1), trg0.size(-1), trg1.size(-1))

    src = F.pad(src, (0, feature_dim - src.size(-1)))
    trg0 = F.pad(trg0, (0, feature_dim - trg0.size(-1)))
    trg1 = F.pad(trg1, (0, feature_dim - trg1.size(-1)))

    src = src.transpose(1, 0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    trg0 = trg0.transpose(1, 0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    trg1 = trg1.transpose(1, 0).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    labels = inputs[-1].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    return src, trg0, trg1, labels, feature_dim