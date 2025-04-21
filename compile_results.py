import pandas as pd
import numpy as np
from io import StringIO
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

def res(models, datasets, entries=[['1'], ['1', '2', '3', '4', '5']]):
    results = defaultdict(defaultdict)
    for dataset in datasets:
        acc_5_baseline = []
        acc_all_baseline = []
        for seed in [1, 17, 36, 91, 511]:
            out = read_output(f'important_outputs/{dataset}_source_{seed}.log')
            if len(out) == 0:
                print(f'No output for {dataset}_source_{seed}')
            out = ''.join(out)
            df = pd.read_csv(StringIO(out), delim_whitespace=True, index_col=0)
            # remove average row
            # df = df.drop("average")
            df = df.drop(df.index[-1])
            # acc_5_baseline.append(float(df['1'].mean()))
            acc_5_baseline.append(float(df.iloc[-1, 0]))
            acc_all_baseline.append(float(df.iloc[-1, -1]))
        for model in models:
            outs = []
            for seed in [1, 17, 36, 91, 511]:
                out = read_output(f'important_outputs/{dataset}_{model}_{seed}.log')
                if len(out) == 0:
                    print(f'No output for {dataset}_{model}_{seed}')
                outs.append(out)
            ce_5 = []
            ce_all = []
            acc_5 = []
            acc_all = []
            acc_clean = []
            for out in outs:
                out = ''.join(out)
                df = pd.read_csv(StringIO(out), delim_whitespace=True, index_col=0)
                acc_clean.append(float(df.iloc[-1, 0]))
                df = df.drop("average")
                # drop the last row
                df.drop(df.index[-1], inplace=True)
                ce_5.append((1 - df['1']) / (1- np.mean(acc_5_baseline)))
                ce_all.append((1 - df['avg']) / (1- np.mean(acc_all_baseline)))
                acc_5.append(float(df['1'].mean()))
                acc_all.append(float(df['avg'].mean()))
                # last row of the first column
            ce_5 = np.mean(ce_5)
            ce_all = np.mean(ce_all)
            acc_5 = np.array(acc_5).mean()
            acc_all = np.array(acc_all).mean()
            acc_clean = np.array(acc_clean).mean()
            # print('acc_5:', round(acc_5 * 100, 2))
            # print('acc_all:', round(acc_all * 100, 2))
            # print('acc_clean:', round(acc_clean * 100, 2))
            # print('ce_5:', round(ce_5 * 100, 2))
            # print('ce_all:', round(ce_all * 100, 2))
            # print('----------------------------------')
            results[dataset][model] = {
                'acc_5': round(acc_5 * 100, 2),
                'acc_all': round(acc_all * 100, 2),
                'acc_clean': round(acc_clean * 100, 2),
                'ce_5': round(ce_5 * 100, 2),
                'ce_all': round(ce_all * 100, 2),
            }
    print(f"{results['cifar10']['source']['acc_clean']} & {results['cifar10']['source']['acc_5']} & {results['cifar10']['source']['ce_5']} & {results['cifar10']['source']['acc_all']} & {results['cifar10']['source']['ce_all']} & {results['cifar100']['source']['acc_clean']} & {results['cifar100']['source']['acc_5']} & {results['cifar100']['source']['ce_5']} & {results['cifar100']['source']['acc_all']} & {results['cifar100']['source']['ce_all']} & {results['tin200']['source']['acc_clean']} & {results['tin200']['source']['acc_5']} & {results['tin200']['source']['ce_5']} & {results['tin200']['source']['acc_all']} & {results['tin200']['source']['ce_all']} \\\ ")
    print(f"{results['cifar10']['norm']['acc_clean']} & {results['cifar10']['norm']['acc_5']} & {results['cifar10']['norm']['ce_5']} & {results['cifar10']['norm']['acc_all']} & {results['cifar10']['norm']['ce_all']} & {results['cifar100']['norm']['acc_clean']} & {results['cifar100']['norm']['acc_5']} & {results['cifar100']['norm']['ce_5']} & {results['cifar100']['norm']['acc_all']} & {results['cifar100']['norm']['ce_all']} & {results['tin200']['norm']['acc_clean']} & {results['tin200']['norm']['acc_5']} & {results['tin200']['norm']['ce_5']} & {results['tin200']['norm']['acc_all']} & {results['tin200']['norm']['ce_all']} \\\ ")
    print(f"{results['cifar10']['pl']['acc_clean']} & {results['cifar10']['pl']['acc_5']} & {results['cifar10']['pl']['ce_5']} & {results['cifar10']['pl']['acc_all']} & {results['cifar10']['pl']['ce_all']} & {results['cifar100']['pl']['acc_clean']} & {results['cifar100']['pl']['acc_5']} & {results['cifar100']['pl']['ce_5']} & {results['cifar100']['pl']['acc_all']} & {results['cifar100']['pl']['ce_all']} & {results['tin200']['pl']['acc_clean']} & {results['tin200']['pl']['acc_5']} & {results['tin200']['pl']['ce_5']} & {results['tin200']['pl']['acc_all']} & {results['tin200']['pl']['ce_all']} \\\ ")
    # shot
    print(f"{results['cifar10']['shot']['acc_clean']} & {results['cifar10']['shot']['acc_5']} & {results['cifar10']['shot']['ce_5']} & {results['cifar10']['shot']['acc_all']} & {results['cifar10']['shot']['ce_all']} & {results['cifar100']['shot']['acc_clean']} & {results['cifar100']['shot']['acc_5']} & {results['cifar100']['shot']['ce_5']} & {results['cifar100']['shot']['acc_all']} & {results['cifar100']['shot']['ce_all']} & {results['tin200']['shot']['acc_clean']} & {results['tin200']['shot']['acc_5']} & {results['tin200']['shot']['ce_5']} & {results['tin200']['shot']['acc_all']} & {results['tin200']['shot']['ce_all']} \\\ ")
    # tent
    print(f"{results['cifar10']['tent']['acc_clean']} & {results['cifar10']['tent']['acc_5']} & {results['cifar10']['tent']['ce_5']} & {results['cifar10']['tent']['acc_all']} & {results['cifar10']['tent']['ce_all']} & {results['cifar100']['tent']['acc_clean']} & {results['cifar100']['tent']['acc_5']} & {results['cifar100']['tent']['ce_5']} & {results['cifar100']['tent']['acc_all']} & {results['cifar100']['tent']['ce_all']} & {results['tin200']['tent']['acc_clean']} & {results['tin200']['tent']['acc_5']} & {results['tin200']['tent']['ce_5']} & {results['tin200']['tent']['acc_all']} & {results['tin200']['tent']['ce_all']} \\\ ")
    # eta
    print(f"{results['cifar10']['eta']['acc_clean']} & {results['cifar10']['eta']['acc_5']} & {results['cifar10']['eta']['ce_5']} & {results['cifar10']['eta']['acc_all']} & {results['cifar10']['eta']['ce_all']} & {results['cifar100']['eta']['acc_clean']} & {results['cifar100']['eta']['acc_5']} & {results['cifar100']['eta']['ce_5']} & {results['cifar100']['eta']['acc_all']} & {results['cifar100']['eta']['ce_all']} & {results['tin200']['eta']['acc_clean']} & {results['tin200']['eta']['acc_5']} & {results['tin200']['eta']['ce_5']} & {results['tin200']['eta']['acc_all']} & {results['tin200']['eta']['ce_all']} \\\ ")
    # eata
    print(f"{results['cifar10']['eata']['acc_clean']} & {results['cifar10']['eata']['acc_5']} & {results['cifar10']['eata']['ce_5']} & {results['cifar10']['eata']['acc_all']} & {results['cifar10']['eata']['ce_all']} & {results['cifar100']['eata']['acc_clean']} & {results['cifar100']['eata']['acc_5']} & {results['cifar100']['eata']['ce_5']} & {results['cifar100']['eata']['acc_all']} & {results['cifar100']['eata']['ce_all']} & {results['tin200']['eata']['acc_clean']} & {results['tin200']['eata']['acc_5']} & {results['tin200']['eata']['ce_5']} & {results['tin200']['eata']['acc_all']} & {results['tin200']['eata']['ce_all']} \\\ ")
    # sar
    print(f"{results['cifar10']['sar']['acc_clean']} & {results['cifar10']['sar']['acc_5']} & {results['cifar10']['sar']['ce_5']} & {results['cifar10']['sar']['acc_all']} & {results['cifar10']['sar']['ce_all']} & {results['cifar100']['sar']['acc_clean']} & {results['cifar100']['sar']['acc_5']} & {results['cifar100']['sar']['ce_5']} & {results['cifar100']['sar']['acc_all']} & {results['cifar100']['sar']['ce_all']} & {results['tin200']['sar']['acc_clean']} & {results['tin200']['sar']['acc_5']} & {results['tin200']['sar']['ce_5']} & {results['tin200']['sar']['acc_all']} & {results['tin200']['sar']['ce_all']} \\\ ")
    # energy
    print(f"{results['cifar10']['energy']['acc_clean']} & {results['cifar10']['energy']['acc_5']} & {results['cifar10']['energy']['ce_5']} & {results['cifar10']['energy']['acc_all']} & {results['cifar10']['energy']['ce_all']} & {results['cifar100']['energy']['acc_clean']} & {results['cifar100']['energy']['acc_5']} & {results['cifar100']['energy']['ce_5']} & {results['cifar100']['energy']['acc_all']} & {results['cifar100']['energy']['ce_all']} & {results['tin200']['energy']['acc_clean']} & {results['tin200']['energy']['acc_5']} & {results['tin200']['energy']['ce_5']} & {results['tin200']['energy']['acc_all']} & {results['tin200']['energy']['ce_all']} \\\ ")




def read_output(path):
    out = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            # print(line)c
            if '1' in line and '2' in line and '3' in line and '4' in line and '5' in line and 'avg' in line:
                out = lines[i:i + 17]
                # out.pop(-2)
                # out[-1] = 'clean            ' + out[-1].split('Test set Accuracy: ')[1].strip()
                break
        # instead find 'Test set Accuracy: ' in the output lines
        res = [line for line in lines if 'Test set Accuracy: ' in line]
        out.append('clean            ' + res[-1].split('Test set Accuracy: ')[1].strip())
        # print(out[-1])
        return out
    except Exception as e:
        print(f'Error reading {path}: {e}')
        return []


if __name__ == '__main__':
    # read_output('logs/cifar10_source_1.log')
    models = ['source', 'norm', 'tent', 'eta', 'eata', 'energy', 'sar', 'shot', 'pl']
    datasets = ['cifar10', 'cifar100', 'tin200']

    res(models, datasets)