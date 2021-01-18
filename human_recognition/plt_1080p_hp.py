import conf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def get_hp_df():
    def read_hp_csv(n):
        df = pd.read_csv(f'1080p/{n}_rxplus4view_rzplus4view_1080p_humanrecog.csv', index_col=0)
        df = df.rename(columns={'HumanRecognition': f'hp{n}'})
        df[f'hp{n}'] = df[f'hp{n}'].str.lower()
        return df[['ca', 'no', 'tr', 'fr', f'hp{n}']]

    dfs = [read_hp_csv(i + 1) for i in range(3)]
    df = pd.concat(dfs, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def plt_confusion_matrix(df, true_col, pred_col):
    cm = confusion_matrix(df[true_col], df[pred_col], labels=conf.TOYBOX_PRED_LABELS)
    sns.heatmap(cm, xticklabels=conf.TOYBOX_PRED_LABELS, yticklabels=conf.TOYBOX_PRED_LABELS)
    plt.title('Human Recognition Confusion Matrix')
    plt.xlabel('Human Recognition')
    plt.ylabel('True Label')
    plt.show()


def main():
    hp_df = get_hp_df()
    plt_confusion_matrix(hp_df, 'ca', 'hp1')


if __name__ == '__main__':
    main()