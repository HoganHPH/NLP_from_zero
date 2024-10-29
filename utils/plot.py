import matplotlib.pyplot as plt

def plot_metrics(history, metric, namefig=''):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric.title())
    plt.legend([metric, f'val_{metric}'])
    plt.savefig(f'{namefig}.jpg')
    plt.show()