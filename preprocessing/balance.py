from matplotlib import pyplot as plt


def balance(counters):
    """
    Plot an pie chart of label counters

    :param counters: Counter for each label
    :return:

    """

    labels = 'Neutrophil', 'Lymphocyte', 'Monocyte', 'Eosinophil'

    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice

    fig1, ax1 = plt.subplots()
    ax1.pie(counters, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('./balance.png')
    plt.show()