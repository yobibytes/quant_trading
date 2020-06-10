def ROC_Curve():
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange', label='Random Forest')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (area = %0.7f)' % auc)
    plt.legend(loc='best')
    plt.show()
    #path = "C:/Users/Charles/Desktop/ROC Curve.png"
    #plt.savefig(path)
    #plt.cla()
    #plt.close('all')