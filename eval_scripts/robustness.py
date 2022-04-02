import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


def relative_robustness(robustness_result, task):
    """Compute the relative robustenss metric given the performance of the method on the task."""
    return get_robustness_metric(robustness_result, task, 'relative')


def effective_robustness(robustness_result, task):
    """Compute the effective robustenss metric given the performance of the method on the task."""
    return get_robustness_metric(robustness_result, task, 'effective')


def get_robustness_metric(robustness_result, task, metric):
    """
    Compute robustness metric given specific method performance and the task.

    :param robustness_result: Performance of the method on datasets applied with different level of noises.
    :param task: Name of the task on which the method is evaluated.
    :param metric: Type of robustness metric to be computed. ( "effective" / "relative" )
    """
    if metric == 'effective' and task not in robustness['LF']:
        return "Invalid example name!"
    else:
        result = dict()
        if metric == 'relative':
            helper = relative_robustness_helper
        elif metric == 'effective':
            helper = effective_robustness_helper
        my_method = helper(robustness_result, task)
        for method in list(robustness.keys()):
            if not method.endswith('Transformer'):
                for t in list(robustness[method].keys()):
                    if t == task:
                        if (method == 'EF' or method == 'LF') and task in robustness[method+'-Transformer']:
                            result[method] = helper((np.array(
                                robustness[method][task])+np.array(robustness[method+'-Transformer'][task]))/2, task)
                        else:
                            result[method] = helper(
                                robustness[method][task], task)
        result['my method'] = my_method
        return maxmin_normalize(result, task)


def relative_robustness_helper(robustness_result, task):
    """
    Helper function that computes the relative robustness metric as the area under the performance curve.

    :param robustness_result: Performance of the method on datasets applied with different level of noises.
    """
    area = 0
    for i in range(len(robustness_result)-1):
        area += (robustness_result[i] + robustness_result[i+1]) * 0.1 / 2
    return area


def effective_robustness_helper(robustness_result, task):
    """
    Helper function that computes the effective robustness metric as the performance difference compared to late fusion method.

    :param robustness_result: Performance of the method on datasets applied with different level of noises.
    :param task: Name of the task on which the method is evaluated.
    """
    f = np.array(robustness_result)
    lf = np.array(robustness['LF'][task])
    beta_f = lf + (f[0] - lf[0])
    return np.sum(f - beta_f)


def maxmin_normalize(result, task):
    """
    Normalize the metric for robustness comparison across all methods.

    :param result: Un-normalized robustness metrics of all methods on the given task.
    :param task: Name of the task.
    """
    tmp = []
    method2idx = dict()
    for i, method in enumerate(list(result.keys())):
        method2idx[method] = i
        tmp.append(result[method])
    tmp = np.array(tmp)
    if task.startswith('finance'):
        tmp = -1 * tmp
    tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    return tmp[method2idx['my method']]


def single_plot(robustness_result, task, xlabel, ylabel, fig_name, method):
    """
    Produce performance vs. robustness plot of a single method.

    :param robustness_result: Performance of the method on dataset applied with different level of noises.
    :param task: Name of the task on which the method is evaluated.
    :param xlabel: Label of x-axis to be appeared in the plot.
    :param ylabel: Label of y-axis to be appeared in the plot.
    :param fig_name: Name of plot to be saved.
    :param method: Name of the method.
    """
    fig, axs = plt.subplots()
    if task.startswith('gentle push') or task.startswith('robotics image') or task.startswith('robotics force'):
        robustness_result = list(np.log(np.array(robustness_result)))
        plt.ylabel('log '+ylabel, fontsize=20)
    axs.plot(np.arange(len(robustness_result)) / 10,
             robustness_result, label=method, linewidth=2.5)
    plt.xlabel(xlabel, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # Uncomment the line below to show legends
    # fig.legend(fontsize=15, loc='upper left', bbox_to_anchor=(0.92, 0.94))
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close(fig)


###############################################################
performance = dict()

performance['Best Unimodal'] = dict()
performance['Best Unimodal']['mimic mortality timeseries'] = 76.7
performance['Best Unimodal']['mimic 7 timeseries'] = 67.6
performance['Best Unimodal']['mimic 1 timeseries'] = 91.4
performance['Best Unimodal']['finance F&B timeseries'] = 1.856
performance['Best Unimodal']['finance tech timeseries'] = 0.541
performance['Best Unimodal']['finance health timeseries'] = 0.152
performance['Best Unimodal']['enrico img'] = 47.0
performance['Best Unimodal']['enrico wireframe image'] = 46.1
performance['Best Unimodal']['imdb image micro'] = 40.12
performance['Best Unimodal']['imdb image macro'] = 25.31
performance['Best Unimodal']['imdb text micro'] = 58.56
performance['Best Unimodal']['imdb text macro'] = 45.59
performance['Best Unimodal']['robotics image'] = 1.99e-04
performance['Best Unimodal']['robotics force'] = 87.2e-04
performance['Best Unimodal']['robotics proprio'] = 0.202e-04
performance['Best Unimodal']['gentle push image'] = 0.334
performance['Best Unimodal']['gentle push proprio'] = 3.885
performance['Best Unimodal']['gentle push haptics'] = 4.266
performance['Best Unimodal']['gentle push controls'] = 3.804
performance['Best Unimodal']['mosi text'] = 74.2
performance['Best Unimodal']['mosi audio'] = 65.5
performance['Best Unimodal']['mosi image'] = 66.3
performance['Best Unimodal']['mosei text'] = 78.8
performance['Best Unimodal']['mosei audio'] = 66.4
performance['Best Unimodal']['mosei image'] = 67.2
performance['Best Unimodal']['sarcasm text'] = 68.6
performance['Best Unimodal']['sarcasm audio'] = 64.9
performance['Best Unimodal']['sarcasm image'] = 65.7
performance['Best Unimodal']['humor text'] = 58.3
performance['Best Unimodal']['humor audio'] = 57.2
performance['Best Unimodal']['humor image'] = 57.3

performance['LF'] = dict()
performance['LF']['mimic mortality timeseries'] = 77.9
performance['LF']['mimic 7 timeseries'] = 68.9
performance['LF']['mimic 1 timeseries'] = 91.5
performance['LF']['finance F&B timeseries'] = (1.893+2.155)/2
performance['LF']['finance tech timeseries'] = (0.541+0.573)/2
performance['LF']['finance health timeseries'] = (0.120+0.143)/2
performance['LF']['enrico image'] = 50.8
performance['LF']['enrico wireframe image'] = 50.8
performance['LF']['imdb image micro'] = 58.78
performance['LF']['imdb image macro'] = 49.21
performance['LF']['imdb text micro'] = 58.78
performance['LF']['imdb text macro'] = 49.21
performance['LF']['mosi multimodal'] = (75.2+79.6)/2
performance['LF']['mosei multimodal'] = (79.2+80.6)/2
performance['LF']['mosi text'] = (75.2+79.6)/2
performance['LF']['mosi audio'] = (75.2+79.6)/2
performance['LF']['mosi image'] = (75.2+79.6)/2
performance['LF']['mosei text'] = (79.2+80.6)/2
performance['LF']['mosei audio'] = (79.2+80.6)/2
performance['LF']['mosei image'] = (79.2+80.6)/2
performance['LF']['sarcasm multimodal'] = (66.1+66.1)/2
performance['LF']['sarcasm text'] = (66.1+66.1)/2
performance['LF']['sarcasm audio'] = (66.1+66.1)/2
performance['LF']['sarcasm image'] = (66.1+66.1)/2
performance['LF']['humor multimodal'] = (62.5+63.4)/2
performance['LF']['humor text'] = (62.5+63.4)/2
performance['LF']['humor audio'] = (62.5+63.4)/2
performance['LF']['humor image'] = (62.5+63.4)/2
performance['LF']['robotics image'] = 0.185e-04
performance['LF']['robotics force'] = 0.185e-04
performance['LF']['robotics proprio'] = 0.185e-04
performance['LF']['gentle push image'] = 0.29
performance['LF']['gentle push proprio'] = 0.29
performance['LF']['gentle push haptics'] = 0.29
performance['LF']['gentle push controls'] = 0.29
performance['LF']['gentle push multimodal'] = 0.29

performance['LRTF'] = dict()
performance['LRTF']['mimic mortality timeseries'] = 78.2
performance['LRTF']['mimic 7 timeseries'] = 68.5
performance['LRTF']['mimic 1 timeseries'] = 91.5
performance['LRTF']['enrico image'] = 47.1
performance['LRTF']['enrico wireframe image'] = 47.1
performance['LRTF']['robotics image'] = 0.232e-04
performance['LRTF']['robotics force'] = 0.232e-04
performance['LRTF']['robotics proprio'] = 0.232e-04
performance['LRTF']['imdb image micro'] = 59.2
performance['LRTF']['imdb image macro'] = 49.2
performance['LRTF']['imdb text micro'] = 59.2
performance['LRTF']['imdb tex macro'] = 49.2

performance['MI'] = dict()
performance['MI']['mimic mortality timeseries'] = 77.6
performance['MI']['mimic 7 timeseries'] = 67.9
performance['MI']['mimic 1 timeseries'] = 91.5
performance['MI']['enrico image'] = 46.7
performance['MI']['enrico wireframe image'] = 46.7
performance['MI']['imdb image micro'] = 58.3
performance['MI']['imdb image macro'] = 48
performance['MI']['imdb text micro'] = 58.3
performance['MI']['imdb tex macro'] = 48

performance['MVAE'] = dict()
performance['MVAE']['mimic mortality timeseries'] = 78
performance['MVAE']['mimic 7 timeseries'] = 68.7
performance['MVAE']['mimic 1 timeseries'] = 91.6

performance['MFM'] = dict()
performance['MFM']['mimic mortality timeseries'] = 78.2
performance['MFM']['mimic 7 timeseries'] = 68.8
performance['MFM']['mimic 1 timeseries'] = 91.5
performance['MFM']['imdb image micro'] = 38.4
performance['MFM']['imdb image macro'] = 22.3
performance['MFM']['imdb text micro'] = 38.4
performance['MFM']['imdb tex macro'] = 22.3


performance['EF'] = dict()
performance['EF']['finance F&B timeseries'] = (1.835+2.144)/2
performance['EF']['finance tech timeseries'] = (0.526+0.573)/2
performance['EF']['finance health timeseries'] = (0.121+0.143)/2
performance['EF']['imdb image micro'] = 58.86
performance['EF']['imdb image macro'] = 49.79
performance['EF']['imdb text micro'] = 58.86
performance['EF']['imdb text macro'] = 49.79
performance['EF']['mosi multimodal'] = (73.2+78.8)/2
performance['EF']['mosei multimodal'] = (78.4+79.6)/2
performance['EF']['mosi text'] = (73.2+78.8)/2
performance['EF']['mosi audio'] = (73.2+78.8)/2
performance['EF']['mosi image'] = (73.2+78.8)/2
performance['EF']['mosei text'] = (78.4+79.6)/2
performance['EF']['mosei audio'] = (78.4+79.6)/2
performance['EF']['mosei image'] = (78.4+79.6)/2
performance['EF']['sarcasm multimodal'] = (66.3+65.3)/2
performance['EF']['sarcasm text'] = (66.3+65.3)/2
performance['EF']['sarcasm audio'] = (66.3+65.3)/2
performance['EF']['sarcasm image'] = (66.3+65.3)/2
performance['EF']['humor multimodal'] = (60.2+62.9)/2
performance['EF']['humor text'] = (60.2+62.9)/2
performance['EF']['humor audio'] = (60.2+62.9)/2
performance['EF']['humor image'] = (60.2+62.9)/2
performance['EF']['gentle push image'] = 0.363
performance['EF']['gentle push proprio'] = 0.363
performance['EF']['gentle push haptics'] = 0.363
performance['EF']['gentle push controls'] = 0.363
performance['EF']['gentle push multimodal'] = 0.363


performance['GradBlend'] = dict()
performance['GradBlend']['finance F&B timeseries'] = 1.820
performance['GradBlend']['finance tech timeseries'] = 0.537
performance['GradBlend']['finance health timeseries'] = 0.138
performance['GradBlend']['enrico image'] = 51.0
performance['GradBlend']['enrico wireframe image'] = 51.0
performance['GradBlend']['mosi multimodal'] = 75.5
performance['GradBlend']['mosei multimodal'] = 78.1
performance['GradBlend']['mosi text'] = 75.5
performance['GradBlend']['mosi audio'] = 75.5
performance['GradBlend']['mosi image'] = 75.5
performance['GradBlend']['mosei text'] = 78.1
performance['GradBlend']['mosei audio'] = 78.1
performance['GradBlend']['mosei image'] = 78.1
performance['GradBlend']['sarcasm multimodal'] = 66.1
performance['GradBlend']['sarcasm text'] = 66.1
performance['GradBlend']['sarcasm audio'] = 66.1
performance['GradBlend']['sarcasm image'] = 66.1
performance['GradBlend']['humor multimodal'] = 62.3
performance['GradBlend']['humor text'] = 62.3
performance['GradBlend']['humor audio'] = 62.3
performance['GradBlend']['humor image'] = 62.3

performance['MulT'] = dict()
performance['MulT']['finance F&B timeseries'] = 2.053
performance['MulT']['finance tech timeseries'] = 0.555
performance['MulT']['finance health timeseries'] = 0.135
performance['MulT']['mosi multimodal'] = 83
performance['MulT']['mosei multimodal'] = 82.1
performance['MulT']['mosi text'] = 83
performance['MulT']['mosi audio'] = 83
performance['MulT']['mosi image'] = 83
performance['MulT']['mosei text'] = 82.1
performance['MulT']['mosei audio'] = 82.1
performance['MulT']['mosei image'] = 82.1
performance['MulT']['sarcasm multimodal'] = 71.8
performance['MulT']['sarcasm text'] = 71.8
performance['MulT']['sarcasm audio'] = 71.8
performance['MulT']['sarcasm image'] = 71.8
performance['MulT']['humor multimodal'] = 66.7
performance['MulT']['humor text'] = 66.7
performance['MulT']['humor audio'] = 66.7
performance['MulT']['humor image'] = 66.7

performance['CCA'] = dict()
performance['CCA']['enrico image'] = 50.1
performance['CCA']['enrico wireframe image'] = 50.1
performance['CCA']['imdb image micro'] = 59.33
performance['CCA']['imdb image macro'] = 50.21
performance['CCA']['imdb text micro'] = 59.33
performance['CCA']['imdb text macro'] = 50.21

performance['TF'] = dict()
performance['TF']['enrico image'] = 46.6
performance['TF']['enrico wireframe image'] = 46.6
performance['TF']['gentle push image'] = 0.574
performance['TF']['gentle push proprio'] = 0.574
performance['TF']['gentle push haptics'] = 0.574
performance['TF']['gentle push controls'] = 0.574
performance['TF']['gentle push multimodal'] = 0.574

performance['ReFNet'] = dict()
performance['ReFNet']['imdb image micro'] = 51.17
performance['ReFNet']['imdb image macro'] = 36.51
performance['ReFNet']['imdb text micro'] = 51.17
performance['ReFNet']['imdb text macro'] = 36.51

performance['RMFE'] = dict()
performance['RMFE']['imdb image micro'] = 58.64
performance['RMFE']['imdb image macro'] = 47.10
performance['RMFE']['imdb text micro'] = 58.64
performance['RMFE']['imdb text macro'] = 47.10

performance['MCTN'] = dict()
performance['MCTN']['mosi multimodal'] = 76.9
performance['MCTN']['mosei multimodal'] = 76.4
performance['MCTN']['mosi text'] = 76.9
performance['MCTN']['mosi audio'] = 76.9
performance['MCTN']['mosi image'] = 76.9
performance['MCTN']['mosei text'] = 76.4
performance['MCTN']['mosei audio'] = 76.4
performance['MCTN']['mosei image'] = 76.4
performance['MCTN']['sarcasm multimodal'] = 63.2
performance['MCTN']['sarcasm text'] = 63.2
performance['MCTN']['sarcasm audio'] = 63.2
performance['MCTN']['sarcasm image'] = 63.2
performance['MCTN']['humor multimodal'] = 63.2
performance['MCTN']['humor text'] = 63.2
performance['MCTN']['humor audio'] = 63.2
performance['MCTN']['humor image'] = 63.2

performance['MFAS'] = dict()
performance['MFAS']['mimic mortality timeseries'] = 77.9
performance['MFAS']['mimic 7 timeseries'] = 68.5
performance['MFAS']['mimic 1 timeseries'] = 91.4

performance['SF'] = dict()
performance['SF']['robotics image'] = 0.258e-04
performance['SF']['robotics force'] = 0.258e-04
performance['SF']['robotics proprio'] = 0.258e-04


robustness = dict()

robustness['Best Unimodal'] = dict()
robustness['Best Unimodal']['mimic mortality timeseries'] = [0.7679337829552422, 0.7685469037400368, 0.7664009809932557, 0.76364193746168,
                                                             0.7624156958920908, 0.7608828939301042, 0.7608828939301042, 0.7608828939301042, 0.7608828939301042, 0.7608828939301042, 0.7608828939301042]
robustness['Best Unimodal']['mimic 7 timeseries'] = [0.6799509503372164, 0.6364193746167995, 0.5836909871244635, 0.5389331698344574,
                                                     0.5067443286327407, 0.4883507050889025, 0.47884733292458614, 0.4754751686082158, 0.47455548743102394, 0.47455548743102394, 0.47455548743102394]
robustness['Best Unimodal']['mimic 1 timeseries'] = [0.9141630901287554, 0.8966891477621092, 0.8819742489270386, 0.869098712446352,
                                                     0.8559166155732679, 0.8424279583077866, 0.8362967504598406, 0.8332311465358676, 0.8283261802575107, 0.8280196198651134, 0.8307786633966892]
robustness['Best Unimodal']['finance F&B timeseries'] = [1.7468543529510498, 1.7499454975128175, 1.7556643962860108,
                                                         1.835334587097168, 2.066647481918335, 2.052792263031006, 2.0245705604553224, 2.1208776950836183, 2.179097318649292]
robustness['Best Unimodal']['finance tech timeseries'] = [0.5087007701396942, 0.5322327971458435, 0.5292805194854736,
                                                          0.5284122347831726, 0.5647271871566772, 0.578832495212555, 0.5865296125411987, 0.5923392415046692, 0.6030311346054077]
robustness['Best Unimodal']['finance health timeseries'] = [0.13544526994228362, 0.13520690202713012, 0.13254896998405458,
                                                            0.13934721648693085, 0.1363735854625702, 0.14343191981315612, 0.14123509526252748, 0.14286822378635405, 0.14537362158298492]
robustness['Best Unimodal']['enrico image'] = [0.476027397260274, 0.4315068493150685, 0.3664383561643836, 0.3904109589041096,
                                               0.3458904109589041, 0.3287671232876712, 0.2773972602739726, 0.2568493150684932, 0.1952054794520548, 0.1780821917808219, 0.1780821917808219]
robustness['Best Unimodal']['enrico wireframe image'] = [0.4041095890410959, 0.363013698630137, 0.2808219178082192, 0.2602739726027397,
                                                         0.2328767123287671, 0.23972602739726026, 0.1815068493150685, 0.17465753424657535, 0.11986301369863013, 0.1267123287671233, 0.11643835616438356]
robustness['Best Unimodal']['imdb image micro'] = [0.38512736733829483, 0.3530907850252784, 0.3197661344726782, 0.2827807770012917,
                                                   0.24832810154224103, 0.2152988325492246, 0.17723926596456624, 0.15668589407446248, 0.12698856683332752, 0.09800768961901433, 0.0720019630525467]
robustness['Best Unimodal']['imdb image macro'] = [0.24226895575816887, 0.20302311355322583, 0.17458299723201112, 0.14042485251724146,
                                                   0.11123276514307273, 0.0932934380504998, 0.06856332971325599, 0.06301048015726372, 0.050413983851831226, 0.03792263331281559, 0.02630802107776935]
robustness['Best Unimodal']['imdb text micro'] = [0.5653659841302863, 0.5578369344634118, 0.5504488330341114, 0.5466415116940675,
                                                  0.536558945658061, 0.5214946490114275, 0.5075646656905807, 0.48528776756523345, 0.46313025538432684, 0.43496734098547785, 0.3928502290765955]
robustness['Best Unimodal']['imdb text macro'] = [0.45606721431009556, 0.44743058548592096, 0.438329928127804, 0.42914264325501605,
                                                  0.42007058208270864, 0.40304532038698543, 0.38652353509059756, 0.3586243957507599, 0.33042799605063333, 0.2956624426561755, 0.2449165656268762]
robustness['Best Unimodal']['robotics image'] = [0.00014103209832683206, 0.00620879465714097, 0.012023922987282276, 0.016574595123529434,
                                                 0.020169781520962715, 0.022540835663676262, 0.024388503283262253, 0.02571929432451725, 0.02669128030538559, 0.027738867327570915]
robustness['Best Unimodal']['robotics force'] = [0.00827085692435503, 0.010384513065218925, 0.010641143657267094, 0.010816379450261593,
                                                 0.010863110423088074, 0.010730101726949215, 0.010553873144090176, 0.010225364938378334, 0.009965472854673862, 0.009790400974452496]
robustness['Best Unimodal']['robotics proprio'] = [2.2045116566005163e-05, 0.06653524935245514, 0.1299981325864792, 0.18672703206539154,
                                                   0.2377871572971344, 0.2809649109840393, 0.317918062210083, 0.3461359143257141, 0.36773908138275146, 0.3807152211666107]
robustness['Best Unimodal']['gentle push image'] = [0.3402138374, 8.570374729, 9.515328663,
                                                    9.994570034, 10.37768268, 10.86402119, 11.10239319, 11.23548274, 11.31663486, 11.36236464]
robustness['Best Unimodal']['gentle push proprio'] = [3.882181227, 10.11027428, 11.5265418,
                                                      12.31016493, 12.85656545, 13.32548696, 13.75450418, 14.12150526, 14.43968048, 14.71565828]
robustness['Best Unimodal']['gentle push haptics'] = [4.258245598, 5.56309104, 5.6800898339999994,
                                                      5.742590739, 5.76590473, 5.765610142000001, 5.736537283, 5.690129585, 5.637320391, 5.588522257999999]
robustness['Best Unimodal']['gentle push controls'] = [3.801201145, 8.296545031, 8.199219515,
                                                       8.063475853, 7.9808950439999995, 7.982850701, 8.083800323, 8.38630186, 9.019196563, 10.06180918]
robustness['Best Unimodal']['mosi text'] = [0.7421, 0.7099,
                                            0.6237, 0.5477, 0.6028, 0.516, 0.4731, 0.4735, 0.334, 0.4469]
robustness['Best Unimodal']['mosi audio'] = [0.651, 0.5965,
                                             0.5599, 0.5488, 0.5221, 0.4952, 0.4906, 0.4657, 0.4435, 0.4262]
robustness['Best Unimodal']['mosi image'] = [0.6733, 0.6234,
                                             0.5919, 0.5848, 0.5332, 0.5234, 0.4867, 0.4853, 0.4643, 0.4432]
robustness['Best Unimodal']['mosei text'] = [0.7826, 0.7121,
                                             0.6275, 0.5434, 0.5227, 0.4992, 0.4692, 0.4747, 0.4312, 0.4398]
robustness['Best Unimodal']['mosei audio'] = [0.659, 0.5948,
                                              0.5613, 0.5519, 0.5196, 0.4986, 0.5047, 0.4602, 0.4464, 0.43]
robustness['Best Unimodal']['mosei image'] = [0.6732, 0.6349,
                                              0.5926, 0.5713, 0.5371, 0.5237, 0.4897, 0.4549, 0.4865, 0.4567]
robustness['Best Unimodal']['sarcasm text'] = [0.682, 0.5926,
                                               0.5164, 0.4393, 0.4071, 0.3591, 0.3584, 0.3988, 0.3441, 0.2933]
robustness['Best Unimodal']['sarcasm audio'] = [0.6474, 0.5968,
                                                0.5749, 0.5436, 0.5378, 0.4984, 0.4701, 0.4037, 0.343, 0.3193]
robustness['Best Unimodal']['sarcasm image'] = [0.6612, 0.6263,
                                                0.6041, 0.5945, 0.5308, 0.4998, 0.489, 0.448, 0.3703, 0.3484]
robustness['Best Unimodal']['humor text'] = [0.5742, 0.5349,
                                             0.5078, 0.4762, 0.4754, 0.4632, 0.3822, 0.4176, 0.3505, 0.3483]
robustness['Best Unimodal']['humor audio'] = [0.5731, 0.5657,
                                              0.5143, 0.4678, 0.3579, 0.3772, 0.3655, 0.3771, 0.3219, 0.2756]
robustness['Best Unimodal']['humor image'] = [0.5748, 0.5375,
                                              0.5086, 0.479, 0.4798, 0.4607, 0.3801, 0.3995, 0.3537, 0.3422]

robustness['LF'] = dict()
robustness['LF']['mimic mortality timeseries'] = [0.7835683629675047, 0.778969957081545, 0.7676272225628449, 0.765787860208461,
                                                  0.7608828939301042, 0.7602697731453096, 0.7605763335377069, 0.7611894543225015, 0.7599632127529123, 0.7608828939301042, 0.7608828939301042]
robustness['LF']['mimic 7 timeseries'] = [0.6888412017167382, 0.6615573267933783, 0.6164929491109749, 0.5524218270999387, 0.5214592274678111,
                                          0.49908031882280807, 0.4834457388105457, 0.4773145309625996, 0.4742489270386266, 0.47455548743102394, 0.47455548743102394]
robustness['LF']['mimic 1 timeseries'] = [0.9175352544451257, 0.9022072348252606, 0.8825873697118333, 0.8666462293071735,
                                          0.8568362967504598, 0.8448804414469651, 0.828632740649908, 0.82158185162477, 0.8203556100551809, 0.8218884120171673, 0.8307786633966892]
robustness['LF']['finance F&B timeseries'] = [2.147226333618164, 2.1456197261810304, 2.1531070709228515,
                                              2.1734951019287108, 2.2157562732696534, 2.2615702629089354, 2.222545337677002, 2.190892744064331, 2.2127240657806397]
robustness['LF']['finance tech timeseries'] = [0.5715468645095825, 0.572181236743927, 0.5732102513313293,
                                               0.5746765255928039, 0.5741363644599915, 0.5756564021110535, 0.5762433767318725, 0.5766653180122375, 0.5771104335784912]
robustness['LF']['finance health timeseries'] = [0.13669011294841765, 0.1372331202030182, 0.13825314939022065,
                                                 0.13884556889533997, 0.1396324962377548, 0.14026106297969818, 0.1400857150554657, 0.14022547602653504, 0.14065399765968323]
robustness['LF']['enrico image'] = [0.5034246575342466, 0.4863013698630137, 0.4417808219178082, 0.4486301369863014, 0.4143835616438356,
                                    0.4178082191780822, 0.4246575342465753, 0.4315068493150685, 0.4315068493150685, 0.4143835616438356, 0.4212328767123288]
robustness['LF']['enrico wireframe image'] = [0.5034246575342466, 0.4965753424657534, 0.4623287671232877, 0.476027397260274,
                                              0.4691780821917808, 0.4246575342465753, 0.4178082191780822, 0.4006849315068493, 0.3595890410958904, 0.3527397260273973, 0.3287671232876712]
robustness['LF']['imdb image micro'] = [0.5972920696324951, 0.5763660703419738, 0.5561813313636783, 0.5328480776409108, 0.5137057109739669,
                                        0.4981840002526608, 0.48676643472182507, 0.48387715930902114, 0.4743016939346212, 0.4718328141225337, 0.46900111060299204]
robustness['LF']['imdb image macro'] = [0.4966323045046603, 0.47167172713170813, 0.44649891922678564, 0.4130371301721295, 0.3843869290429416,
                                        0.3615454227888607, 0.3466988301012139, 0.337743361662678, 0.3248116161723605, 0.31725136163076945, 0.31062159795553324]
robustness['LF']['imdb text micro'] = [0.5973701433926341, 0.5906568820476106, 0.5878271272618514, 0.5809663777331214, 0.5756544266048136,
                                       0.5610587161545482, 0.5491803278688525, 0.5311035959169743, 0.5094036480483887, 0.4814398922005839, 0.4522058823529411]
robustness['LF']['imdb text macro'] = [0.4961344152976692, 0.48801067069159965, 0.4820569741096813, 0.4742386599604151, 0.46348228941124187,
                                       0.4471172039006327, 0.42948253918909307, 0.3975653204509653, 0.36445386722073503, 0.3259684563867358, 0.28516585328982536]
robustness['LF']['mosi multimodal'] = [0.7521, 0.7025, 0.6212,
                                       0.5535, 0.5297, 0.4811, 0.4891, 0.4672, 0.4541, 0.4469]
robustness['LF']['mosi text'] = [0.7521, 0.6849, 0.6534,
                                 0.6326, 0.6012, 0.5623, 0.5429, 0.5192, 0.4538, 0.4566]
robustness['LF']['mosi audio'] = [0.7521, 0.6987, 0.6618,
                                  0.6387, 0.6078, 0.5239, 0.536, 0.5192, 0.4738, 0.4603]
robustness['LF']['mosi image'] = [0.7521, 0.6872, 0.6779999999999999,
                                  0.6279, 0.5936, 0.5125, 0.5206, 0.5294, 0.474, 0.4605]
robustness['LF']['mosei multimodal'] = [0.7902, 0.7055, 0.6191,
                                        0.5443, 0.5302, 0.4827, 0.4963, 0.4662, 0.4578, 0.4572]
robustness['LF']['mosei text'] = [0.7902, 0.7042, 0.6387, 0.5615, 0.5302,
                                  0.5770000000000001, 0.49200000000000005, 0.4673, 0.4647, 0.4448]
robustness['LF']['mosei audio'] = [0.7902, 0.6939, 0.6629999999999999,
                                   0.6857, 0.6053, 0.5133, 0.537, 0.5181, 0.4746, 0.4597]
robustness['LF']['mosei image'] = [0.7902, 0.6948, 0.6643,
                                   0.6337, 0.6027, 0.5238, 0.5358, 0.5181, 0.4619, 0.4341]
robustness['LF']['sarcasm multimodal'] = [0.6616, 0.5971, 0.5564,
                                          0.5297, 0.4273, 0.3976, 0.4421, 0.4096, 0.3769, 0.3424]
robustness['LF']['sarcasm text'] = [0.6616, 0.6103, 0.5487,
                                    0.5243, 0.3615, 0.3963, 0.3963, 0.4026, 0.376, 0.3474]
robustness['LF']['sarcasm audio'] = [0.6616, 0.5799, 0.5667,
                                     0.5092, 0.437, 0.3993, 0.4511, 0.4072, 0.3778, 0.3253]
robustness['LF']['sarcasm image'] = [0.6616, 0.5919, 0.5621,
                                     0.5113, 0.3481, 0.3714, 0.4325, 0.3803, 0.379, 0.3943]
robustness['LF']['humor multimodal'] = [0.6194, 0.5041, 0.4772,
                                        0.4873, 0.4791, 0.4875, 0.4472, 0.4306, 0.3872, 0.3652]
robustness['LF']['humor text'] = [0.6194, 0.5029, 0.4749,
                                  0.4851, 0.4791, 0.4861, 0.4459, 0.4276, 0.4078, 0.3625]
robustness['LF']['humor audio'] = [0.6194, 0.5799, 0.5667,
                                   0.5092, 0.437, 0.3993, 0.4511, 0.4072, 0.3778, 0.3253]
robustness['LF']['humor image'] = [0.6194, 0.4973, 0.4714,
                                   0.4872, 0.4845, 0.4935, 0.4312, 0.4483, 0.3818, 0.3661]
robustness['LF']['robotics image'] = [2.1844e-05, 0.0003, 0.0004,
                                      0.0006, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007]
robustness['LF']['robotics force'] = [2.2348e-05, 2.1718e-05, 2.1818e-05,
                                      2.2084e-05, 2.2351e-05, 2.1937e-05, 2.1958e-05, 2.1920e-05, 2.1594e-05, 2.1556e-05]
robustness['LF']['robotics proprio'] = [2.2035e-05, 0.0586,
                                        0.1074, 0.1544, 0.1901, 0.2233, 0.2492, 0.27, 0.285, 0.2941]
robustness['LF']['gentle push image'] = [0.30024437449999997, 6.489797876, 7.298926889,
                                         7.837444693999999, 8.150960949, 8.34612102, 8.585395415, 8.84260658, 9.080718362, 9.289110332]
robustness['LF']['gentle push proprio'] = [0.30024437449999997, 2.474324503, 3.3083693089999997,
                                           3.7756676330000003, 4.100822561, 4.395647919, 4.709313462, 5.123329762, 5.72107685, 6.643224793]
robustness['LF']['gentle push haptics'] = [0.30024437449999997, 0.39677922369999996, 0.5125797627,
                                           0.5798155335, 0.6114375767, 0.6130500737, 0.5960424066, 0.5597530293999999, 0.5031489268, 0.42545655270000005]
robustness['LF']['gentle push controls'] = [0.30024437449999997, 1.127455943, 1.183305545, 1.2249671009999998,
                                            1.272628643, 1.341969143, 1.457898519, 1.653203247, 1.9717935969999998, 2.4725279540000002]
robustness['LF']['gentle push multimodal'] = [0.30024437449999997, 3.391955101, 4.6935460419999995,
                                              5.556768718, 6.295041949, 6.994594133, 7.76214176, 8.544142639, 9.404722888, 10.29976233]
robustness['LF-Transformer'] = dict()
robustness['LF-Transformer']['finance F&B timeseries'] = [2.147226333618164, 2.1456197261810304, 2.1531070709228515,
                                                          2.1734951019287108, 2.2157562732696534, 2.2615702629089354, 2.222545337677002, 2.190892744064331, 2.2127240657806397]
robustness['LF-Transformer']['finance tech timeseries'] = [0.5809576392173768, 0.5809600591659546, 0.5805474042892456,
                                                           0.5812169909477234, 0.5804373979568481, 0.5805263996124268, 0.5806226253509521, 0.5805852770805359, 0.5804862380027771]
robustness['LF-Transformer']['finance health timeseries'] = [0.14129721224308014, 0.14125592708587648, 0.14108223021030425,
                                                             0.1410686790943146, 0.14106203913688659, 0.14086924493312836, 0.14075282216072083, 0.1409447342157364, 0.14098310172557832]
robustness['LF-Transformer']['mosi multimodal'] = [0.7982, 0.7287,
                                                   0.6902, 0.6407, 0.5902, 0.5123, 0.4833, 0.4623, 0.4524, 0.4449]
robustness['LF-Transformer']['mosi text'] = [0.7982, 0.7167,
                                             0.6902, 0.6401, 0.6131, 0.5516, 0.4751, 0.4523, 0.4656, 0.4603]
robustness['LF-Transformer']['mosi audio'] = [0.7982, 0.7167,
                                              0.69, 0.663, 0.6324, 0.5516, 0.5226, 0.5103, 0.4956, 0.5102]
robustness['LF-Transformer']['mosi image'] = [0.7982, 0.7032,
                                              0.6531, 0.6348, 0.6213, 0.6029, 0.5539, 0.5417, 0.5649, 0.4921]
robustness['LF-Transformer']['mosei multimodal'] = [0.7998, 0.732,
                                                    0.689, 0.6411, 0.5822, 0.5148, 0.4832, 0.4767, 0.4512, 0.4427]
robustness['LF-Transformer']['mosei text'] = [0.7902, 0.7042,
                                              0.6387, 0.5615, 0.5302, 0.577, 0.492, 0.4673, 0.4647, 0.4448]
robustness['LF-Transformer']['mosei audio'] = [0.7902, 0.6939,
                                               0.663, 0.6857, 0.6053, 0.5133, 0.537, 0.5181, 0.4746, 0.4597]
robustness['LF-Transformer']['mosei image'] = [0.7902, 0.6948,
                                               0.6643, 0.6337, 0.6027, 0.5238, 0.5358, 0.5181, 0.4619, 0.4341]
robustness['LF-Transformer']['sarcasm multimodal'] = [0.6623, 0.6131,
                                                      0.5991, 0.558, 0.5568, 0.4433, 0.431, 0.4199, 0.3415, 0.3333]
robustness['LF-Transformer']['sarcasm text'] = [0.6623, 0.6012,
                                                0.6057, 0.5571, 0.567, 0.4416, 0.4388, 0.4331, 0.3388, 0.3404]
robustness['LF-Transformer']['sarcasm audio'] = [0.6623, 0.5851,
                                                 0.5733, 0.5696, 0.5587, 0.4397, 0.4344, 0.4221, 0.3366, 0.3019]
robustness['LF-Transformer']['sarcasm image'] = [0.6623, 0.6183,
                                                 0.6088, 0.5599, 0.5657, 0.4098, 0.4412, 0.4299, 0.3412, 0.3428]
robustness['LF-Transformer']['humor multimodal'] = [0.6323, 0.5075,
                                                    0.6082, 0.5481, 0.4366, 0.4331, 0.4198, 0.424, 0.4302, 0.4303]
robustness['LF-Transformer']['humor text'] = [0.6323, 0.5109,
                                              0.606, 0.5472, 0.4326, 0.4307, 0.422, 0.4262, 0.4325, 0.4305]
robustness['LF-Transformer']['humor audio'] = [0.6323, 0.5851,
                                               0.5733, 0.5696, 0.5587, 0.4397, 0.4344, 0.4221, 0.3366, 0.3019]
robustness['LF-Transformer']['humor image'] = [0.6323, 0.5099,
                                               0.6062, 0.5448, 0.4327, 0.4322, 0.4189, 0.4249, 0.4324, 0.4322]


robustness['LRTF'] = dict()
robustness['LRTF']['mimic mortality timeseries'] = [0.7817290006131208, 0.773758430410791, 0.7639484978540773, 0.7627222562844881,
                                                    0.7621091354996934, 0.7642550582464746, 0.7593500919681178, 0.7593500919681178, 0.7593500919681178, 0.7593500919681178, 0.7608828939301042]
robustness['LRTF']['mimic 7 timeseries'] = [0.687614960147149, 0.6646229307173513, 0.6376456161863887, 0.605456774984672,
                                            0.5619251992642551, 0.530349478847333, 0.5021459227467812, 0.4840588595953403, 0.4819129368485592, 0.4760882893930104, 0.47455548743102394]
robustness['LRTF']['mimic 1 timeseries'] = [0.9141630901287554, 0.8963825873697119, 0.8761496014714899, 0.8479460453709381,
                                            0.8249540159411404, 0.8154506437768241, 0.7964438994481913, 0.7746781115879828, 0.7820355610055181, 0.7982832618025751, 0.8179031269160024]
robustness['LRTF']['enrico image'] = [0.4863013698630137, 0.4623287671232877, 0.3972602739726027, 0.410958904109589, 0.3767123287671233,
                                      0.3253424657534247, 0.2636986301369863, 0.2773972602739726, 0.2226027397260274, 0.1678082191780822, 0.10616438356164383]
robustness['LRTF']['enrico wireframe image'] = [0.4863013698630137, 0.4383561643835616, 0.4075342465753425, 0.3698630136986301,
                                                0.3527397260273973, 0.3321917808219178, 0.3082191780821918, 0.2465753424657534, 0.2363013698630137, 0.20205479452054795, 0.1917808219178082]
robustness['LRTF']['robotics image'] = [2.1845e-05, 0.005, 0.009,
                                        0.0015, 0.0023, 0.0034, 0.0045, 0.0056, 0.0067, 0.0074]
robustness['LRTF']['robotics force'] = [2.1447e-05, 2.1792e-05, 2.1781e-05,
                                        2.2270e-05, 2.2075e-05, 2.2589e-05, 2.2186e-05, 2.2074e-05, 2.2572e-05, 2.1517e-05]
robustness['LRTF']['robotics proprio'] = [2.2028e-05, 0.0735,
                                          0.1396, 0.1979, 0.2469, 0.2881, 0.3218, 0.3474, 0.3651, 0.3759]
robustness['LRTF']['imdb image micro'] = [0.5913649259810668, 0.5745157767687799, 0.551883616778386, 0.5381301803273819, 0.5167599446350124,
                                          0.5027594153678209, 0.49471702696137965, 0.49681064333880076, 0.49385034837927894, 0.49662039831019916, 0.4976367522654063]
robustness['LRTF']['imdb image macro'] = [0.5021858121516295, 0.4740446812854553, 0.4518231380876541, 0.4319055295920208, 0.4057951861194304,
                                          0.3904169422268742, 0.3719047396823358, 0.3738588985035647, 0.36442718102833954, 0.363926338992062, 0.36383377999422256]
robustness['LRTF']['imdb text micro'] = [0.5919325368938863, 0.5873802816901408, 0.5816164817749603, 0.5777803047532409,
                                         0.5693205216197667, 0.5571106355907698, 0.5461641521523853, 0.5283687943262412, 0.511760813330927, 0.4852150950348512, 0.4629594840002505]
robustness['LRTF']['imdb text macro'] = [0.5019679402309065, 0.49360173691357145, 0.4844043193606097, 0.47541387082118197,
                                         0.46478104828041805, 0.4469420055769647, 0.42661843628508633, 0.3965706042979783, 0.3707555407203844, 0.3257371366018358, 0.2867556426227773]

robustness['MI'] = dict()
robustness['MI']['mimic mortality timeseries'] = [0.7765174739423667, 0.7734518700183937, 0.7709993868792152, 0.769773145309626,
                                                  0.7608828939301042, 0.7581238503985285, 0.7581238503985285, 0.7578172900061312, 0.759656652360515, 0.7587369711833231, 0.7608828939301042]
robustness['MI']['mimic 7 timeseries'] = [0.6799509503372164, 0.6548129981606376, 0.6146535867565911, 0.5781729000613121, 0.5361741263028816,
                                          0.5128755364806867, 0.49724095646842426, 0.47884733292458614, 0.4776210913549969, 0.4742489270386266, 0.47455548743102394]
robustness['MI']['mimic 1 timeseries'] = [0.9132434089515634, 0.8963825873697119, 0.8874923359901901, 0.8669527896995708,
                                          0.858982219497241, 0.8421213979153893, 0.8381361128142244, 0.8274064990803188, 0.8274064990803188, 0.8267933782955242, 0.8307786633966892]
robustness['MI']['enrico image'] = [0.4417808219178082, 0.4417808219178082, 0.4520547945205479, 0.3801369863013699, 0.3767123287671233,
                                    0.3116438356164384, 0.3082191780821918, 0.2534246575342466, 0.2465753424657534, 0.1917808219178082, 0.2054794520547945]
robustness['MI']['enrico wireframe image'] = [0.4417808219178082, 0.4452054794520548, 0.4452054794520548, 0.4383561643835616,
                                              0.4246575342465753, 0.3732876712328767, 0.3801369863013699, 0.3424657534246575, 0.339041095890411, 0.3116438356164384, 0.2945205479452055]
robustness['MI']['imdb image micro'] = [0.5777196044211751, 0.5597235643796304, 0.5456367924528303, 0.5246357929645861, 0.5128129341175776,
                                        0.5013947415276871, 0.4877021213469811, 0.4838719267473318, 0.47399221697412586, 0.47339035400332624, 0.47035175879396984]
robustness['MI']['imdb image macro'] = [0.47106023328788327, 0.4443405925674904, 0.4289366994431545, 0.3960658264667671, 0.38540966251686526,
                                        0.36999305870748783, 0.3532958793302592, 0.3492110891393618, 0.33855555565860534, 0.33416640973057593, 0.33218310996819445]
robustness['MI']['imdb text micro'] = [0.5780555151867461, 0.5725688982112683, 0.5678218689731752, 0.5628128934039874, 0.5532405355084671,
                                       0.5431856671717622, 0.5308938212357528, 0.5157522016766033, 0.4950763170851797, 0.4713371801304566, 0.4420206659012629]
robustness['MI']['imdb text macro'] = [0.4733887585319076, 0.46442199555777214, 0.45819610193732296, 0.4469055769222182, 0.43540552917112907,
                                       0.4255376351275636, 0.405264874994762, 0.38485015695983643, 0.35577654905055317, 0.31855399746961166, 0.2784701119483579]

robustness['MVAE'] = dict()
robustness['MVAE']['mimic mortality timeseries'] = [0.7823421213979154, 0.7792765174739423, 0.775904353157572, 0.7673206621704476,
                                                    0.759656652360515, 0.76364193746168, 0.7605763335377069, 0.7608828939301042, 0.7578172900061312, 0.7608828939301042, 0.7608828939301042]
robustness['MVAE']['mimic 7 timeseries'] = [0.6842427958307786, 0.6646229307173513, 0.628755364806867, 0.5800122624156959, 0.5300429184549357,
                                            0.5104230533415083, 0.49233599019006746, 0.48283261802575106, 0.47670141017780504, 0.47455548743102394, 0.47455548743102394]
robustness['MVAE']['mimic 1 timeseries'] = [0.9153893316983446, 0.8988350705088902, 0.882280809319436, 0.8746167995095033,
                                            0.8562231759656652, 0.8464132434089515, 0.8384426732066217, 0.8277130594727161, 0.8261802575107297, 0.8267933782955242, 0.8307786633966892]

robustness['MFM'] = dict()
robustness['MFM']['mimic mortality timeseries'] = [0.7786633966891477, 0.776824034334764, 0.7691600245248313, 0.7660944206008584,
                                                   0.7627222562844881, 0.7621091354996934, 0.7581238503985285, 0.7611894543225015, 0.7605763335377069, 0.7605763335377069, 0.7608828939301042]
robustness['MFM']['mimic 7 timeseries'] = [0.6882280809319435, 0.6532801961986512, 0.5901287553648069, 0.545677498467198, 0.5052115266707541,
                                           0.49478847332924586, 0.48313917841814835, 0.47854077253218885, 0.4742489270386266, 0.4742489270386266, 0.47455548743102394]
robustness['MFM']['mimic 1 timeseries'] = [0.9153893316983446, 0.8936235438381361, 0.8779889638258737, 0.8556100551808706,
                                           0.8461066830165542, 0.8378295524218271, 0.8323114653586756, 0.8280196198651134, 0.8295524218271, 0.8274064990803188, 0.8307786633966892]
robustness['MFM']['imdb image micro'] = [0.38080951341915775, 0.3777138945927447, 0.36800526662277816, 0.36441191039682974, 0.35804172645987425,
                                         0.3564034797929743, 0.3492758038162231, 0.3538559451008606, 0.34850880843390203, 0.35101904257395383, 0.34142525909992455]
robustness['MFM']['imdb image macro'] = [0.23147720015398293, 0.22821069574434785, 0.21837752245395167, 0.21319344751087604, 0.20880684703683933,
                                         0.20376780642781955, 0.19708015707914733, 0.20049007733845628, 0.19841494228815373, 0.19973477304004272, 0.19094900293499076]
robustness['MFM']['imdb text micro'] = [0.38452718934265123, 0.3802497472470421, 0.3797620020150859, 0.3744429559559286, 0.3734649122807017,
                                        0.3703154600115312, 0.36673633263667366, 0.36132666869183405, 0.3568707558107437, 0.3486650586376112, 0.34062057279368874]
robustness['MFM']['imdb text macro'] = [0.23474928943826315, 0.2302628290785667, 0.2304606193666061, 0.22282105747491562, 0.2220746144844292,
                                        0.2174948192952526, 0.2170391861559183, 0.21087324573974095, 0.20374315992778708, 0.1938992262552609, 0.18419838326110158]

robustness['EF'] = dict()
robustness['EF']['finance F&B timeseries'] = [1.66147058, 1.71470122, 1.7545939,
                                              1.83128624, 1.74875638, 1.8410166, 2.0659102, 2.16126256, 2.18665128]
robustness['EF']['finance tech timeseries'] = [0.5731123566627503, 0.5743420124053955, 0.5756734490394593,
                                               0.5752590298652649, 0.576827096939087, 0.5775237202644348, 0.577285373210907, 0.5777166843414306, 0.5782142043113708]
robustness['EF']['finance health timeseries'] = [0.13770209550857543, 0.13934325873851777, 0.1396014302968979,
                                                 0.1413800001144409, 0.14153531789779664, 0.14134591519832612, 0.1417454242706299, 0.14189106822013856, 0.142113533616066]
robustness['EF']['imdb image micro'] = [0.5960651364419497, 0.578808170615047, 0.5637010676156583, 0.5386240629793639, 0.5178299120234603,
                                        0.5061349693251533, 0.48732428210981615, 0.4736166309996943, 0.4640074211502783, 0.4536528749333417, 0.4450398724082935]
robustness['EF']['imdb image macro'] = [0.5066658310319822, 0.4837641916821099, 0.4680108460488566, 0.4370470195309652, 0.4054666167012437,
                                        0.3913335625867006, 0.3670020411244168, 0.3430957686205422, 0.3330902784425957, 0.3114770253237497, 0.2935699838020227]
robustness['EF']['imdb text micro'] = [0.5980051676714916, 0.5925781576236743, 0.5855048177628822, 0.580581980973699, 0.5746317005154784,
                                       0.5625459273076706, 0.5513805863933959, 0.5353877996895303, 0.5214188346529771, 0.49692511402089157, 0.47215155777691314]
robustness['EF']['imdb text macro'] = [0.5108359796999509, 0.5022868499651811, 0.4948879578249415, 0.48283684196058596, 0.47441204981099083,
                                       0.4603006708341554, 0.44851372077004237, 0.4211327682480375, 0.4015704831254156, 0.3654004933537068, 0.3309278872825064]
robustness['EF']['mosi multimodal'] = [0.736, 0.6901, 0.6023,
                                       0.5245, 0.556, 0.4951, 0.4892, 0.456, 0.4642, 0.4302]
robustness['EF']['mosi text'] = [0.736, 0.6811, 0.6234,
                                 0.5217, 0.5197, 0.4987, 0.4629, 0.4581, 0.453, 0.4415]
robustness['EF']['mosi audio'] = [0.736, 0.6945, 0.6536,
                                  0.6015, 0.5893, 0.5487, 0.5329, 0.511, 0.4532, 0.4615]
robustness['EF']['mosi image'] = [0.736, 0.6892, 0.661,
                                  0.6002, 0.5912, 0.5515, 0.5315, 0.453, 0.4721, 0.4645]
robustness['EF']['mosei multimodal'] = [0.7866, 0.6941, 0.6007,
                                        0.5231, 0.5528, 0.4964, 0.4876, 0.4556, 0.4617, 0.436]
robustness['EF']['mosei text'] = [0.7866, 0.6947, 0.5998,
                                  0.5476, 0.5564, 0.4903, 0.5003, 0.4807, 0.459, 0.4441]
robustness['EF']['mosei audio'] = [0.7866, 0.6996, 0.6596,
                                   0.6046, 0.6443, 0.5547, 0.5393, 0.5239, 0.4536, 0.4603]
robustness['EF']['mosei image'] = [0.7866, 0.6922, 0.655,
                                   0.603, 0.5928, 0.5419, 0.5401, 0.508, 0.4528, 0.4674]
robustness['EF']['sarcasm multimodal'] = [0.6654, 0.6013, 0.554,
                                          0.5275, 0.4507, 0.4397, 0.443, 0.4007, 0.3926, 0.3517]
robustness['EF']['sarcasm text'] = [0.6654, 0.6002, 0.5679,
                                    0.5158, 0.4421, 0.4476, 0.4583, 0.3929, 0.3913, 0.3341]
robustness['EF']['sarcasm audio'] = [0.6654, 0.6072, 0.6145,
                                     0.5211, 0.4228, 0.4501, 0.5615, 0.3911, 0.3952, 0.3416]
robustness['EF']['sarcasm image'] = [0.6654, 0.606, 0.552,
                                     0.643, 0.4374, 0.4055, 0.467, 0.4057, 0.3671, 0.3577]
robustness['EF']['humor multimodal'] = [0.6055, 0.5691, 0.5263,
                                        0.4962, 0.5345, 0.5216, 0.471, 0.3844, 0.3196, 0.3046]
robustness['EF']['humor text'] = [0.6055, 0.572, 0.5266,
                                  0.4938, 0.5347, 0.5203, 0.469, 0.3937, 0.3589, 0.3027]
robustness['EF']['humor audio'] = [0.6055, 0.6072, 0.6145,
                                   0.5211, 0.4228, 0.4501, 0.5615, 0.3911, 0.3952, 0.3416]
robustness['EF']['humor image'] = [0.6055, 0.5664, 0.5341,
                                   0.4942, 0.5311, 0.5226, 0.4735, 0.384, 0.3216, 0.3068]
robustness['EF']['gentle push image'] = [0.33170643980000003, 9.053995443, 9.537654717999999,
                                         9.772067782, 9.801991497000001, 9.927901372000001, 10.04480793, 10.09950978, 10.16860172, 10.21986848]
robustness['EF']['gentle push proprio'] = [0.33170643980000003, 5.838992061, 6.571795754, 6.791380137000001,
                                           6.821320997999999, 6.793797229, 6.677612456, 6.4099902360000005, 6.188345958999999, 6.205650488]
robustness['EF']['gentle push haptics'] = [0.33170643980000003, 0.3502492059, 0.4118775333,
                                           0.476365505, 0.527892331, 0.5594398699, 0.5658528965, 0.5494982986, 0.4978467962, 0.42436661649999996]
robustness['EF']['gentle push controls'] = [0.33170643980000003, 0.3502492059, 0.4118775333,
                                            0.476365505, 0.527892331, 0.5594398699, 0.5658528965, 0.5494982986, 0.4978467962, 0.42436661649999996]
robustness['EF']['gentle push multimodal'] = [0.33170643980000003, 5.951489512, 6.887291512000001,
                                              7.255283361, 7.562531842, 7.976519149, 8.659514973, 9.633260381, 10.98019031, 12.69156413]
robustness['EF-Transformer'] = dict()
robustness['EF-Transformer']['finance F&B timeseries'] = [2.157018041610718, 2.145638656616211, 2.2730966091156004,
                                                          2.339062786102295, 2.3593160629272463, 2.318608617782593, 2.4487050056457518, 2.346283531188965, 2.5139325141906737]
robustness['EF-Transformer']['finance health timeseries'] = [0.1417102575302124, 0.14171962440013885, 0.1417176216840744,
                                                             0.14173041582107543, 0.14174440503120422, 0.14172920286655427, 0.14176398515701294, 0.1417480707168579, 0.14174745082855225]
robustness['EF-Transformer']['finance tech timeseries'] = [0.5761273384094239, 0.5761261582374573, 0.5761314392089844,
                                                           0.5761253833770752, 0.5761243581771851, 0.5761215567588807, 0.5761086583137512, 0.57608722448349, 0.5760318040847778]
robustness['EF-Transformer']['mosi multimodal'] = [0.7902, 0.7089,
                                                   0.6502, 0.5857, 0.5421, 0.4862, 0.4862, 0.4723, 0.4601, 0.4593]
robustness['EF-Transformer']['mosi text'] = [0.7902, 0.7282,
                                             0.6612, 0.6039, 0.5616, 0.4903, 0.4603, 0.4521, 0.4535, 0.4542]
robustness['EF-Transformer']['mosi audio'] = [0.7902, 0.7182,
                                              0.6933, 0.6537, 0.6428, 0.5412, 0.5238, 0.5041, 0.483, 0.4859]
robustness['EF-Transformer']['mosi image'] = [0.7902, 0.7161,
                                              0.6821, 0.6456, 0.6428, 0.6212, 0.5738, 0.5757, 0.5515, 0.503]
robustness['EF-Transformer']['mosei multimodal'] = [0.7915, 0.7155,
                                                    0.6493, 0.585, 0.5443, 0.4878, 0.4859, 0.4755, 0.4602, 0.4564]
robustness['EF-Transformer']['mosei text'] = [0.7915, 0.7071,
                                              0.6503, 0.5842, 0.5489, 0.4819, 0.4833, 0.4709, 0.4599, 0.461]
robustness['EF-Transformer']['mosei audio'] = [0.7915, 0.7183,
                                               0.6946, 0.6548, 0.6444, 0.5389, 0.5257, 0.5048, 0.4817, 0.4867]
robustness['EF-Transformer']['mosei image'] = [0.7915, 0.725,
                                               0.7036, 0.6717, 0.6496, 0.5409, 0.5266, 0.5135, 0.4679, 0.4782]
robustness['EF-Transformer']['sarcasm multimodal'] = [0.6548, 0.6192,
                                                      0.6032, 0.5643, 0.5492, 0.4526, 0.4228, 0.4009, 0.3922, 0.3884]
robustness['EF-Transformer']['sarcasm text'] = [0.6548, 0.6305,
                                                0.5853, 0.575, 0.5532, 0.3903, 0.4049, 0.4097, 0.3962, 0.3787]
robustness['EF-Transformer']['sarcasm audio'] = [0.6548, 0.632,
                                                 0.6181, 0.55, 0.5542, 0.4506, 0.4272, 0.3908, 0.3823, 0.3853]
robustness['EF-Transformer']['sarcasm image'] = [0.6548, 0.5974,
                                                 0.6185, 0.5621, 0.5543, 0.5076, 0.4172, 0.4408, 0.3858, 0.3646]
robustness['EF-Transformer']['humor multimodal'] = [0.6288, 0.5049,
                                                    0.5327, 0.5141, 0.5154, 0.4902, 0.4689, 0.3459, 0.4279, 0.4487]
robustness['EF-Transformer']['humor text'] = [0.6288, 0.5149,
                                              0.5304, 0.5089, 0.5144, 0.4916, 0.4713, 0.3472, 0.4266, 0.4552]
robustness['EF-Transformer']['humor audio'] = [0.6288, 0.632,
                                               0.6181, 0.55, 0.5542, 0.4506, 0.4272, 0.3908, 0.3823, 0.3853]
robustness['EF-Transformer']['humor image'] = [0.6288, 0.5032,
                                               0.5426, 0.5195, 0.5136, 0.4856, 0.469, 0.3417, 0.4231, 0.4497]

robustness['GradBlend'] = dict()
robustness['GradBlend']['finance F&B timeseries'] = [1.5008137702941895, 1.5597750902175904, 1.4509791374206542,
                                                     1.7539066553115845, 1.5992064476013184, 2.0632896900177, 2.180710029602051, 2.122863674138183, 2.183281755447388]
robustness['GradBlend']['finance tech timeseries'] = [0.5373285174369812, 0.5378936171531677, 0.5392054557800293,
                                                      0.5571645736694336, 0.5736293792724609, 0.5806585907936096, 0.5778189182281495, 0.5982250094413757, 0.5962010025978088]
robustness['GradBlend']['finance health timeseries'] = [0.13074953705072404, 0.13255454301834108, 0.13356374502182006,
                                                        0.1311644658446312, 0.13756160736083983, 0.13639299869537352, 0.14471893310546874, 0.14702820181846618, 0.14836240112781524]
robustness['GradBlend']['enrico image'] = [0.4897260273972603, 0.4486301369863014, 0.410958904109589, 0.386986301369863,
                                           0.363013698630137, 0.3116438356164384, 0.23972602739726026, 0.273972602739726, 0.273972602739726, 0.2602739726027397, 0.1917808219178082]
robustness['GradBlend']['enrico wireframe image'] = [0.4965753424657534, 0.4657534246575342, 0.4383561643835616, 0.4417808219178082,
                                                     0.4486301369863014, 0.4280821917808219, 0.4417808219178082, 0.4041095890410959, 0.3972602739726027, 0.3972602739726027, 0.3767123287671233]
robustness['GradBlend']['mosi multimodal'] = [0.7543, 0.7122,
                                              0.6568, 0.6443, 0.6179, 0.532, 0.5014, 0.4823, 0.4902, 0.4899]
robustness['GradBlend']['mosi text'] = [0.7543, 0.7233, 0.7003,
                                        0.6487, 0.6122, 0.5809, 0.5784, 0.5528, 0.5029, 0.4802]
robustness['GradBlend']['mosi audio'] = [0.7543, 0.7235, 0.7045,
                                         0.6804, 0.6328, 0.6022, 0.5812, 0.5435, 0.5689, 0.563]
robustness['GradBlend']['mosi image'] = [0.7543, 0.7244,
                                         0.6989, 0.6641, 0.6502, 0.598, 0.5725, 0.5738, 0.5522, 0.55]
robustness['GradBlend']['mosei multimodal'] = [0.7732, 0.7115,
                                               0.6312, 0.6447, 0.6167, 0.5312, 0.5024, 0.4711, 0.4992, 0.4878]
robustness['GradBlend']['mosei text'] = [0.7732, 0.7111, 0.6559,
                                         0.6415, 0.6154, 0.5299, 0.5032, 0.4828, 0.4885, 0.4952]
robustness['GradBlend']['mosei audio'] = [0.7732, 0.722, 0.7041,
                                          0.6824, 0.6322, 0.6007, 0.5814, 0.541, 0.5683, 0.5026]
robustness['GradBlend']['mosei image'] = [0.7732, 0.7203,
                                          0.7043, 0.6821, 0.6383, 0.6055, 0.574, 0.55, 0.5312, 0.4653]
robustness['GradBlend']['sarcasm multimodal'] = [0.6578, 0.6234,
                                                 0.6415, 0.4569, 0.5348, 0.4931, 0.49, 0.3999, 0.3607, 0.3187]
robustness['GradBlend']['sarcasm text'] = [0.6578, 0.6393,
                                           0.6503, 0.4866, 0.5295, 0.5148, 0.484, 0.3942, 0.3664, 0.3126]
robustness['GradBlend']['sarcasm audio'] = [0.6578, 0.6358,
                                            0.6285, 0.4557, 0.5133, 0.4917, 0.4775, 0.3992, 0.3433, 0.3279]
robustness['GradBlend']['sarcasm image'] = [0.6578, 0.5982,
                                            0.6297, 0.4398, 0.533, 0.4993, 0.4986, 0.3962, 0.3629, 0.2961]
robustness['GradBlend']['humor multimodal'] = [0.6251, 0.5859,
                                               0.6021, 0.6046, 0.5577, 0.4607, 0.3817, 0.3513, 0.364, 0.3822]
robustness['GradBlend']['humor text'] = [0.6251, 0.5864, 0.6033,
                                         0.6045, 0.5576, 0.4604, 0.3835, 0.3538, 0.3646, 0.3864]
robustness['GradBlend']['humor audio'] = [0.6251, 0.6358, 0.6285,
                                          0.4557, 0.5133, 0.4917, 0.4775, 0.3992, 0.3433, 0.3279]
robustness['GradBlend']['humor image'] = [0.6251, 0.5972, 0.603,
                                          0.6216, 0.5622, 0.5695, 0.3793, 0.3549, 0.363, 0.4561]

robustness['MulT'] = dict()
robustness['MulT']['finance F&B timeseries'] = [2.0660665035247803, 2.091330051422119, 2.1647129535675047,
                                                2.153086709976196, 2.117071104049683, 2.205156850814819, 2.1811957359313965, 2.232287311553955, 2.2354227542877196]
robustness['MulT']['finance tech timeseries'] = [0.5653551816940308, 0.5692847371101379, 0.5836835503578186,
                                                 0.5806204080581665, 0.5986919403076172, 0.6019670367240906, 0.6118748188018799, 0.6261817812919617, 0.6223957538604736]
robustness['MulT']['finance health timeseries'] = [0.13463806509971618, 0.13365633189678192, 0.1398583859205246,
                                                   0.14236030280590056, 0.14418508410453795, 0.14859049916267394, 0.14711943566799163, 0.14752028584480287, 0.150432026386261]
robustness['MulT']['mosi multimodal'] = [0.8337, 0.7488, 0.6693,
                                         0.6526, 0.5884, 0.5206, 0.4979, 0.4965, 0.4995, 0.4858]
robustness['MulT']['mosi text'] = [0.8337, 0.7524, 0.7048,
                                   0.6659, 0.6302, 0.5706, 0.5821, 0.5317, 0.5032, 0.4921]
robustness['MulT']['mosi audio'] = [0.8337, 0.753, 0.7139,
                                    0.6833, 0.6314, 0.6062, 0.6121, 0.5862, 0.5923, 0.5736]
robustness['MulT']['mosi image'] = [0.8337, 0.7402, 0.714,
                                    0.681, 0.6521, 0.6014, 0.6124, 0.5723, 0.5987, 0.5524]
robustness['MulT']['mosei multimodal'] = [0.8152, 0.7505,
                                          0.6632, 0.6454, 0.589, 0.5226, 0.5028, 0.4958, 0.496, 0.4525]
robustness['MulT']['mosei text'] = [0.8152, 0.7091, 0.6689,
                                    0.6543, 0.5886, 0.5222, 0.5036, 0.5017, 0.4979, 0.4847]
robustness['MulT']['mosei audio'] = [0.8152, 0.7491, 0.6913,
                                     0.6878, 0.6314, 0.6047, 0.6136, 0.5824, 0.594, 0.5043]
robustness['MulT']['mosei image'] = [0.8152, 0.7085, 0.6628,
                                     0.6476, 0.6218, 0.6052, 0.6073, 0.5853, 0.5068, 0.4749]
robustness['MulT']['sarcasm multimodal'] = [0.7023, 0.6421,
                                            0.6184, 0.5729, 0.528, 0.5084, 0.5116, 0.4842, 0.4527, 0.3707]
robustness['MulT']['sarcasm text'] = [0.7023, 0.6332, 0.6108,
                                      0.4837, 0.5291, 0.5173, 0.5046, 0.4765, 0.4852, 0.3998]
robustness['MulT']['sarcasm audio'] = [0.7023, 0.6368, 0.6281,
                                       0.5595, 0.5244, 0.5214, 0.5212, 0.4917, 0.4521, 0.367]
robustness['MulT']['sarcasm image'] = [0.7023, 0.6453, 0.6008,
                                       0.5585, 0.5191, 0.5041, 0.5219, 0.4776, 0.4515, 0.3196]
robustness['MulT']['humor multimodal'] = [0.6667, 0.5071, 0.5201,
                                          0.5019, 0.4637, 0.4125, 0.3906, 0.4165, 0.4124, 0.3252]
robustness['MulT']['humor text'] = [0.6667, 0.5062, 0.5312,
                                    0.5027, 0.4653, 0.4147, 0.3935, 0.4134, 0.4136, 0.3275]
robustness['MulT']['humor audio'] = [0.6667, 0.6368, 0.6281,
                                     0.5595, 0.5244, 0.5214, 0.5212, 0.4917, 0.4521, 0.367]
robustness['MulT']['humor image'] = [0.6667, 0.5032, 0.5189,
                                     0.4923, 0.4657, 0.4074, 0.3932, 0.4194, 0.4109, 0.3275]

robustness['CCA'] = dict()
robustness['CCA']['enrico image'] = [0.5205479452054794, 0.4726027397260274, 0.4657534246575342, 0.410958904109589, 0.2910958904109589,
                                     0.2636986301369863, 0.20205479452054795, 0.1267123287671233, 0.1232876712328767, 0.04794520547945205, 0.030821917808219176]
robustness['CCA']['enrico wireframe image'] = [0.5205479452054794, 0.4417808219178082, 0.4212328767123288, 0.386986301369863, 0.339041095890411,
                                               0.22945205479452055, 0.19863013698630136, 0.17465753424657535, 0.0821917808219178, 0.06164383561643835, 0.02054794520547945]
robustness['CCA']['imdb image micro'] = [0.5979549412888365, 0.579187598679489, 0.5572366877785597, 0.5324244545205563,
                                         0.4986794113947931, 0.4728051114066098, 0.4416219839142091, 0.4219911870008262, 0.39983132445444, 0.3870092790863669, 0.3756657849922099]
robustness['CCA']['imdb image macro'] = [0.5052922945325112, 0.4774283379013967, 0.4532107402427913, 0.418973491941983, 0.38054600150333684,
                                         0.35749543061204075, 0.32890581051077594, 0.3142437600675067, 0.29028787184588, 0.2775146534809951, 0.2647465350449697]
robustness['CCA']['imdb text micro'] = [0.598883371208933, 0.5941217831528618, 0.5879466426013142, 0.582861350249207, 0.5741235297460084,
                                        0.5654620768461649, 0.5526694695617117, 0.5325935393911224, 0.5167864294969962, 0.4955762757305848, 0.46750739154045734]
robustness['CCA']['imdb text macro'] = [0.5058683324335825, 0.49697946097057644, 0.4878780921741972, 0.4788777504454515, 0.46609929940466605,
                                        0.45284790176161155, 0.43065018443821895, 0.40027685907861377, 0.37174486267299517, 0.3325863279594893, 0.2827813687079348]

robustness['TF'] = dict()
robustness['TF']['enrico image'] = [0.5, 0.4349315068493151, 0.3801369863013699, 0.3424657534246575, 0.2842465753424658,
                                    0.2226027397260274, 0.1678082191780822, 0.1541095890410959, 0.08904109589041095, 0.0821917808219178, 0.04794520547945205]
robustness['TF']['enrico wireframe image'] = [0.5, 0.4417808219178082, 0.386986301369863, 0.3219178082191781, 0.2534246575342466,
                                              0.21575342465753425, 0.18835616438356165, 0.1678082191780822, 0.10616438356164383, 0.07876712328767123, 0.04794520547945205]
robustness['TF']['gentle push image'] = [0.5139880641, 283.0832693, 385.2962356, 456.36261060000004,
                                         515.0258595, 561.080705, 585.566162, 595.8546951000001, 584.152784, 556.276079]
robustness['TF']['gentle push proprio'] = [0.5139880641, 2907.2389239999998, 4791.740975,
                                           6813.597632999999, 9165.042824, 11276.50928, 12668.8312, 13407.66258, 13144.31753, 10564.76569]
robustness['TF']['gentle push haptics'] = [0.5139880641, 2.7141630589999997, 5.188060706,
                                           7.317023043, 9.06796194, 10.43746778, 11.17741652, 11.35476332, 10.61318271, 8.339987006]
robustness['TF']['gentle push controls'] = [0.5139880641, 150.6061621, 290.7440733, 413.71387769999995,
                                            521.8158717, 601.4511812999999, 650.1761617000001, 664.2449238, 629.1892726, 499.4921552]
robustness['TF']['gentle push multimodal'] = [0.5139880641, 728.0306158, 3483.39041,
                                              8962.23025, 16685.06471, 25265.4308, 32007.18022, 30971.03163, 25212.46658, 10955.77158]

robustness['ReFNet'] = dict()
robustness['ReFNet']['imdb image micro'] = [0.5976573446169366, 0.5835463029432879, 0.5616664699634132, 0.5364713027634377,
                                            0.5107123977735627, 0.4892952720785014, 0.4682876290996212, 0.4597912426967767, 0.4498865914992932, 0.4487830618033111, 0.4477985763248089]
robustness['ReFNet']['imdb image macro'] = [0.4999160237577528, 0.47935456745475646, 0.45167922260039106, 0.4226309022780442,
                                            0.3877997424348276, 0.3661711453517022, 0.3477797280117274, 0.33301180353866633, 0.31998692522679895, 0.3046172757682068, 0.3029995643603409]
robustness['ReFNet']['imdb text micro'] = [0.599182805328557, 0.5955059331781076, 0.5918183861541919, 0.5834299120897758,
                                           0.5769263613332953, 0.5667840073266363, 0.5553725694143481, 0.5403148088660457, 0.5210393466963623, 0.4958891733172714, 0.4671595810227973]
robustness['ReFNet']['imdb text macro'] = [0.5008901973258216, 0.49699338944114185, 0.49214286719209255, 0.4778059110798838,
                                           0.4682993832966679, 0.4537934006137857, 0.437817596921358, 0.41052447100609374, 0.37963331512004456, 0.3363953103962002, 0.28299847974068115]
robustness['ReFNet']['robotics image'] = [2.4650e-05, 0.0003,
                                          0.0005, 0.0006, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007]
robustness['ReFNet']['robotics proprio'] = [2.4424e-05, 0.0603,
                                            0.1169, 0.1681, 0.2093, 0.2449, 0.2746, 0.2979, 0.3145, 0.3246]
robustness['ReFNet']['robotics force'] = [2.4938e-05, 2.4861e-05, 2.4803e-05,
                                          2.4877e-05, 2.5277e-05, 2.4626e-05, 2.5879e-05, 2.4421e-05, 2.4932e-05, 2.5057e-05]

robustness['RMFE'] = dict()
robustness['RMFE']['imdb image micro'] = [0.6024415524588276, 0.5827195551217204, 0.5569574184156254, 0.5370364604415107, 0.5113943362044306,
                                          0.49437473656907566, 0.48330265579805415, 0.47117777851524906, 0.4625138371741974, 0.455428126797253, 0.4446875427321209]
robustness['RMFE']['imdb image macro'] = [0.47716090909756703, 0.45046200743638426, 0.4243874262395849, 0.3935334619561985,
                                          0.3583584701200579, 0.3297657981864145, 0.312041038651002, 0.29152674372139775, 0.2786631693242129, 0.2640122413543502, 0.24515909662609933]
robustness['RMFE']['imdb text micro'] = [0.6027342179332529, 0.5976063446286951, 0.5948144089060652, 0.5873894767357588, 0.5801793408640968,
                                         0.5719892221180881, 0.5623299822590184, 0.5469268234380136, 0.5314249402366327, 0.5086175058475932, 0.48454385524001004]
robustness['RMFE']['imdb text macro'] = [0.4772888994608969, 0.4727183926954104, 0.4656519534553542, 0.45563175396079997, 0.44373609975172396,
                                         0.42812249931047186, 0.4107754418873646, 0.3904075948756661, 0.35949643390342695, 0.32367189293067294, 0.28797348144513757]

robustness['MCTN'] = dict()
robustness['MCTN']['mosi multimodal'] = [0.779, 0.724, 0.6832,
                                         0.6593, 0.5609, 0.5032, 0.4821, 0.4633, 0.4645, 0.443]
robustness['MCTN']['mosi text'] = [0.779, 0.72, 0.6721,
                                   0.6554, 0.5632, 0.5021, 0.4789, 0.4633, 0.4611, 0.4412]
robustness['MCTN']['mosi audio'] = [0.779, 0.779, 0.779,
                                    0.779, 0.779, 0.779, 0.779, 0.779, 0.779, 0.779]
robustness['MCTN']['mosi image'] = [0.779, 0.779, 0.779,
                                    0.779, 0.779, 0.779, 0.779, 0.779, 0.779, 0.779]
robustness['MCTN']['mosei multimodal'] = [0.761, 0.7251, 0.6847,
                                          0.6574, 0.5616, 0.5008, 0.4856, 0.4656, 0.4533, 0.4446]
robustness['MCTN']['mosei text'] = [0.761, 0.7249, 0.6523,
                                    0.6589, 0.5624, 0.5045, 0.4859, 0.4631, 0.4599, 0.4428]
robustness['MCTN']['mosei audio'] = [0.761, 0.761, 0.761,
                                     0.761, 0.761, 0.761, 0.761, 0.761, 0.761, 0.761]
robustness['MCTN']['mosei image'] = [0.761, 0.761, 0.761,
                                     0.761, 0.761, 0.761, 0.761, 0.761, 0.761, 0.761]
robustness['MCTN']['sarcasm multimodal'] = [0.6342, 0.6022,
                                            0.5703, 0.5665, 0.4844, 0.4495, 0.5125, 0.3915, 0.3992, 0.3016]
robustness['MCTN']['sarcasm text'] = [0.6342, 0.6576, 0.5265,
                                      0.5789, 0.4313, 0.4843, 0.5053, 0.3957, 0.3603, 0.3042]
robustness['MCTN']['sarcasm audio'] = [0.6342, 0.6342, 0.6342,
                                       0.6342, 0.6342, 0.6342, 0.6342, 0.6342, 0.6342, 0.6342]
robustness['MCTN']['sarcasm image'] = [0.6342, 0.6342, 0.6342,
                                       0.6342, 0.6342, 0.6342, 0.6342, 0.6342, 0.6342, 0.6342]
robustness['MCTN']['humor multimodal'] = [0.6312, 0.5378, 0.5044,
                                          0.4946, 0.5089, 0.3638, 0.4191, 0.4027, 0.4437, 0.3839]
robustness['MCTN']['humor text'] = [0.6312, 0.5382, 0.5209,
                                    0.4941, 0.5034, 0.3635, 0.4172, 0.3969, 0.442, 0.3851]
robustness['MCTN']['humor audio'] = [0.6312, 0.6312, 0.6312,
                                     0.6312, 0.6312, 0.6312, 0.6312, 0.6312, 0.6312, 0.6312]
robustness['MCTN']['humor image'] = [0.6312, 0.6312, 0.6312,
                                     0.6312, 0.6312, 0.6312, 0.6312, 0.6312, 0.6312, 0.6312]

robustness['MFAS'] = dict()
robustness['MFAS']['mimic mortality timeseries'] = [0.7614960147148988, 0.7614960147148988, 0.7608828939301042, 0.7611894543225015,
                                                    0.7599632127529123, 0.7611894543225015, 0.7605763335377069, 0.7608828939301042, 0.7605763335377069, 0.7608828939301042, 0.7608828939301042]
robustness['MFAS']['mimic 7 timeseries'] = [0.6879215205395462, 0.6397915389331699, 0.5996321275291232, 0.5748007357449417,
                                            0.5389331698344574, 0.5174739423666462, 0.5052115266707541, 0.4865113427345187, 0.48038013488657266, 0.4757817290006131, 0.47455548743102394]
robustness['MFAS']['mimic 1 timeseries'] = [0.9141630901287554, 0.8969957081545065, 0.8776824034334764, 0.8611281422440221,
                                            0.8531575720416922, 0.8408951563458001, 0.835683629675046, 0.8310852237890864, 0.8274064990803188, 0.8277130594727161, 0.8307786633966892]

robustness['SF'] = dict()
robustness['SF']['robotics image'] = [2.5372693926328793e-05, 0.00013537800987251103, 0.0002216917637269944, 0.0002810211735777557,
                                      0.00031372791272588074, 0.000328845955664292, 0.0003345597942825407, 0.0003262482350692153, 0.00031598060741089284, 0.0003046182682737708]
robustness['SF']['robotics proprio'] = [2.5808196369325742e-05, 0.07510929554700851, 0.14351879060268402, 0.20326630771160126,
                                        0.2524733245372772, 0.2911565601825714, 0.3221781849861145, 0.34433481097221375, 0.35922226309776306, 0.366399347782135]
robustness['SF']['robotics force'] = [2.5194265617756173e-05, 2.5528639525873587e-05, 2.640512684592977e-05, 2.6921765311271884e-05,
                                      2.6588038963382132e-05, 2.632199539220892e-05, 2.5984008971136063e-05, 2.670147478056606e-05, 2.5790803192649037e-05, 2.5958863261621445e-05]

acc_tasks = ['sarcasm text', 'imdb text macro', 'sarcasm multimodal', 'mimic 1 timeseries', 'mimic mortality timeseries', 'imdb text micro', 'mosei image', 'imdb tex macro', 'sarcasm audio', 'mosei text', 'imdb image micro', 'enrico wireframe image',
             'mimic 7 timeseries', 'mosi image', 'mosei multimodal', 'mosi multimodal', 'mosi audio', 'mosi text', 'enrico image', 'sarcasm image', 'mosei audio', 'imdb image macro', 'humor multimodal', 'humor text', 'humor audio', 'humor image']
mse_tasks = ['finance health timeseries', 'finance F&B timeseries', 'finance tech timeseries', 'gentle push image', 'gentle push proprio',
             'gentle push haptics', 'gentle push controls', 'gentle push multimodal', 'robotics image', 'robotics force', 'robotics proprio']
