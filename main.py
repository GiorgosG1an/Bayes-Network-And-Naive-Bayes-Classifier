"""
Main module for the second project for AI.

University: University of Peloponnese, Department of Informatics and Telecommunications\

Course: Artificial Intelligence

Authors: 
- Giannopoulos Georgios
- Giannopoulos Ioannis

Project Description: Implementation of Naive Bayes Classifier for the classification of the spam emails. 
"""
from probability_distribution.probdist import ProbDist
from bayes_networks.bayes_node import BayesNode
from bayes_networks.bayes_net import BayesNet
from bayes_networks.bayes_network_utils import extend, enumerate_all, enumeration_ask

def main() -> None:

    node_specs = [
        ('Taksidevei', [], 0.05), 
        ('Apati', ['Taksidevei'], {True: 0.01, False: 0.004}),  # Apati 
        ('AgoraEksoterikou', ['Apati', 'Taksidevei'], {(True, True): 0.90, (True, False): 0.10, (False, True): 0.90, (False, False): 0.01}),  # AgoraEksoterikou 
        ('DiatheteiIpologisti', [], 0.60),  # DiatheteiIpologisti 
        ('AgoraDiadiktiou', ['Apati', 'DiatheteiIpologisti'], {(True, True): 0.02, (True, False): 0.011, (False, True): 0.01, (False, False): 0.001}),  # AgoraDiadiktiou 
        ('AgoraSxetikiMeIpologisti', ['DiatheteiIpologisti'], {True: 0.10, False: 0.001})  # AgoraSxetikiMeIpologisti 
    ]

    bayes_net = BayesNet(node_specs)
    print("\t\tDiktyo Bayes\n",bayes_net)

    # Question 2.2:
    # 1. Probability that the current transaction is a fraud (P(Apati))
    result_p_f = enumeration_ask('Apati', {}, bayes_net)
    print("P(Apati):", result_p_f.show_approx())

    # 2. Probability that the current transaction is a fraud given evidence (P(Apati | AgoraEksoterikou=True, AgoraDiadiktiou=False, AgoraSxetikiMeIpologisti=True))
    evidence = {'AgoraEksoterikou': True, 'AgoraDiadiktiou': False, 'AgoraSxetikiMeIpologisti': True}
    result_p_apati_given_evidence = enumeration_ask('Apati', evidence, bayes_net)
    print("P(Apati | AgoraEksoterikou=True, AgoraDiadiktiou=False, AgoraSxetikiMeIpologisti=True):", result_p_apati_given_evidence.show_approx())
    
    # 3. Probability that the current transaction is a fraud given evidence (P(Apati | Taksidevei=True, AgoraEksoterikou=True, AgoraDiadiktiou=False, AgoraSxetikiMeIpologisti=True))
    evidence = {'Taksidevei': True, 'AgoraEksoterikou': True, 'AgoraDiadiktiou': False, 'AgoraSxetikiMeIpologisti': True}
    result_p_apati_given_evidence = enumeration_ask('Apati', evidence, bayes_net)
    print("P(Apati | Taksidevei=True, AgoraEksoterikou=True, AgoraDiadiktiou=False, AgoraSxetikiMeIpologisti=True):", result_p_apati_given_evidence.show_approx())
    
if __name__ == "__main__":
    main()
