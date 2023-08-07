import pandas as pd
import numpy as np
import sys

paths = ["/Users/zone/Desktop/Research_z-Chen/Opensmile/", "C:/Users/zihen/Desktop/Research_z-Chen/Opensmile"]
sys.path.append(paths[1])

import audio_analysis as au

questionnaire_excel = pd.read_excel('test_data/november_elderly_qol_answer_all.xlsx')
questionnaire_results = questionnaire_excel[[
    '氏名',
    '過去1日間に、家族、友人、近所の人、その他の仲間とのふだんのつきあいが、身体的あるいは心理的な理由で、どのくらい妨げられましたか。',
    '過去1日間に、友人や親せきを訪ねるなど、人とのつきあいが、身体的あるいは心理的な理由で、時間的にどのくらい妨げられましたか。']]

print(np.array(questionnaire_results[questionnaire_results["氏名"] == au.name_to_experiment_num[3]]))
