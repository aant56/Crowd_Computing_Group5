# coding: utf-8

from csv import reader
from unicodecsv  import writer, QUOTE_MINIMAL
import numpy as np

# Selected values for aggregation
MIN_RESPONSES = 3			# Minimum number of annotations for a tweet
OUTLIER_MIN_DISTANCE = 2	# Minimum distance to the mean in order to considere outlier
OUTLIER_RATIO = 0.03		# Labels with less than (e.g.) 3% of the answers can be considered outliers
MIN_KAPPA = 0.1				# Minimum agreement lebel to use the tweet for machine learning


filename = "f1389958.csv"
raw_data = []

# Read Figure Eight data
with open(filename, 'r', encoding="utf-8") as f:
    for ri, row in enumerate(reader(f)):
        if ri==0:
            col_names = row
        else:
            raw_data.append(row)

data = np.array(raw_data)
# print(data.shape)


# Simpler access to columns by name instead of index
col_dict = {}
for i, x in enumerate(col_names):
    col_dict[x] = i
for k in col_dict.keys():
#     print("{:02d}: {}".format(col_dict[k],k))
    pass

# Unique workers that took part in the tasks
workers = np.unique(data[:,col_dict['_worker_id']])
# print(workers)
# print(len(workers))


# Get demographics
demographics = {}
for i in range(data.shape[0]):
    if data[i,col_dict['_worker_id']] not in demographics: # Not already added
        if data[i,col_dict['age_range']] != '': # Demographic data entered
            # Age, gender, region, highest education, own stance, trust in government
            demographics[data[i,col_dict['_worker_id']]] = [data[i,col_dict['age_range']], data[i,col_dict['gender']], data[i,col_dict['general_geographic_region']], data[i,col_dict['highest_education_achieved']], data[i,col_dict['what_is_your_own_stance_on_vaccinations']], data[i,col_dict['how_much_do_you_trust_the_government_of_the_country_you_reside_in']]]


# Given a list of workers that annotated a tweet, give weights to their votes according to their demographics.
# Voting weight is inversely proportional to the size of the groups they belong to
def getWeights(worker_list):
	alpha = 1
    demographic_counts = {}
    smoothing = [0] * len(demographics[worker_list[0]])
    for w in worker_list: # Iterate over all workers
        for i,fi in enumerate(demographics[w]): # Iterate over all features
            if fi in demographic_counts:
                demographic_counts[fi] += 1
            else:
                demographic_counts[fi] = 1
                smoothing[i] += 1
    total = len(worker_list)
    weights = [1] * total
    for wi,w in enumerate(worker_list):
        for di,d in enumerate(demographics[w]):
            weights[wi] *= (total-demographic_counts[d]+alpha)/(total+smoothing[di]*alpha)
    return dict(zip(worker_list, weights))


# for k in demographics.keys():
#     print("{}: {}".format(k, demographics[k]))

# Exclude data from workers who didn't complete the demographic survey, non-relevant tweets and golden standards (they contain attention checks)
filtered_data = []
for i in range(data.shape[0]):
    if data[i,col_dict['_worker_id']] in demographics and data[i,col_dict['_golden']]=='false' and data[i,col_dict['is_this_tweet_relevant_to_the_subject_of_vaccinations']]=='yes':
        filtered_data.append(data[i,:])
filtered_data = np.array(filtered_data)
# print(filtered_data.shape)

# print("{} tweets to begin with".format(len(np.unique(filtered_data[:,col_dict['_unit_id']]))))

filtered_data = filtered_data[filtered_data[:,0].argsort()] # Order by task ID

vote_columns = ['how_pro_vaccinations_is_this_tweet', 'how_against_vaccinations_is_this_tweet', 'how_positive_is_the_sentiment_of_this_text', 'how_negative_is_the_sentiment_of_this_text', 'how_based_on_emotions_are_the_statements_made_in_this_tweet', 'how_based_on_facts_are_the_statements_made_in_this_tweet']
possible_votes = ['1', '2', '3', '4', '5']


f_for = open("labels/for.csv", 'wb')
f_aga = open("labels/against.csv", 'wb')
f_pos = open("labels/positive.csv", 'wb')
f_neg = open("labels/negative.csv", 'wb')
f_emo = open("labels/emotional.csv", 'wb')
f_fac = open("labels/factual.csv", 'wb')
w_for = writer(f_for, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL, encoding='utf-8')
w_aga = writer(f_aga, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL, encoding='utf-8')
w_pos = writer(f_pos, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL, encoding='utf-8')
w_neg = writer(f_neg, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL, encoding='utf-8')
w_emo = writer(f_emo, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL, encoding='utf-8')
w_fac = writer(f_fac, delimiter=',', quotechar='"', quoting=QUOTE_MINIMAL, encoding='utf-8')


not_enough_annotators = 0
bad_kappa = 0

units = np.unique(filtered_data[:,col_dict['_unit_id']]) # Number of tweets
for u in units:
    unit_data = filtered_data[filtered_data[:,col_dict['_unit_id']]==u] # Select all annotations of the same tweet
#     print(unit_data)
    if unit_data.shape[0] >= MIN_RESPONSES:
        worker_list = unit_data[:,col_dict['_worker_id']] # Is already unique, the same worker can't submit it twice
        num_workers = len(worker_list)
        weights = getWeights(worker_list)
        
        # Original votes
        votes = np.zeros((len(possible_votes), len(vote_columns)))
        for ci, c in enumerate(vote_columns):
            for vi, v in enumerate(possible_votes):
                votes[vi,ci] = np.count_nonzero(unit_data[:,col_dict[c]] == v)
        
		# Remove outliers
        for ci in range(votes.shape[1]):
            avg = np.sum(votes[:,ci] * np.arange(len(possible_votes)) / num_workers)
            for vi in range(votes.shape[0]):
                if votes[vi,ci]/num_workers <= OUTLIER_RATIO and np.abs(vi-avg) >= OUTLIER_MIN_DISTANCE: 
                    votes[vi,ci] = 0
		
		# Calculate agreement
        p = np.sum(votes,axis=1) / np.sum(votes)
        P = [float(np.sum(votes[:,ci] * (votes[:,ci]-1))) / (num_workers*(num_workers-1)) for ci in range(votes.shape[1])]
        P_bar = np.mean(P)
        P_e = np.sum(p**2)
        kappa = (P_bar - P_e) / (1 - P_e)
        
        if kappa > MIN_KAPPA: # Minimum consensus has to be reached
            weighted_votes = np.zeros((len(possible_votes), len(vote_columns)))
            for row in unit_data:
                for ci, c in enumerate(vote_columns):
                    weighted_votes[int(row[col_dict[c]])-1,ci] += weights[row[col_dict['_worker_id']]]
#             weighted_votes /= np.sum(weighted_votes,axis=0) # Normalized
            w_for.writerow([unit_data[0,col_dict['text']], np.argmax(weighted_votes[:,0])+1])
            w_aga.writerow([unit_data[0,col_dict['text']], np.argmax(weighted_votes[:,1])+1])
            w_pos.writerow([unit_data[0,col_dict['text']], np.argmax(weighted_votes[:,2])+1])
            w_neg.writerow([unit_data[0,col_dict['text']], np.argmax(weighted_votes[:,3])+1])
            w_emo.writerow([unit_data[0,col_dict['text']], np.argmax(weighted_votes[:,4])+1])
            w_fac.writerow([unit_data[0,col_dict['text']], np.argmax(weighted_votes[:,5])+1])
        else:
            print("Kappa={}".format(kappa))
            bad_kappa += 1
    else:
        not_enough_annotators += 1
print("Less than {} annotators: {}".format(MIN_RESPONSES, not_enough_annotators))
print("Not enough agreement: {}".format(bad_kappa))
f_for.close()
f_aga.close()
f_pos.close()
f_neg.close()
f_emo.close()
f_fac.close()

