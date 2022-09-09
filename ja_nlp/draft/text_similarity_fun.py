class Gemini:
"""Given two sequences that are mostly the same, class Gemini marks up
differences between the two.



Input: two strings



Output/Available Method Calls:
Define
delta = Gemini(str1, str2)



The following method calls return Numpy arrays of start_indices and
end_indices (inclusive) of intervals:
* delta.get_str1_diff_intervals()
- Intervals of str1's substrings that are Not found in str2
* delta.get_str2_diff_intervals()
- Intervals of str2's substrings that are Not found in str1
* delta.get_str1_match_intervals()
- Intervals of str1's substrings that are also found in str2
* delta.get_str2_match_intervals()
- Intervals of str2's substrings that are also found in str1



For the following method calls, each returned interval is augmented with a
0/1 signal, e.g. [0/1, start, end (inclusive)], where
signal = 0/1
indicates an interval does not/does exist in the other sequence
* delta.get_str1_partition()
- Full partitions of diff/matching intervals in str1 w.r.t. str2
* delta.get_str2_partitions() # Positions in str2
- Full partitions of diff/matching intervals in str2 w.r.t. str1



Other method calls include:
* delta.get_Levenshtein_distance()
- Num of edits to equalize str1 and str2
* delta.get_recall_str1_by_str2():
- Ratio of how much of str1 can be found in str2
* delta.get_recall_str2_by_str1():
- Ratio of how much of str2 can be found in str1
* delta.get_similarity()
- Ratio (between 0 and 1) of how str1 and str2 are similar
"""
def __init__(self, str1, str2):
if str1.strip() == "": str1 = "..."
if str2.strip() == "": str2 = "..."
self.str1_len = len(str1)
self.str2_len = len(str2)
self.Levenshtein = Levenshtein.distance(str1, str2)
tempDict = match_two_sequences(str1, str2)
self.similarity = tempDict["similarity"]
self.str1_matching_intervals = tempDict["str1_matching_intervals"]
self.str2_matching_intervals = tempDict["str2_matching_intervals"]



def get_Levenshtein_distance(self):
"""Return minimum number of edits to equalize str1 and str2"""
return self.Levenshtein



def get_recall_str1_by_str2(self):
"""Return a ratio of how much of str1 can be found in str2"""
if self.str1_matching_intervals: # Non-empty
intervals = np.array(self.str1_matching_intervals)
return np.sum(1 + intervals[:, 1] - intervals[:, 0])/self.str1_len
else:
return 0.0



def get_recall_str2_by_str1(self):
"""Return a ratio of how much of str2 can be found in str1"""
if self.str2_matching_intervals: # Non-empty
intervals = np.array(self.str2_matching_intervals)
return np.sum(1 + intervals[:, 1] - intervals[:, 0])/self.str2_len
else:
return 0.0



def get_similarity(self):
"""Return a ratio of similarity between str1 and str2"""
return self.similarity



def get_str1_diff_intervals(self):
"""Return sub-string intervals (Start/End indices) in str1 that are
Not found in str2"""
return [[b1, b2] for signal, b1, b2 in self.get_str1_partition()
if signal == 0]



def get_str2_diff_intervals(self):
"""Return sub-string intervals (Start/End indices) in str2 that are
Not found in str1"""
return [[b1, b2] for signal, b1, b2 in self.get_str2_partition()
if signal == 0]



def get_str1_matching_intervals(self):
"""Return sub-string intervals (Start/End indices) in str1 that are
found in str2"""
return self.str1_matching_intervals



def get_str2_matching_intervals(self):
"""Return sub-string intervals (Start/End indices) in str2 that are
found in str1"""
return self.str2_matching_intervals



def get_str1_partition(self):
"""Return all sub-string intervals of the form [0/1, Start, End]
in str1 where 0 indicating Diff and 1 Matching"""
if self.str1_matching_intervals: # Non-empty list
return interpolate_partition(
np.array(self.str1_matching_intervals), 0, self.str1_len)
else:
return [[0, 0, self.str1_len - 1]]



def get_str2_partition(self):
"""Return all sub-string intervals of the form [0/1, Start, End]
in str2 where 0 indicating a Diff interval and 1 a Matching interval"""
if self.str2_matching_intervals: # Non-empty list
return interpolate_partition(
np.array(self.str2_matching_intervals), 0, self.str2_len)
else:
return [[0, 0, self.str2_len - 1]]