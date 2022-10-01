# dp to find min edit distance
def min_edit_distance(word1, word2):
	dp = [[0] * (len(word1) + 1) for _ in range(len(word2) + 1)]
	
	for j in range(1, len(word1) + 1):
		dp[0][j] = 1 + dp[0][j - 1]
	for i in range(1, len(word2) + 1):
		dp[i][0] = 1 + dp[i - 1][0]
	
	
	for j in range(1, len(word1) + 1):
		for i in range(1, len(word2) + 1):
			dp[i][j] = min(dp[i - 1][j] + 1, 
						   dp[i][j - 1] + 1, 
						   dp[i - 1][j - 1] + (0 if word2[i - 1] == word1[j - 1] else 1))
	
	return dp[-1][-1]
