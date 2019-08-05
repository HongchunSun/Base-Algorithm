class Solution:
    @staticmethod
    def minDistance_dp(word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        本思路采用动态规划
        """
        m = len(word1)
        n = len(word2)
        dp = []
        for i in range(m + 1):
            this_tmp = []
            for j in range(n + 1):
                if i == 0:
                    this_tmp.append(j)
                elif j == 0:
                    this_tmp.append(i)
                else:
                    this_tmp.append(False)
            dp.append(this_tmp)
        print(dp)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1], dp[i][j - 1]) + 1
        print(dp)
        return dp[m][n]


if __name__ == "__main__":
    word1 = "sunday"
    word2 = "sunday"
    print(Solution.minDistance_dp(word1, word2))
