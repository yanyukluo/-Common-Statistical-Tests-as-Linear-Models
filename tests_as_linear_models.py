import numpy as np
from scipy import stats
import statsmodels.api as sm

np.random.seed(1)

print("1. Student's t-test")

group1 = np.random.normal(70, 10, 20)
group2 = np.random.normal(75, 10, 20)

print("t-test result:")
print(stats.ttest_ind(group1, group2))

y = np.concatenate([group1, group2])
x = np.concatenate([np.zeros(20), np.ones(20)])

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

print("linear model coefficient:")
print(model.params)
print("linear model p-values:")
print(model.pvalues)

print("""
The t-test compares the means of two groups.
In the linear model, x is the group variable.
The coefficient of x is the difference between the two group means.
So the t-test and this linear model are asking almost the same question.
""")

print("2. Mann-Whitney test")

print("Mann-Whitney result:")
print(stats.mannwhitneyu(group1, group2))

rank_y = stats.rankdata(y)

rank_model = sm.OLS(rank_y, X).fit()

print("linear model on ranks coefficient:")
print(rank_model.params)
print("linear model on ranks p-values:")
print(rank_model.pvalues)

print("""
The Mann-Whitney test is similar to comparing two groups,
but it uses ranks instead of the original data.
So I ranked the data first and then used a linear model.
This is why Mann-Whitney is related to the t-test idea.
""")

print("3. Wilcoxon signed-rank test")

before = np.random.normal(70, 10, 20)
after = before + np.random.normal(4, 6, 20)

diff = after - before

print("Wilcoxon result:")
print(stats.wilcoxon(diff))

signed_rank = np.sign(diff) * stats.rankdata(abs(diff))

one = np.ones(20)
wilcox_model = sm.OLS(signed_rank, one).fit()

print("intercept:")
print(wilcox_model.params)
print("p-value:")
print(wilcox_model.pvalues)

print("""
The Wilcoxon signed-rank test is for paired data.
I first calculated the difference between after and before.
Then I used signed ranks of the differences.
The model only has an intercept, so it checks whether the signed ranks
are centered around zero.
""")

print("4. Wald test")

x = np.random.normal(0, 1, 40)
y = 2 + 1.2 * x + np.random.normal(0, 1, 40)

X = sm.add_constant(x)
reg = sm.OLS(y, X).fit()

beta = reg.params[1]
se = reg.bse[1]

wald = (beta / se) ** 2
p = 1 - stats.chi2.cdf(wald, 1)

print("beta:")
print(beta)
print("standard error:")
print(se)
print("Wald statistic:")
print(wald)
print("p-value:")
print(p)

print("""
The Wald test checks whether a coefficient is different from zero.
It divides the coefficient by its standard error.
So it is also about testing a model parameter.
""")

print("""
Conclusion:

These tests are connected because they all test some kind of difference.

Student's t-test compares raw means.
Mann-Whitney compares ranks.
Wilcoxon signed-rank compares signed ranks for paired data.
Wald test checks whether a model coefficient is different from zero.

After reading the article, I understand that these tests are not totally
separate. They can all be explained using the idea of linear models
or testing model coefficients.
""")
