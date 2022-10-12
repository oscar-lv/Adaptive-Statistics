import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage.interpolation import shift


from typing import Tuple, List, Dict

df = pd.read_csv('Data_PG_UN.csv').set_index('Date')
rf_df = df.apply(lambda x: x - df['RF'].values) * 100

pg_rf = rf_df['PG']
un_rf = rf_df['UN']

X = pd.concat((rf_df['MKT'], rf_df['HH']), axis=1).values

X = rf_df['MKT'].values

y = pg_rf.values

model = sm.OLS(y, sm.tools.tools.add_constant(X), has_constant=True)
result = model.fit()
result.params
result.summary()
result.conf_int(alpha=0.05)
result.HC0_se


def pretty_print_results(results: Dict):
	print("----------------------")
	print('OLS Regression Results')
	print("----------------------")

	for key, value in results.items():
		if isinstance(value, float):
			print(f'{key}:{value:,.4f}')
		else:
			print(f'{key}:{np.round(value, 3)}')


def multiply(*matrices):
	for i, m in enumerate(matrices):
		if i == 0:
			result = m
		else:
			result = np.matmul(result, m)
	return result


def get_ols_coefficients(X, y, add_const=True, return_x0=False):
	"""
	(a)
	Inputs
	X,y : regression variables

	Output:
	alpha, betas

	"""

	n_obs = y.shape[0]

	# Compute OLS

	if len(X.shape) == 1:
		X = X.reshape(-1, 1)
		n_regressors = 1
	else:
		n_regressors = X.shape[1]
	if add_const:
		const = np.ones(shape=(X.shape[0], 1))

		x_0 = np.hstack([const, X])
	else:
		x_0 = X
	y = y.reshape(-1, 1)

	XXinv = np.linalg.inv(x_0.T.dot(x_0))
	coefs = XXinv.dot(x_0.T).dot(y)

	x_coefs = coefs[1:]
	alpha = coefs[0][0]
	if return_x0:
		return alpha, x_coefs, coefs, n_obs, n_regressors, x_0
	return alpha, x_coefs, coefs, n_obs, n_regressors


def get_information_criteria(loglikelihood, n_obs, n_regressors):
	"""
	(j) Computing AIC, BIC, and Hannah-Quinn IC using loglikelihood

	"""
	bic = (n_regressors + 1) * np.log(n_obs) - 2 * loglikelihood
	aic = 2 * (n_regressors + 1) - 2 * loglikelihood
	hqic = np.log(np.log(n_obs)) * 2 * (n_regressors + 1) - 2 * loglikelihood
	return bic, aic, hqic


def get_white_errors(X, errsq, n_obs):
	"""
	(g-h) Computing the Var-Cov Matrix, std errors under Heteroskedacity
	"""
	x_0 = X

	# Using pseudo inverse matrix
	x_0_pseudo = np.linalg.pinv(x_0)
	scaled_transpose = errsq * x_0_pseudo.T

	# (g) Variance-Covariance Matrix under heteroskedacity
	V_white = x_0_pseudo @ scaled_transpose

	std_white = np.sqrt(np.diag(V_white))
	return std_white


def get_white_test(errsq, X, n_regressors, n_obs):
	"""
	(e) White test and visualisation for heteroskedacity in errors

	"""
	x_0 = X
	plt.plot(errsq)
	plt.xlabel('Observations')
	plt.ylabel('Squared Errors')
	plt.title('Visualisation of error variance')
	plt.show()
	interraction_x = x_0
	new_regressors = []

	# Generating interaction terms for regressors (X^2, X1*X2...)
	if n_regressors > 1:
		for i in (1, n_regressors):
			new_regressors.append((interraction_x[:, i] ** 2).reshape(-1, 1))
		new_regressors.append(interraction_x[:, i].reshape(-1, 1) * interraction_x[:, i - 1].reshape(-1, 1))
	else:
		new_regressors.append((interraction_x[:, 1] ** 2).reshape(-1, 1))
	interraction_x = np.hstack([interraction_x, np.hstack(new_regressors)])

	# Regressing the errors on the interraction matrix
	XIXinv = np.linalg.inv(interraction_x.T.dot(interraction_x))
	white_test_coefs = XIXinv.dot(interraction_x.T).dot(errsq)
	errsq_hat = interraction_x @ white_test_coefs
	err3 = errsq - errsq_hat
	err3_2 = (err3.T @ err3)
	wt_r2 = 1 - (err3_2 / (n_obs)) / np.var(errsq)
	test_statistic = n_obs * wt_r2

	# Test stat is distirbuted Chi^2 2 df
	wt_pv = stats.chi2.sf(test_statistic, 2)
	return test_statistic, wt_pv


def get_jarque_bera_test(err, n_obs, qqplot=True):
	"""
	(l) QQPlot and Jarque Bera test for normality of errors

	"""
	skewness = stats.skew(err, axis=0)
	kurtosis = 3 + stats.kurtosis(err, axis=0)

	jb = (n_obs / 6) * (skewness ** 2 + (1 / 4) * (kurtosis - 3) ** 2)
	p_value = stats.chi2.sf(jb, 2)
	if qqplot:
		err_s = err
		err_s.shape = err_s.shape[0]
		stats.probplot(err, plot=plt)
	return jb, p_value


def get_durbin_watson_test(err):
	"""
	(k) Durbin Watson Statistic

	"""
	shift_resids = np.diff(err, 1, axis=0)
	dw = np.sum(shift_resids ** 2, axis=0) / np.sum(err ** 2, axis=0)
	return dw


def get_breusch_godfrey_test(err):
	"""
	(k) Beursch-Godfrey Statistic, regressing the errors on two lagged period errors, using R-Squared and number of obs.

	"""
	err_s = err
	err_s.shape = err_s.shape[0]
	err_t_1 = shift(err_s, 1, cval=np.NaN)[2:].reshape(-1, 1)
	err_t_2 = shift(err_s, 2, cval=np.NaN)[2:].reshape(-1, 1)
	lagged_errors = np.hstack([err_t_1, err_t_2])
	alpha, x_coefs, _, _, _ = get_ols_coefficients(lagged_errors, err[2:])
	errs = err[2:].reshape(-1, 1) - (lagged_errors @ x_coefs)
	r2_bg = 1 - (errs.T @ errs / (262)) / np.var(err[2:])
	bg = (err.shape[0] - 2) * r2_bg
	return bg


def reset_test(X, y, y_hat, add_const):
	"""
	(n) Checking for evidence against linear model specification

	Regressing y on the usual regressors and adding a higher power of fitted values

	"""
	X_n = np.hstack([X, y_hat ** 2])
	alpha, x_coefs, coefs, n_obs, n_regressors = get_ols_coefficients(X_n, y, add_const)

	y_hat2 = X_n @ coefs
	err = y - y_hat2
	err2 = (err.T @ err)
	variance = err2 / (n_obs - n_regressors)

	# Variance-Covariance matrix (c)
	V = (variance) * (np.linalg.inv(X_n.T @ X_n))
	std = np.sqrt(np.diag(V))

	# Test for significance of the added regressor, transformed y_hat
	t_stats, ci = t_test(coefs, std, n_obs, n_regressors, 0, verbose=False)
	p_value = stats.t.sf(abs(t_stats[-1]), (n_obs - n_regressors)) * 2  # Two tailed p-value for the y_hat regressor
	return t_stats, p_value, coefs


def t_test(betas, se, n_obs, n_regressors, b0, verbose=True):
	"""
	(d) (f) (h) Computing the t-statistics and confidence intervals
	"""
	se = se.reshape(-1, 1)
	t_stats = (betas - b0) / se
	dfreedom = n_obs - n_regressors
	critical_99 = stats.t.ppf(1 - 0.01 / 2, dfreedom)
	critical_95 = stats.t.ppf(1 - 0.05 / 2, dfreedom)
	critical_90 = stats.t.ppf(1 - 0.1 / 2, dfreedom)

	if verbose:

		print('-----T-Test Results-----')
		for i, test in enumerate(t_stats):
			print(f'Regressor {i + 1}, t-test value={np.round(t_stats[i, 0], 2)}, b0={b0}')
			pct1 = abs(t_stats[i]) >= critical_99
			pct5 = abs(t_stats[i]) >= critical_95
			pct10 = abs(t_stats[i]) >= critical_90
			print(f'Accepted at the : 1%: {pct1}, 5%: {pct5}, 10%: {pct10}')

		print('90% CI - lower/upper')
		lower = betas - (critical_90 * se)
		upper = betas + (critical_90 * se)
		ci90 = np.hstack([lower, upper])
		print(np.round(ci90, 5))

		print('95% CI - lower/upper')
		lower = betas - (critical_95 * se)
		upper = betas + (critical_95 * se)
		ci95 = np.hstack([lower, upper])
		print(np.round(ci95, 5))

		print('99% CI - lower/upper')
		lower = betas - (critical_99 * se)
		upper = betas + (critical_99 * se)
		ci99 = np.hstack([lower, upper])
		print(np.round(ci99, 5))

	else:

		for i, test in enumerate(t_stats):
			pct1 = abs(t_stats[i]) >= critical_99
			pct5 = abs(t_stats[i]) >= critical_95
			pct10 = abs(t_stats[i]) >= critical_90

		lower = betas - (critical_90 * se)
		upper = betas + (critical_90 * se)
		ci90 = np.hstack([lower, upper])

		lower = betas - (critical_95 * se)
		upper = betas + (critical_95 * se)
		ci95 = np.hstack([lower, upper])

		lower = betas - (critical_99 * se)
		upper = betas + (critical_99 * se)
		ci99 = np.hstack([lower, upper])

	return t_stats, {'ci90': ci90, 'ci95': ci95, 'ci99': ci99}


def rolling_regression(X, y, window):
	betas = []
	cis = []
	for i in range(0, X.shape[0] - window):
		if len(X.shape) > 1:
			Xw = X[i:i + window, :]
		else:
			Xw = X[i:i + window]
		yw = y[i:i + window].reshape(-1, 1)
		alpha, x_coefs, coefs, n_obs, n_regressors, x_0 = get_ols_coefficients(Xw, yw, add_const=True, return_x0=True)
		y_hat = x_0 @ coefs
		err = yw - y_hat
		err2 = (err.T @ err)
		variance = err2 / (n_obs - n_regressors)
		V = (variance) * (np.linalg.inv(x_0.T @ x_0))
		std = np.sqrt(np.diag(V))
		t_stats, ci = t_test(coefs, std, n_obs, n_regressors, 0, verbose=False)
		betas.append(coefs)
		cis.append(ci)

	x1s = [b[1] for b in betas]
	ci_l = [ci['ci95'][1][0] for ci in cis]
	ci_u = [ci['ci95'][1][1] for ci in cis]

	fig, ax = plt.subplots(1, 1, figsize=(15, 10))
	plt.plot(x1s, label='coef')
	plt.plot(ci_l, '--', label='lower ci', color='lightcoral')
	plt.plot(ci_u, '--', label='upper_ci', color='green')
	plt.legend()
	plt.title('Rolling Regressions - 60 day windows')
	plt.show()

	if len(X.shape) > 1:
		x1s = [b[2] for b in betas]
		ci_l = [ci['ci95'][2][0] for ci in cis]
		ci_u = [ci['ci95'][2][1] for ci in cis]

		fig, ax = plt.subplots(1, 1, figsize=(15, 10))
		plt.plot(x1s, label='coef')
		plt.plot(ci_l, '--', label='lower ci', color='lightcoral')
		plt.plot(ci_u, '--', label='upper_ci', color='green')
		plt.legend()
		plt.title('Rolling Regressions - 2nd regressor - 60 day windows')
		plt.show()


def ols_analysis(X, y, qqplot: bool = True) -> float:
	# # Initialization

	n_obs = y.shape[0]
	if len(X.shape) == 1:
		n_regressors = 1
	else:
		n_regressors = X.shape[1]

	# (a) Compute OLS, r2, adj r2

	if n_regressors == 1:
		X = X.reshape(-1, 1)
	const = np.ones(shape=(X.shape[0], 1))

	x_0 = np.hstack([const, X])
	y = y.reshape(-1, 1)

	XXinv = np.linalg.inv(x_0.T.dot(x_0))
	coefs = XXinv.dot(x_0.T).dot(y)

	x_coefs = coefs[1:]
	alpha = coefs[0][0]

	y_hat = x_0 @ coefs
	plt.scatter(x_0[:, 1], y)
	plt.plot(x_0[:, 1], x_0[:, 1] * coefs[1])
	plt.title('Regression of Market on Stock')
	plt.xlabel('MKT')
	plt.ylabel('Stock')
	plt.show()

	# (d) Standard Errors under homoskedacity
	err = y - y_hat
	err2 = (err.T @ err)
	variance = err2 / (n_obs - n_regressors)

	# Variance-Covariance matrix (c)
	V = (variance) * (np.linalg.inv(x_0.T @ x_0))
	std = np.sqrt(np.diag(V))

	# (a) R2 and LogLikelihood, Information criteria
	r2 = 1 - (err2 / (n_obs)) / np.var(y)
	adj_r2 = 1 - (variance / np.var(y))

	# (d) T-Tests
	t_stats, ci = t_test(coefs, std, n_obs, n_regressors, 0)

	errsq = (err ** 2).reshape(-1, 1)

	# (e) White Test and plot
	white_test_stat, white_test_pv = get_white_test(errsq, x_0, n_regressors, n_obs)

	# (h) Homoskedastic T-Test
	std_white = get_white_errors(x_0, errsq, n_obs)
	print('Robust Standard T-Test')
	t_stats2, ci2 = t_test(coefs, std_white, n_obs, n_regressors, 0)

	# (j) Information Criteria
	loglikelihood = -n_obs * np.log(2 * np.pi) / 2 - n_obs * np.log(variance) / 2 - err2 / (2 * variance)
	bic, aic, hqic = get_information_criteria(loglikelihood, n_obs, n_regressors)

	# (l) Normality of errors
	jarque_bera, jb_pvalue = get_jarque_bera_test(err, n_obs, qqplot=True)

	# (m) Condition Number - Colinearity
	condition_no = np.linalg.cond(X.T @ X)

	# (n) Reset Test for model mispecification
	t_reset, p_y_hat, coefs = reset_test(x_0, y, y_hat, add_const=False)

	results = {
		'alpha': alpha,
		'coeffs': x_coefs,
		'se': std,
		't-stats': t_stats,
		'90% CI': ci['ci90'],
		'95% CI': ci['ci95'],
		'99% CI': ci['ci99'],
		'R-Squared': r2[0, 0],
		'Adj, R-Squared': adj_r2[0, 0],
		'AIC': aic[0, 0],
		'BIC': bic[0, 0],
		'Hannah-Quinn IC': hqic[0, 0],
		'Jarque-Bera': jarque_bera[0],
		'Jarque-Bera Pvalue': jb_pvalue[0],
		'Condition Number': condition_no,
		'Durbin Watson': get_durbin_watson_test(err),  # (k)
		'Breusch-Godfrey': get_breusch_godfrey_test(err)[0, 0],  # (k)
		'White Test Statistic': white_test_stat[0, 0],
		'White Test P-Value': white_test_pv[0, 0],
		'White Errors': std_white,
		'White T-tests': t_stats2,
		'Heteroskedacity 90% CI': ci2['ci90'],
		'Heteroskedacity 95% CI': ci2['ci95'],
		'Heteroskedacity 99% CI': ci2['ci99'],
		'RESET y_hat_coef': coefs[-1][0],
		'RESET yhat prob.': p_y_hat[0]
	}

	return results


result.summary()
result_0 = ols_analysis(X, y)
pretty_print_results(result_0)

rolling_regression(X, y, 60)
# Regression 1 : Capm Market/Unilever

X = rf_df['MKT'].values
y = un_rf.values
result_0 = ols_analysis(X, y)
pretty_print_results(result_0)
model = sm.OLS(y, sm.tools.tools.add_constant(X), has_constant=True)
result = model.fit()
result.params
result.summary()
rolling_regression(X, y, 60)

# Regression 2 : Capm Market/HH/Unilever

X = pd.concat((rf_df['MKT'], rf_df['HH']), axis=1).values
y = un_rf.values
result_0 = ols_analysis(X, y)
pretty_print_results(result_0)
model = sm.OLS(y, sm.tools.tools.add_constant(X), has_constant=True)
result = model.fit()
result.params
result.summary()
rolling_regression(X, y, 60)

# Regression 3 : Capm Market/Procter&Gamble

X = rf_df['MKT'].values
y = pg_rf.values
result_0 = ols_analysis(X, y)
pretty_print_results(result_0)
model = sm.OLS(y, sm.tools.tools.add_constant(X), has_constant=True)
result = model.fit()
result.params
result.summary()
rolling_regression(X, y, 60)
