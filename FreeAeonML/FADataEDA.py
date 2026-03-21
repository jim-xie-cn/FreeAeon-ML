'''
数据探索
1. CFADataDistribution (数据分布)
    --显示直方图、KED图和Q-Q图
    --检查是否为高斯分布
    --比较两个分布是否相私
2. CFADataTest（假设检验）
    --检验是否为平稳序列
    --显示自相关和偏自相关图
3. CFAFitter (数据拟合)
    --linear:线性拟合
    --polynomial:多项式拟合
    --exponential:指数拟合
4. CFACommonStats (计算常见统计量)
    --mean,std,cv
    --skew,kurt_fisher,kurt_pearson,autocorr_lag1
    --min,max,range,iqr

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot,shapiro,kstest,normaltest,chisquare
import seaborn as sns
import json
from scipy.stats import f_oneway,norm, expon
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import grangercausalitytests

class CFADataDistribution:
    def __init__(self):
        pass
    '''
    显示分布图
    '''
    @staticmethod
    def show_dist(ds,bins = 10 ):
        fig, ax =plt.subplots(1,3,constrained_layout=True, figsize=(8, 3))
        ds.hist(ax=ax[0],bins=bins)
        ax[0].set_title("Histogram")
        ds.plot(kind="kde",ax=ax[1])
        ax[1].set_title("KDE")
        probplot(x=ds,dist='norm',plot=ax[2])
        ax[2].set_title("Q-Q")
        plt.show()
   
    '''
    检查是否为高斯分布
    '''
    @staticmethod
    def normal_test(ds , p_value = 0.05):
        ret = {}
        ret['shapiro-wilk'] = {}
        temp = shapiro(ds)
        ret['shapiro-wilk']['is_normal'] = str(temp[1] >= p_value)
        ret['shapiro-wilk']['detail'] = temp
        ret['shapiro-wilk']['describe'] = "Shapiro-Wilk检验，适用于小样本场合（3≤n≤50），受异常值影响较大。"
        ret['kolmogorov-smirnov'] = {}
        temp = kstest(ds,"norm")
        ret['kolmogorov-smirnov']['is_normal'] = str(temp[1] >= p_value)
        ret['kolmogorov-smirnov']['detail'] = temp
        ret['kolmogorov-smirnov']['describe'] = "Kolmogorov-Smirnov检验是一项拟合优度的统计检验。 此测试比较两个分布（在这种情况下，两个分布之一是高斯分布）。 此检验的零假设是，两分布相同（或），两个分布之间没有差异。"
        ret['skewness-kurtosis'] = {}
        temp = normaltest(ds,)
        ret['skewness-kurtosis']['is_normal'] = str(temp[1] >= p_value)
        ret['skewness-kurtosis']['detail'] = temp
        ret['skewness-kurtosis']['describe'] = "DAgostino-Pearson方法使用偏度和峰度测试正态性。 该检验的零假设是，分布是从正态分布中得出的。"
        return ret

    '''
    检查两个分布是否相同
    '''
    @staticmethod
    def dist_test(ds1, ds2 , bins = 50, p_value = 0.05):
        ret = {}
        # F检验（对原始数据）
        ret['f-test'] = {}
        temp = f_oneway(ds1, ds2)
        ret['f-test']['is_same'] = str(temp.pvalue >= p_value)
        ret['f-test']['detail'] = temp
        ret['f-test']['describe'] = "F检验(方差分析)"

        # 卡方检验（先转换为直方图频数）
        hist1, _ = np.histogram(ds1, bins=bins)
        hist2, _ = np.histogram(ds2, bins=bins)

        # 为避免 sum 不一致，归一化 hist2 到与 hist1 相同的总和
        hist2_scaled = hist2 * (hist1.sum() / hist2.sum())

        ret['chis-test'] = {}
        temp = chisquare(f_obs=hist1, f_exp=hist2_scaled)
        ret['chis-test']['is_same'] = str(temp.pvalue >= p_value)
        ret['chis-test']['detail'] = temp
        ret['chis-test']['describe'] = f"卡方检验（将原始数据分为 {bins} 个箱后进行）"

        return ret
    '''
    正态拟合
    '''
    @staticmethod
    def normal_fit(ds_data,bins=100):
        data = pd.Series(ds_data).to_numpy()
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
        
        def normal_pdf(x, mu, sigma):
            return norm.pdf(x, loc=mu, scale=sigma)
        def exponential_pdf(x, lamb):
            return expon.pdf(x, scale=1/lamb)
        
        params_normal, _ = curve_fit(normal_pdf, bin_centers, hist)
        params_exponential, _ = curve_fit(exponential_pdf, bin_centers, hist)
        
        plt.hist(data, bins=bins, density=True, alpha=0.5, color='blue')
        
        x_range = np.linspace(-4, 4, 1000)
        plt.plot(x_range, normal_pdf(x_range, *params_normal), color='red', label='Normal Fit')
        plt.plot(x_range, exponential_pdf(x_range, *params_exponential), color='green', label='Exponential Fit')
        
        plt.title('Fitting Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        
class CFADataTest:

    '''
    稳定性测试，平稳序列可以使用ARIMA
    1.判断是否为白噪声，白噪声满足ARIMA
        -- 平稳、独立和等方差
        -- 自相关和偏自相关函数在所有滞后阶数上接近于零
        -- 白噪声适合使用ARIMA模型进行预测
    2. 满足以下条件，可以使用ARIMA
        #ADF Test result同时小于1%、5%、10%
        #P-value (不变显著性) 接近0。
    '''
    @staticmethod
    def stationarity_test(timeSeries):
        dftest = adfuller(timeSeries)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        del dfoutput['#Lags Used']
        del dfoutput['Number of Observations Used']
        return dfoutput

    '''
    绘制自相关ACF(与不同滞后阶数之间的相关性),和偏自相关性PACF（与特定滞后阶数的相关性）
    1. 自相关系数ACF：x轴表示滞后阶数（lag），y轴表示自相关系数
       -- 自相关性：自相关系数在某个滞后阶数上远离0，说明在这个滞后阶数上存在显著的自相关
       -- 显著性区域：自相关系数超过置信区间的边界，则认为该系数是显著的
       -- 截尾特性：随着滞后阶数的增加，自相关系数逐渐减小并趋于零，这种情况下，时间序列可能是平稳的，可以使用ARMA模型进行建模
       -- 周期性： 在某些滞后阶数上显示出周期性或周期性衰减。
    2. PACF
       -- 偏自相关性：x轴表示滞后阶数（lag），y轴表示偏自相关系数
       -- 显著性区域：如果偏自相关系数超过置信区间的边界，则认为该系数是显著的
       -- 截尾特性： 随着滞后阶数的增加，偏自相关系数逐渐减小并趋于零。这种情况下，时间序列可能是平稳的，可以使用AR模型进行建模
       -- 选择模型阶数：在某个滞后阶数上截尾，而在后续的滞后阶数上迅速衰减至接近零，那么该滞后阶数可能是适合的AR模型的阶数。
    '''
    @staticmethod
    def show_acf_pacf(timeSeries):
        plot_acf(timeSeries).show()
        plot_pacf(timeSeries).show()

    '''
    格兰特因果检验
    ds_result：结果序列（必须为平稳序列）
    ds_source：原因序列（必须为平稳序列）
    maxlag：时间间隔（整数或列表，为整数时，遍历所有的lag）
    返回值：
    1. 最小的p值
    2. 最佳lag
    3. approve_list通过测试的lag
    4. 详细测试结果
    '''
    @staticmethod
    def granger_test(ds_result,ds_source,maxlag,p_value = 0.05):

        df_test = pd.DataFrame()
        df_test['result'] = ds_result
        df_test['source'] = ds_source

        gc_result = grangercausalitytests(df_test[['result', 'source']], maxlag=maxlag,verbose=False )
        min_lag_p = 100
        best_lag = -1
        detail = {}
        approve_list = []
        for lag in gc_result:

            result = gc_result[lag][0]
            detail[lag] = result.copy()

            max_p = 0
            for test in result:
                test_value = result[test][0]
                test_p = result[test][1]
                if test_p > max_p:
                    max_p = test_p

            if max_p > p_value:
                continue
            #if max_p == 0:
            #    continue   
            approve_list.append({'lag':lag,"max p-value":max_p})
            
            if min_lag_p > max_p:
                min_lag_p = max_p
                best_lag = lag

        return min_lag_p, best_lag, approve_list, detail

class CFAFitter:
    '''
    对数据进行拟合
    fitter：拟合方式（linear，polynomial，exponential）
    degree：多项式阶数
    '''
    #linear,polynomial,exponential
    def __init__(self, fitter='polynomial',degree=2):
        self.params = None
        self.fitter = fitter
        self.degree = degree
        if fitter == 'linear':
            self.model_func = self.linear
        elif fitter == 'polynomial':
            self.model_func = self.polynomial
        elif fitter == 'exponential':
            self.model_func = self.exponential
        else:
            raise ValueError("Unsupported model type. Choose 'linear', 'polynomial', or 'exponential'.")

    @staticmethod
    def linear(x, a, b):
        return a * x + b

    @staticmethod
    def polynomial(x, *coeffs):
        return sum(c * x**i for i, c in enumerate(coeffs))

    @staticmethod
    def exponential(x, a, b):
        return a * np.exp(b * x)
    
    def get_fit_data(self,x_data,params):
        return self.model_func(x_data, *params) 

    def get_fit_param(self):
        return self.params

    def fit(self, x_data, y_data):
        if  self.fitter == 'linear':
            self.params, _ = curve_fit(self.linear, x_data, y_data)
            self.model_func = self.linear
        elif  self.fitter == 'polynomial':
            self.params, _ = curve_fit(self.polynomial, x_data, y_data, p0=[1] * (self.degree + 1))
            #self.params, _ = curve_fit(self.polynomial, self.x_data, self.y_data, p0=[1]*degree)
            self.model_func = self.polynomial
        elif  self.fitter == 'exponential':
            self.params, _ = curve_fit(self.exponential, x_data, y_data)
            self.model_func = self.exponential
        else:
            raise ValueError("Unsupported model type. Choose 'linear', 'polynomial', or 'exponential'.")
        return self.params

    def plot(self,x_data,y_data, params = None):
        plt.scatter(x_data, y_data, label='Data', color='blue', s=10)
        if params == None:
            params = self.get_fit_param()
        y_fit = self.get_fit_data(x_data,params)
        plt.plot(x_data, y_fit, label='Fitted Curve', color='red')
        plt.title(f'Fit: {self.model_func.__name__}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

class CFACommonStats:

    @staticmethod
    def get_one_stats(x, q=(0.25, 0.5, 0.75), autocorr_lags=(1,), add_corr=False):
        def corr_safe(df):
            num = df.select_dtypes(include="number").copy()
            num = num.dropna(axis=1, how="all")
            num = num.loc[:, num.nunique(dropna=True) > 1]
            if num.shape[1] < 2:
                return None

            with np.errstate(invalid="ignore", divide="ignore"):
                C = num.corr()
            return C

        def autocorr_safe(s: pd.Series, lag: int):
            s = pd.to_numeric(s, errors="coerce")
            s2 = s.dropna()
            if len(s2) <= lag + 1:
                return np.nan
            if s2.std(ddof=0) == 0:
                return np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                return s2.autocorr(lag=lag)

        def _one(s: pd.Series) -> pd.Series:
            s = s.copy()
            out = {}

            out["count"] = int(s.count())
            out["n"] = int(len(s))
            out["n_missing"] = int(s.isna().sum())
            out["missing_rate"] = (out["n_missing"] / out["n"]) if out["n"] else np.nan
            out["nunique"] = int(s.nunique(dropna=True))

            if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                m = s.mode(dropna=True)
                out["mode"] = m.iloc[0] if len(m) else np.nan
                return pd.Series(out)

            s_num = s.astype("float") if pd.api.types.is_bool_dtype(s) else pd.to_numeric(s, errors="coerce")

            out["mean"] = s_num.mean()
            out["median"] = s_num.median()
            m = s_num.mode(dropna=True)
            out["mode"] = m.iloc[0] if len(m) else np.nan

            out["min"] = s_num.min()
            out["max"] = s_num.max()
            out["range"] = (out["max"] - out["min"]) if pd.notna(out["max"]) and pd.notna(out["min"]) else np.nan

            qs = s_num.quantile(list(q))
            for qi, val in qs.items():
                out[f"q_{qi:g}"] = val
            out["iqr"] = (qs.loc[0.75] - qs.loc[0.25]) if (0.25 in qs.index and 0.75 in qs.index) else np.nan

            out["var"] = s_num.var()
            out["std"] = s_num.std()
            out["cv"] = (out["std"] / out["mean"]) if pd.notna(out["mean"]) and out["mean"] != 0 else np.nan

            out["mad_mean"] = (s_num - out["mean"]).abs().mean()
            out["skew"] = s_num.skew()
            out["kurt_fisher"] = s_num.kurt()
            out["kurt_pearson"] = out["kurt_fisher"] + 3 if pd.notna(out["kurt_fisher"]) else np.nan

            for lag in autocorr_lags:
                out[f"autocorr_lag{lag}"] = autocorr_safe(s_num, lag)

            return pd.Series(out)

        if isinstance(x, pd.Series):
            return _one(x).to_frame().T

        if isinstance(x, pd.DataFrame):
            stats = pd.DataFrame([_one(x[c]).rename(c) for c in x.columns])

            if add_corr:
                C = corr_safe(x)
                if C is not None:
                    corr_long = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack()
                    corr_row = corr_long.rename(lambda idx: f"corr__{idx[0]}__{idx[1]}").to_frame().T
                    corr_row.index = ["__corr__"]
                    stats = pd.concat([stats, corr_row], axis=0, sort=False)

            return stats

        return get_one_stats(pd.Series(x), q=q, autocorr_lags=autocorr_lags)
    
    @staticmethod
    def get_stats(df_data):
        result = {}
        numeric_columns = df_data.select_dtypes(include=['number'])
        for key in numeric_columns:
            result[key] = {}
            t = CFACommonStats.get_one_stats(df_data[key]).to_dict(orient='records')[0]
            result[key]['mean'] = t['mean']
            result[key]['skew'] = t['skew']
            result[key]['kurt_fisher'] = t['kurt_fisher']
            result[key]['kurt_pearson'] = t['kurt_pearson']
            result[key]['autocorr_lag1'] = t['autocorr_lag1']
            result[key]['range'] = t['range']
            result[key]['min'] = t['min']
            result[key]['max'] = t['max']
            result[key]['iqr'] = t['iqr']
            result[key]['std'] = t['std']
            result[key]['cv'] = t['cv']
            
        return result

def main():
    #生成测试数据
    np.random.seed(11)
    X_Normal = np.random.randn(1000)
    X_linear = np.linspace(1,100,1000)
    df_data = pd.DataFrame()
    df_data['Normal'] = pd.Series(X_Normal)
    df_data['Linear'] = pd.Series(X_linear)

    #是否为高斯分布
    test = CFADataDistribution.normal_test(df_data['Normal'])
    print(json.dumps(test,indent=4))
    test = CFADataDistribution.dist_test(df_data['Normal'],df_data['Normal'])
    print(json.dumps(test,indent=4))
    CFADataDistribution.show_dist(df_data['Normal'],bins = 20)

    #平稳性检查
    df_test = CFADataTest.stationarity_test(df_data.Normal)
    print(df_test)
    CFADataTest.show_acf_pacf(df_data.Normal)
    plt.show()

    #数据拟合
    np.random.seed(0)
    x_data = np.linspace(0, 10, 100)
    y_data_linear = 2 * x_data + 1 + np.random.normal(size=x_data.size)
    y_data_polynomial = 1 * x_data**2 - 2 * x_data + 1 + np.random.normal(size=x_data.size)
    y_data_exponential = 2 * np.exp(0.5 * x_data) + np.random.normal(size=x_data.size)

    # 线性拟合
    linear_fitter = CFAFitter(fitter='linear')
    linear_params = linear_fitter.fit(x_data,y_data_linear)
    print("Linear fit parameters:", linear_params)
    linear_fitter.plot(x_data,y_data_linear)

    # 多项式拟合
    poly_fitter = CFAFitter(fitter='polynomial')
    poly_params = poly_fitter.fit(x_data,y_data_polynomial)
    print("Polynomial fit parameters:", poly_params)
    poly_fitter.plot(x_data,y_data_polynomial)

    # 指数拟合
    exp_fitter = CFAFitter(fitter="exponential")
    exp_params = exp_fitter.fit(x_data,y_data_exponential)
    print("Exponential fit parameters:", exp_params)
    exp_fitter.plot(x_data,y_data_exponential)
    plt.show()

    # 计算常见统计量
    stats = CFACommonStats.get_stats(df_data)
    print(stats)


if __name__ == "__main__":
    main()
